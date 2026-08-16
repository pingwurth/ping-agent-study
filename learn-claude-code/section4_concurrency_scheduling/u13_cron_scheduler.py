#!/usr/bin/env python3
"""
s14: Cron Scheduler — 独立守护线程 + 队列处理器（类 Cron 定时任务调度器）

运行:  python s14_cron_scheduler/code.py
依赖:  pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

相比 s13 的变化:
  - CronJob 数据类 (id, cron, prompt, recurring, durable)
  - cron_matches: 5 字段 cron 表达式匹配，支持 DOM/DOW OR 语义
  - schedule_job / cancel_job: 注册/移除 cron 任务（带验证）
  - cron_scheduler_loop: 独立守护线程，每秒轮询一次
  - cron_queue: 线程安全队列，调度器写入，队列处理器消费
  - queue_processor_loop: 当队列有任务且 agent 空闲时自动唤醒 agent
  - 持久化存储: .scheduled_tasks.json（重启后恢复）
  - 3 个新工具: schedule_cron, list_crons, cancel_cron

四层架构:
  1. 调度器（Scheduler）: 守护线程检查时间 → 触发匹配的任务
  2. 队列（Queue）: cron_queue 解耦调度器和 agent 循环
  3. 队列处理器（Queue processor）: 当队列有任务且 agent 空闲时唤醒 agent
  4. 消费者（Consumer）: agent_loop 消费队列中的任务并注入消息
"""

# 标准库导入
import os, subprocess, json, time, random, threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# readline 用于支持终端行编辑（历史记录、光标移动等）
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

# 第三方库：Anthropic API 客户端和环境变量加载
from anthropic import Anthropic
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(override=True)
# 如果设置了自定义 API 基础 URL，移除可能冲突的 auth token
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# ── 全局配置 ──
WORKDIR = Path.cwd()                          # 工作目录
MEMORY_DIR = WORKDIR / ".memory"              # 记忆存储目录
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"       # 记忆索引文件
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))  # API 客户端
MODEL = os.environ["MODEL_ID"]                # 使用的模型 ID

# ── 任务系统（来自 s12，已同步） ──
# 任务系统提供任务的创建、查询、认领和完成功能
# 每个任务存储为独立的 JSON 文件

TASKS_DIR = WORKDIR / ".tasks"    # 任务文件存储目录
TASKS_DIR.mkdir(exist_ok=True)


@dataclass
class Task:
    """任务数据类 - 表示一个可跟踪的工作单元"""
    id: str                          # 任务唯一标识符
    subject: str                     # 任务标题
    description: str                 # 任务描述
    status: str                      # 状态: pending（待处理） | in_progress（进行中） | completed（已完成）
    owner: str | None                # 任务所有者（谁在执行）
    blockedBy: list[str]             # 依赖的前置任务 ID 列表


def _task_path(task_id: str) -> Path:
    """获取任务文件的存储路径"""
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """创建新任务并保存到文件"""
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",  # 基于时间戳和随机数生成唯一 ID
        subject=subject, description=description,
        status="pending", owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    """将任务保存为 JSON 文件"""
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    """从 JSON 文件加载任务"""
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    """列出所有任务，按文件名排序"""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task(task_id: str) -> str:
    """返回任务的完整详情（JSON 格式）"""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """检查任务的所有前置依赖是否已完成。
    缺失的依赖被视为阻塞状态。"""
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():    # 依赖任务不存在
            return False
        if load_task(dep_id).status != "completed":  # 依赖任务未完成
            return False
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领任务 - 设置所有者并将状态改为进行中"""
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id):
        # 找出未完成的依赖任务
        deps = [d for d in task.blockedBy
                if not _task_path(d).exists() or load_task(d).status != "completed"]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress (owner: {owner})\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    """完成任务 - 将状态改为已完成，并报告新解锁的下游任务"""
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    # 查找因本任务完成而解锁的待处理任务
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")
    return msg


# ── 提示词组装（来自 s10，已同步） ──
# 将系统提示词拆分为多个部分，便于组合和缓存

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",  # 身份定义
    "tools": "Available tools: bash, read_file, write_file, "   # 可用工具列表
             "create_task, list_tasks, get_task, claim_task, complete_task, "
             "schedule_cron, list_crons, cancel_cron.",
    "workspace": f"Working directory: {WORKDIR}",               # 工作目录
    "memory": "Relevant memories are injected below when available.",  # 记忆说明
}


def assemble_system_prompt(context: dict) -> str:
    """组装系统提示词 - 合并各部分为完整提示词"""
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)


# 提示词缓存机制 - 避免重复组装相同的提示词
_last_context_key, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    """获取系统提示词（带缓存）"""
    global _last_context_key, _last_prompt
    # 使用 context 的 JSON 作为缓存键
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        return _last_prompt    # 缓存命中
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


# ── 工具函数 ──
# 这些是 agent 可以调用的工具的具体实现


def safe_path(p: str) -> Path:
    """安全路径解析 - 防止路径遍历攻击（Path Traversal）"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")  # 路径逃逸出工作目录
    return path


def run_bash(command: str, run_in_background: bool = False) -> str:
    """执行 shell 命令（同步模式，120 秒超时）
    run_in_background 由 agent_loop 的调度逻辑处理，不在此处执行"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"  # 截断过长输出
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容，可选限制行数"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件内容（自动创建父目录）"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ── 任务工具的包装函数 ──
# 这些函数将任务系统的 API 包装为 agent 可调用的工具


def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """工具：创建新任务"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """工具：列出所有任务及其状态"""
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        # 状态图标映射
        icon = {"pending": "○", "in_progress": "●",
                "completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}")
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    """工具：获取指定任务的详细信息"""
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    """工具：认领任务（设置所有者为 agent）"""
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    """工具：完成任务"""
    return complete_task(task_id)


# ── 后台任务系统（来自 s13，已同步） ──
# 允许长时间运行的命令在后台执行，不阻塞主循环
# 典型场景：npm install、docker build、测试运行等

_bg_counter = 0                                      # 后台任务计数器
background_tasks: dict[str, dict] = {}               # 正在运行的后台任务
background_results: dict[str, str] = {}              # 已完成任务的结果
background_lock = threading.Lock()                   # 线程锁保护共享状态


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """启发式判断：命令是否可能耗时超过 30 秒"""
    if tool_name != "bash":
        return False
    cmd = tool_input.get("command", "").lower()
    # 已知的慢操作关键词
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(kw in cmd for kw in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """判断是否应该在后台运行（显式请求优先，否则使用启发式）"""
    if tool_input.get("run_in_background"):   # 模型显式请求后台运行
        return True
    return is_slow_operation(tool_name, tool_input)  # 启发式判断


def execute_tool(block) -> str:
    """执行工具调用，返回输出结果"""
    # 工具名称到处理函数的映射表
    handler = {
        "bash": run_bash, "read_file": run_read, "write_file": run_write,
        "create_task": run_create_task, "list_tasks": run_list_tasks,
        "get_task": run_get_task, "claim_task": run_claim_task,
        "complete_task": run_complete_task,
        "schedule_cron": run_schedule_cron, "list_crons": run_list_crons,
        "cancel_cron": run_cancel_cron,
    }.get(block.name)
    if handler:
        return handler(**block.input)
    return f"Unknown tool: {block.name}"


def start_background_task(block) -> str:
    """在守护线程中运行工具，返回后台任务 ID"""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        """后台工作线程 - 执行工具并存储结果"""
        result = execute_tool(block)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running",           # 状态: running → completed
        }
    threading.Thread(target=worker, daemon=True).start()
    print(f"  \033[33m[background] dispatched {bg_id}: {cmd[:40]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    """收集已完成的后台任务结果，返回为任务通知消息列表"""
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)          # 移除已完成的任务
            output = background_results.pop(bg_id, "")  # 获取并移除结果
        summary = output[:200] if len(output) > 200 else output
        # 构建 XML 格式的任务通知
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
        print(f"  \033[32m[background done] {bg_id}: "
              f"{task['command'][:40]} ({len(output)} chars)\033[0m")
    return notifications


# ── Cron 调度器（s14 新增功能） ──
# 这是本单元的核心：实现类 Unix cron 的定时任务调度器
# 支持 5 字段 cron 表达式、持久化存储、后台守护线程

DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"  # 持久化任务存储路径


@dataclass
class CronJob:
    """Cron 任务数据类 - 表示一个定时任务"""
    id: str               # 任务唯一标识符
    cron: str             # cron 表达式，如 "0 9 * * *"（每天 9:00）
    prompt: str           # 触发时注入的消息内容
    recurring: bool       # True = 循环执行，False = 一次性执行
    durable: bool         # True = 持久化到磁盘，False = 仅当前会话


scheduled_jobs: dict[str, CronJob] = {}   # 已注册的任务（id → CronJob）
cron_queue: list[CronJob] = []            # 已触发待消费的任务队列
cron_lock = threading.Lock()              # 保护 scheduled_jobs 和 cron_queue 的线程锁
agent_lock = threading.Lock()             # 保护 agent 会话的线程锁
_last_fired: dict[str, str] = {}          # 防止同一分钟内重复触发: job_id → "YYYY-MM-DD HH:MM"


def _cron_field_matches(field: str, value: int) -> bool:
    """匹配单个 cron 字段与值是否符合

    支持的格式：
    - "*"      : 匹配任意值
    - "*/N"    : 每 N 个单位（如 */5 表示每 5 分钟）
    - "1,3,5"  : 逗号分隔的多个值
    - "1-5"    : 范围（包含两端）
    - "3"      : 精确值
    """
    if field == "*":
        return True                          # 通配符：匹配任意值
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0  # 步进匹配
    if "," in field:
        # 递归匹配逗号分隔的每个值
        return any(_cron_field_matches(f.strip(), value)
                   for f in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)   # 范围匹配
    return value == int(field)               # 精确值匹配


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """检查 5 字段 cron 表达式是否匹配给定的时间

    Cron 表达式格式: 分 时 日 月 星期
    标准 cron 语义：日和星期字段使用 OR 逻辑（两者都受限时任一匹配即可）

    示例:
    - "0 9 * * *"     → 每天 9:00
    - "*/5 * * * *"   → 每 5 分钟
    - "0 0 * * 1"     → 每周一 0:00
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7  # Python: Monday=0 → cron: Sunday=0

    # 匹配各个字段
    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    # 分钟、小时、月份必须全部匹配
    if not (m and h and month_ok):
        return False
    # 日和星期：如果都受限，任一匹配即可（OR 语义）
    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True                    # 都是通配符，已匹配
    if dom_unconstrained:
        return dow_ok                  # 只有星期受限
    if dow_unconstrained:
        return dom_ok                  # 只有日受限
    return dom_ok or dow_ok            # 两者都受限，OR 逻辑


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """验证单个 cron 字段的值是否在有效范围 [lo, hi] 内

    返回 None 表示有效，返回错误消息字符串表示无效
    """
    if field == "*":
        return None                           # 通配符总是有效
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None
    if "," in field:
        # 递归验证逗号分隔的每个部分
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err: return err
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"Invalid range: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    val = int(field)
    if val < lo or val > hi:
        return f"Value {val} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """验证完整的 cron 表达式是否有效

    返回 None 表示有效，返回错误消息字符串表示无效
    5 个字段的有效范围：分(0-59) 时(0-23) 日(1-31) 月(1-12) 星期(0-6)
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    # 每个字段的有效范围
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for i, (field, (lo, hi), name) in enumerate(zip(fields, bounds, names)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    """将持久化任务保存到 .scheduled_tasks.json 文件"""
    durable = [asdict(j) for j in scheduled_jobs.values() if j.durable]
    DURABLE_PATH.write_text(json.dumps(durable, indent=2))


def load_durable_jobs():
    """启动时从磁盘加载持久化任务"""
    if not DURABLE_PATH.exists():
        return
    try:
        jobs = json.loads(DURABLE_PATH.read_text())
        for j in jobs:
            job = CronJob(**j)
            # 验证加载的 cron 表达式是否仍然有效
            err = validate_cron(job.cron)
            if err:
                print(f"  \033[31m[cron] skipping invalid job {job.id}: {err}\033[0m")
                continue
            scheduled_jobs[job.id] = job
        valid = [j for j in jobs if j["id"] in scheduled_jobs]
        if valid:
            print(f"  \033[35m[cron] loaded {len(valid)} durable job(s)\033[0m")
    except Exception:
        pass


def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    """注册新的 cron 任务

    参数:
        cron: 5 字段 cron 表达式
        prompt: 触发时注入的消息
        recurring: 是否循环执行
        durable: 是否持久化到磁盘

    返回: CronJob 对象（成功）或错误消息字符串（失败）
    """
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",  # 随机 6 位数 ID
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable,
    )
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()   # 持久化到磁盘
    print(f"  \033[35m[cron register] {job.id} '{cron}' → {prompt[:40]}\033[0m")
    return job


def cancel_job(job_id: str) -> str:
    """取消 cron 任务"""
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()   # 更新持久化文件
    print(f"  \033[31m[cron cancel] {job_id}\033[0m")
    return f"Cancelled {job_id}"


def cron_scheduler_loop():
    """独立守护线程：每秒轮询一次，触发匹配的任务

    这是调度器的核心循环：
    1. 每秒检查一次当前时间
    2. 遍历所有已注册的任务
    3. 如果 cron 表达式匹配且未在同一分钟内触发，则加入队列
    4. 一次性任务触发后自动移除

    单个任务的错误会被捕获，防止一个坏任务杀死整个调度线程
    """
    while True:
        time.sleep(1)
        now = datetime.now()
        # 使用日期感知的标记，防止每日任务在第 2 天之后被跳过
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        # 检查是否在同一分钟内已触发过
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)          # 加入待消费队列
                            _last_fired[job.id] = minute_marker
                            print(f"  \033[35m[cron fire] {job.id} → "
                                  f"{job.prompt[:40]}\033[0m")
                        # 一次性任务触发后自动移除
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:
                    print(f"  \033[31m[cron error] {job.id}: {e}\033[0m")


def consume_cron_queue() -> list[CronJob]:
    """消费已触发的任务队列（由 agent_loop 调用）"""
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


def has_cron_queue() -> bool:
    """检查是否有待消费的已触发任务"""
    with cron_lock:
        return bool(cron_queue)


# 启动时加载持久化任务，然后启动调度器守护线程
load_durable_jobs()
threading.Thread(target=cron_scheduler_loop, daemon=True).start()
print("  \033[35m[cron] scheduler thread started\033[0m")


# ── Cron 工具的包装函数 ──
# 将 cron 调度器的 API 包装为 agent 可调用的工具


def run_schedule_cron(cron: str, prompt: str,
                      recurring: bool = True, durable: bool = True) -> str:
    """工具：注册新的 cron 定时任务"""
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' → {prompt}"


def run_list_crons() -> str:
    """工具：列出所有已注册的 cron 任务"""
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs. Use schedule_cron to add one."
    lines = []
    for j in jobs:
        tag = "recurring" if j.recurring else "one-shot"    # 循环/一次性
        dur = "durable" if j.durable else "session"         # 持久化/会话级
        lines.append(f"  {j.id}: '{j.cron}' → {j.prompt[:40]} "
                     f"[{tag}, {dur}]")
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    """工具：取消指定的 cron 任务"""
    return cancel_job(job_id)


# ── 工具定义（传递给 Claude API 的工具 schema） ──
# 这些定义告诉模型有哪些工具可用，以及每个工具的参数格式

TOOLS = [
    # 基础文件操作工具
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {
                          "command": {"type": "string"},
                          "run_in_background": {"type": "boolean"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    # 任务管理工具
    {"name": "create_task",
     "description": "Create a new task with optional blockedBy dependencies.",
     "input_schema": {"type": "object",
                      "properties": {
                          "subject": {"type": "string"},
                          "description": {"type": "string"},
                          "blockedBy": {"type": "array",
                                        "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks",
     "description": "List all tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "get_task",
     "description": "Get full details of a specific task by ID.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task",
     "description": "Claim a pending task. Sets owner, changes status to in_progress.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task",
     "description": "Complete an in-progress task. Reports unblocked downstream tasks.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    # Cron 定时任务工具（s14 新增）
    {"name": "schedule_cron",
     "description": "Schedule a cron job. cron is 5-field: min hour dom month dow.",
     "input_schema": {"type": "object",
                      "properties": {
                          "cron": {"type": "string",
                                   "description": "5-field cron expression"},
                          "prompt": {"type": "string",
                                     "description": "Message to inject when fired"},
                          "recurring": {"type": "boolean",
                                        "description": "True=recurring, False=one-shot"},
                          "durable": {"type": "boolean",
                                      "description": "True=persist to disk"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons",
     "description": "List all registered cron jobs.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "cancel_cron",
     "description": "Cancel a cron job by ID.",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
]


# ── 上下文管理 ──
# 从实际状态派生上下文信息，用于组装系统提示词


def update_context(context: dict, messages: list) -> dict:
    """从实际状态派生上下文信息"""
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": [t["name"] for t in TOOLS],  # 可用工具列表
        "workspace": str(WORKDIR),                     # 工作目录
        "memories": memories,                          # 记忆内容
    }


# ── Agent 主循环（简化版，聚焦 cron 调度器） ──
# 教学代码保持基本的 agent 循环。s11 的完整错误恢复已省略。
# cron_scheduler_loop 产生工作；queue_processor_loop 在队列有任务且
# agent 空闲时唤醒此循环。

def agent_loop(messages: list, context: dict) -> dict:
    """Agent 主循环 - 处理用户输入、调用工具、管理 cron 任务

    流程:
    1. 消费已触发的 cron 任务并注入消息
    2. 调用 Claude API 获取响应
    3. 如果响应包含工具调用，执行工具并继续循环
    4. 如果响应是最终文本，返回上下文
    """
    system = get_system_prompt(context)
    while True:
        # 第 4 层：消费已触发的 cron 任务 → 注入为用户消息
        fired = consume_cron_queue()
        for job in fired:
            messages.append({"role": "user",
                             "content": f"[Scheduled] {job.prompt}"})
            print(f"  \033[35m[inject cron] {job.prompt[:50]}\033[0m")

        try:
            # 调用 Claude API
            response = client.messages.create(
                model=MODEL, system=system, messages=messages,
                tools=TOOLS, max_tokens=8000)
        except Exception as e:
            messages.append({"role": "assistant", "content": [
                {"type": "text",
                 "text": f"[Error] {type(e).__name__}: {e}"}]})
            return context

        # 将助手响应添加到消息历史
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return context   # 模型不再调用工具，循环结束

        # 处理工具调用
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            # 判断是否应该在后台运行
            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"[Background task {bg_id} started] "
                                           f"Result will be available when complete."})
            else:
                # 同步执行工具
                output = execute_tool(block)
                print(str(output)[:300])
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})

        # 合并工具结果和后台通知为一条用户消息
        user_content = list(results)
        bg_notifications = collect_background_results()
        if bg_notifications:
            for notif in bg_notifications:
                user_content.append({"type": "text", "text": notif})
        messages.append({"role": "user", "content": user_content})
        # 更新上下文（可能包含新的记忆）
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 会话状态 ──
session_history: list = []                    # 消息历史
session_context = update_context({}, [])      # 当前上下文


def print_latest_assistant_text(messages: list):
    """打印最新助手消息中的文本块"""
    if not messages:
        return
    msg = messages[-1]
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return
    content = msg.get("content", "")
    if isinstance(content, str):
        print(content)
        return
    for block in content:
        if getattr(block, "type", None) == "text":
            print(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""))


def run_agent_turn_locked(user_query: str | None = None):
    """运行一轮 agent 交互（调用者必须持有 agent_lock）"""
    global session_context
    if user_query is not None:
        session_history.append({"role": "user", "content": user_query})
    session_context = agent_loop(session_history, session_context)
    session_context = update_context(session_context, session_history)
    print_latest_assistant_text(session_history)
    print()


def queue_processor_loop():
    """队列处理器循环 - 当 agent 空闲时自动消费已触发的 cron 任务

    这是四层架构中的第 3 层：
    - 每 0.2 秒检查一次队列
    - 如果有任务且 agent 空闲，获取锁并执行
    - 使用 non-blocking acquire 避免阻塞用户交互
    """
    global session_context
    while True:
        time.sleep(0.2)
        if not has_cron_queue():
            continue                          # 队列为空，继续等待
        if not agent_lock.acquire(blocking=False):
            continue                          # agent 忙碌，跳过本轮
        try:
            if not has_cron_queue():
                continue                      # 双重检查
            print("\n  \033[35m[queue processor] delivering scheduled work\033[0m")
            run_agent_turn_locked()
        finally:
            agent_lock.release()


# ── 主程序入口 ──
if __name__ == "__main__":
    print("s14: cron scheduler")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    # 启动队列处理器守护线程
    threading.Thread(target=queue_processor_loop, daemon=True).start()
    print("  \033[35m[queue processor] started\033[0m")
    # 主循环：接收用户输入
    while True:
        try:
            query = input("\033[36ms14 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        with agent_lock:
            run_agent_turn_locked(query)
