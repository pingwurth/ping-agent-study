#!/usr/bin/env python3
"""
s12: Background Tasks — thread-based async execution + notification injection.
s12: 后台任务 — 基于线程的异步执行 + 通知注入机制。

Run:  python s12_background_tasks/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s12 / 相比 s12 的变更:
  - threading.Thread for background execution
    使用 threading.Thread 实现后台执行
  - background_tasks dict for lifecycle tracking (bg_id, command, status)
    background_tasks 字典用于生命周期追踪
  - background_results dict + threading.Lock for thread-safe storage
    background_results 字典 + threading.Lock 实现线程安全存储
  - should_run_background: model explicit request via run_in_background param
    should_run_background: 模型通过 run_in_background 参数显式请求后台执行
  - is_slow_operation: fallback heuristic when model doesn't specify
    is_slow_operation: 模型未指定时的启发式回退判断
  - start_background_task: dispatch to daemon thread, return bg task id
    start_background_task: 分派到守护线程，返回后台任务 ID
  - collect_background_results: gather completed, return as notifications
    collect_background_results: 收集已完成任务，以通知形式返回
  - agent_loop: slow ops → background + placeholder, inject notifications
    agent_loop: 慢操作 → 后台执行 + 占位符，注入通知
  - Notifications use <task_notification> format, not reused tool_use_id
    通知使用 <task_notification> 格式，不再复用 tool_use_id

Note: Teaching code keeps a basic agent loop to stay focused on background
tasks. S11's full error recovery (RecoveryState, backoff, escalation,
reactive compact, fallback model) is omitted.
注意：教学代码保持基础 agent 循环，聚焦后台任务主题。
S11 的完整错误恢复（RecoveryState、退避、升级、响应式压缩、回退模型）在此省略。
"""

import os, subprocess, json, time, random, threading  # threading: 本课核心，用于后台任务
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
# 自定义 base_url 时清除 auth token，避免冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# ── 全局配置 ──
WORKDIR = Path.cwd()                          # 工作目录
MEMORY_DIR = WORKDIR / ".memory"              # 记忆存储目录
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"       # 记忆索引文件
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]                # 模型 ID，从环境变量读取

# ── Task System (from s12, synced) / 任务系统 ──
# 任务持久化到 .tasks/ 目录，每个任务一个 JSON 文件
# 支持依赖关系（blockedBy）和状态流转（pending → in_progress → completed）

TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)


@dataclass
class Task:
    """任务数据模型 / Task data model"""
    id: str               # 任务唯一 ID
    subject: str          # 任务标题
    description: str      # 任务描述
    status: str           # 状态: pending | in_progress | completed
    owner: str | None     # 认领者（谁在执行）
    blockedBy: list[str]  # 依赖的前置任务 ID 列表


def _task_path(task_id: str) -> Path:
    """获取任务 JSON 文件路径 / Get task file path"""
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """创建新任务并持久化 / Create and persist a new task"""
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject, description=description,
        status="pending", owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    """保存任务到 JSON 文件 / Save task to JSON file"""
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    """从 JSON 文件加载任务 / Load task from JSON file"""
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    """列出所有任务（按 ID 排序）/ List all tasks sorted by ID"""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task(task_id: str) -> str:
    """Return full task details as JSON. / 返回任务完整详情（JSON 格式）"""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """Check if all blockedBy dependencies are completed.
    检查所有前置依赖是否已完成。缺失的依赖视为阻塞。
    Missing dependencies are treated as blocked."""
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False                       # 依赖任务不存在 → 阻塞
        if load_task(dep_id).status != "completed":
            return False                       # 依赖任务未完成 → 阻塞
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领任务：设置 owner，状态变为 in_progress
    Claim task: set owner, change status to in_progress"""
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id):
        deps = [d for d in task.blockedBy
                if not _task_path(d).exists() or load_task(d).status != "completed"]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress (owner: {owner})\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    """完成任务，并报告被解锁的下游任务
    Complete task and report newly unblocked downstream tasks"""
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    # 检查哪些 pending 任务现在可以开始了
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")
    return msg


# ── Prompt Assembly (from s10, synced) / 提示词组装 ──
# 将系统提示词拆分为模块化段落，按需组合

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, "
             "create_task, list_tasks, get_task, claim_task, complete_task.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    """组装系统提示词：身份 + 工具 + 工作区 + 记忆
    Assemble system prompt: identity + tools + workspace + memories"""
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)


# 缓存机制：context 不变时直接返回上次的 prompt，避免重复组装
_last_context_key, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    """带缓存的系统提示词获取 / Get system prompt with caching"""
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        return _last_prompt                           # 缓存命中，直接返回
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


# ── Tools / 工具定义 ──
# 定义 agent 可调用的工具：文件操作 + 任务管理

def safe_path(p: str) -> Path:
    """路径安全检查：防止路径逃逸出工作目录
    Path safety check: prevent path traversal outside workspace"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str, run_in_background: bool = False) -> str:
    """执行 shell 命令（同步模式，超时 120s）
    Execute shell command (sync mode, 120s timeout)
    注意：run_in_background 参数由 agent_loop 层处理，此处不处理"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容，可选限制行数 / Read file with optional line limit"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件内容（自动创建父目录）/ Write file (auto-create parent dirs)"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ── Task tools / 任务管理工具 ──
# 以下工具封装任务系统的 CRUD 操作，供 agent 调用

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """创建任务工具 / Create task tool"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """列出所有任务工具 / List all tasks tool"""
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        icon = {"pending": "○", "in_progress": "●",
                "completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}")
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    """获取任务详情工具 / Get task details tool"""
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    """认领任务工具 / Claim task tool"""
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    """完成任务工具 / Complete task tool"""
    return complete_task(task_id)


# 工具定义列表（发送给 Claude API 的 tools 参数）
# 注意 bash 工具包含 run_in_background 可选参数，模型可以显式请求后台执行
TOOLS = [
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
]

# 工具名称 → 处理函数的映射表
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "create_task": run_create_task, "list_tasks": run_list_tasks,
    "get_task": run_get_task, "claim_task": run_claim_task,
    "complete_task": run_complete_task,
}


# ── Background Tasks (s12 new) / 后台任务系统（本课核心）──
# 核心思想：慢操作不阻塞 agent 循环，而是分派到后台线程执行，
#          完成后以 <task_notification> 格式注入到下一轮对话中。

_bg_counter = 0                                      # 后台任务全局计数器
background_tasks: dict[str, dict] = {}               # bg_id → {tool_use_id, command, status}
background_results: dict[str, str] = {}              # bg_id → 输出结果
background_lock = threading.Lock()                   # 线程锁：保护共享数据的并发访问


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """启发式判断：命令是否可能耗时 > 30s
    Fallback heuristic: commands likely to take > 30s.
    当模型未显式指定 run_in_background 时，用关键词匹配做兜底判断。"""
    if tool_name != "bash":
        return False
    cmd = tool_input.get("command", "").lower()
    # 慢操作关键词列表：安装、构建、测试、部署等
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(kw in cmd for kw in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """判断是否应后台执行：模型显式请求优先，否则回退到启发式
    Model explicit request takes priority; fallback to heuristic."""
    if tool_input.get("run_in_background"):
        return True                                # 模型显式请求后台执行
    return is_slow_operation(tool_name, tool_input)  # 回退到启发式判断


def execute_tool(block) -> str:
    """执行工具调用，返回输出 / Execute a tool call block, return output."""
    handler = TOOL_HANDLERS.get(block.name)
    if handler:
        return handler(**block.input)
    return f"Unknown tool: {block.name}"


def start_background_task(block) -> str:
    """在守护线程中运行工具，返回后台任务 ID
    Run tool in a daemon thread. Returns background task ID.

    流程 / Flow:
    1. 生成唯一 bg_id
    2. 在 background_tasks 中注册（状态: running）
    3. 启动守护线程执行工具
    4. 线程完成后更新状态和结果（线程安全）"""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        """线程工作函数：执行工具并安全地存储结果"""
        result = execute_tool(block)
        with background_lock:                      # 加锁更新共享状态
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running",
        }
    # daemon=True: 主进程退出时线程自动终止，不会挂起
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(f"  \033[33m[background] dispatched {bg_id}: {cmd[:40]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    """收集已完成的后台任务结果，以 <task_notification> 格式返回
    Collect completed background results as task_notification messages.

    这是通知注入机制的核心：
    - 遍历 background_tasks 找到 status == "completed" 的任务
    - 从字典中移除已收集的任务（pop）
    - 生成 XML 格式的通知文本，后续注入到 user 消息中"""
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)       # 移除已完成任务
            output = background_results.pop(bg_id, "")  # 取出结果
        summary = output[:200] if len(output) > 200 else output
        # 生成 <task_notification> XML 格式通知
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


# ── Context / 上下文管理 ──
# 从实际状态派生上下文信息，用于组装系统提示词

def update_context(context: dict, messages: list) -> dict:
    """从实际状态派生上下文 / Derive context from real state.
    包含：已启用工具列表、工作区路径、记忆内容"""
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": list(TOOL_HANDLERS.keys()),
        "workspace": str(WORKDIR),
        "memories": memories,
    }


# ── Agent Loop (simplified, focused on background tasks) / Agent 主循环 ──
# 简化版 agent 循环，聚焦后台任务机制
# S11 的完整错误恢复在此省略，详见模块顶部说明

def agent_loop(messages: list, context: dict):
    """Agent 主循环：调用模型 → 执行工具 → 注入结果/通知 → 循环
    Main agent loop: call model → execute tools → inject results/notifications → loop

    后台任务的关键流程 / Background task flow:
    1. 模型返回 tool_use 时，检查 should_run_background()
    2. 如果需要后台执行 → start_background_task() 分派到线程，返回占位符
    3. 如果不需要 → 同步执行 execute_tool()
    4. 每轮结束后 collect_background_results() 收集已完成的后台任务
    5. 将工具结果 + 后台通知一起注入到 user 消息中"""
    system = get_system_prompt(context)
    while True:
        # 调用 Claude API
        try:
            response = client.messages.create(
                model=MODEL, system=system, messages=messages,
                tools=TOOLS, max_tokens=8000)
        except Exception as e:
            messages.append({"role": "assistant", "content": [
                {"type": "text",
                 "text": f"[Error] {type(e).__name__}: {e}"}]})
            return

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return                               # 模型不再调用工具，循环结束

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            # 核心分派逻辑：后台执行 vs 同步执行
            if should_run_background(block.name, block.input):
                # ── 后台路径：分派到线程，返回占位符 ──
                bg_id = start_background_task(block)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"[Background task {bg_id} started] "
                                           f"Command: {block.input.get('command', '')}. "
                                           f"Result will be available when complete."})
            else:
                # ── 同步路径：直接执行并返回结果 ──
                output = execute_tool(block)
                print(str(output)[:300])
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})

        # 注入工具结果 + 后台通知到同一条 user 消息中
        # 这样模型可以在下一轮看到后台任务的完成情况
        user_content = list(results)
        bg_notifications = collect_background_results()
        if bg_notifications:
            for notif in bg_notifications:
                user_content.append({"type": "text", "text": notif})
            print(f"  \033[32m[inject] {len(bg_notifications)} background "
                  f"notification(s)\033[0m")
        messages.append({"role": "user", "content": user_content})
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 主入口 / Main Entry Point ──
if __name__ == "__main__":
    print("s12: background tasks / 后台任务")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []
    context = update_context({}, [])
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        context = update_context(context, history)
        # 打印模型最后一条回复的文本内容
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))
        print()
