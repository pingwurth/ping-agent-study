"""
s14: 任务系统 — 基于文件持久化的任务图，支持 blockedBy 依赖关系。

运行:  python s14_task_system/code.py
依赖:  pip install anthropic python-dotenv + .env 中配置 ANTHROPIC_API_KEY

相比 s11 的变更:
  - Task dataclass（包含 id, subject, description, status, owner, blockedBy）
  - TASKS_DIR = .tasks/ 用于持久化 JSON 存储
  - create_task / save_task / load_task / list_tasks / get_task 函数
  - can_start: 检查 blockedBy 依赖是否全部完成（缺失依赖 = 被阻塞）
  - claim_task: 设置 owner 并将状态从 pending 转为 in_progress
  - complete_task: 标记完成并报告下游被解除阻塞的任务
  - 新增 5 个工具: create_task, list_tasks, get_task, claim_task, complete_task

注意: 教学代码保持了基本的 agent loop，聚焦于任务系统演示。
s11 的完整错误恢复机制（RecoveryState, backoff, escalation,
reactive compact, fallback model）在此省略 — 在真实的 Claude Code 中，
tasks.ts 和 withRetry 是独立的层，可以自然组合。
"""
import os, json, subprocess
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from common import create_client
from dotenv import load_dotenv

# 尝试导入 readline 以支持终端行编辑（如历史记录、光标移动）
try:
    import readline
    # 禁用 bind-tty-special-chars，避免某些终端环境下快捷键冲突
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

# 加载 .env 环境变量（override=True 强制覆盖已存在的变量）
load_dotenv(override=True)
# 如果设置了自定义 base_url，移除 auth_token 以避免冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# ── 基础路径配置 ──
WORKDIR = Path.cwd()                              # 工作目录
MEMORY_DIR = WORKDIR / ".memory"                   # 记忆存储目录
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"            # 记忆索引文件
client, MODEL = create_client()                    # 初始化 API 客户端和模型名

# ── 任务系统 (Task System) ──
# 任务以 JSON 文件形式持久化存储在 .tasks/ 目录中
TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)


@dataclass
class Task:
    """任务数据结构 — 支持依赖关系的有向无环图(DAG)节点"""
    id: str                    # 任务唯一标识（时间戳+随机数生成）
    subject: str               # 任务标题（简短描述）
    description: str           # 任务详细描述
    status: str                # 状态: pending | in_progress | completed
    owner: str | None          # 任务所有者（多 agent 场景下的分配标识）
    blockedBy: list[str]       # 依赖的任务 ID 列表（前置任务必须先完成）


def _task_path(task_id: str) -> Path:
    """根据任务 ID 获取对应的 JSON 文件路径"""
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """创建新任务并持久化到文件系统。
    ID 格式: task_{时间戳}_{4位随机数}，确保唯一性。"""
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject,
        description=description,
        status="pending",       # 新任务默认状态为"待处理"
        owner=None,             # 初始无所有者
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    """将任务序列化为 JSON 并写入文件（持久化存储）"""
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    """从文件反序列化并加载任务对象"""
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    """列出所有任务，按文件名排序（即按创建时间排序）"""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task(task_id: str) -> str:
    """返回任务的完整详情（JSON 格式）"""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """检查任务是否可以开始执行 — 所有 blockedBy 依赖必须已完成。
    缺失的依赖任务被视为阻塞状态（安全保守策略）。"""
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        # 依赖文件不存在 → 阻塞
        if not _task_path(dep_id).exists():
            return False
        # 依赖任务未完成 → 阻塞
        if load_task(dep_id).status != "completed":
            return False
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领任务：设置所有者并将状态从 pending 转为 in_progress。
    前置条件：任务必须是 pending 状态且所有依赖已满足。"""
    task = load_task(task_id)
    # 只有 pending 状态的任务才能被认领
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    # 检查依赖是否满足
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
    """完成任务：标记为 completed 并报告下游被解除阻塞的任务。
    这是任务依赖链传播的核心 — 完成一个任务可能解锁多个后续任务。"""
    task = load_task(task_id)
    # 只有 in_progress 状态的任务才能被标记为完成
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    # 查找因本任务完成而被解除阻塞的下游任务
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")
    return msg


# ── 提示词组装 (Prompt Assembly) ──
# 从 s10 同步过来的模块化提示词系统

# 提示词各段落的定义（按模块拆分，便于维护）
PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, "
             "create_task, list_tasks, get_task, claim_task, complete_task.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    """组装 system prompt — 拼接身份、工具列表、工作区和记忆等段落。"""
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    # 如果有相关记忆，注入到提示词中
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)


# 缓存机制：避免重复组装相同的 system prompt（节省 token 开销）
_last_context_key, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    """获取带缓存的 system prompt — context 未变时直接返回缓存值。"""
    global _last_context_key, _last_prompt
    # 将 context 序列化为字符串作为缓存 key
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


# ── 基础工具 (Tools) ──

def safe_path(p: str) -> Path:
    """路径安全检查 — 防止路径遍历攻击（Path Traversal）。
    确保解析后的路径仍在工作目录内。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令 — 限制超时 120 秒，输出截断至 50000 字符。"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容 — 可选限制行数（用于大文件的分页读取）。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件内容 — 自动创建父目录（mkdir -p 语义）。"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ── 任务工具包装器 (Task Tool Wrappers) ──
# 这些函数是 LLM 可调用的工具入口，包装底层任务函数并添加日志输出

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """工具: 创建任务 — 支持设置依赖关系"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """工具: 列出所有任务 — 用图标区分状态（○ 待处理 / ● 进行中 / ✓ 已完成）"""
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
    """工具: 获取任务详情（JSON 格式）"""
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    """工具: 认领任务（默认 owner 为 "agent"）"""
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    """工具: 完成任务（自动报告下游解除阻塞的任务）"""
    return complete_task(task_id)


# ── 工具定义 (Tool Definitions for LLM) ──
# 这些 JSON Schema 定义会传给 LLM，让它知道有哪些工具可用以及如何调用
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
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

# 工具名称 → 处理函数的映射（调度器核心）
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "create_task": run_create_task, "list_tasks": run_list_tasks,
    "get_task": run_get_task, "claim_task": run_claim_task,
    "complete_task": run_complete_task,
}


# ── 上下文管理 (Context) ──

def update_context(context: dict, messages: list) -> dict:
    """从实际状态派生上下文 — 包含已启用工具、工作区路径和相关记忆。"""
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


# ── Agent 主循环 (简化版，聚焦任务系统演示) ──

def agent_loop(messages: list, context: dict):
    """Agent 核心循环：发送请求 → 处理工具调用 → 循环直到 LLM 停止调用工具。
    这是一个简化的版本，省略了 s11 的错误恢复机制。"""
    system = get_system_prompt(context)
    while True:
        # 1. 调用 LLM API
        try:
            response = client.messages.create(
                model=MODEL, system=system, messages=messages,
                tools=TOOLS, max_tokens=8000)
        except Exception as e:
            # API 调用失败时，将错误信息加入对话并退出循环
            messages.append({"role": "assistant", "content": [
                {"type": "text",
                 "text": f"[Error] {type(e).__name__}: {e}"}]})
            return

        # 2. 将 LLM 回复追加到消息历史
        messages.append({"role": "assistant", "content": response.content})

        # 3. 如果 LLM 不再请求工具调用，循环结束（任务完成）
        if response.stop_reason != "tool_use":
            return

        # 4. 处理所有工具调用请求（批量执行）
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")
            # 通过调度器找到对应的处理函数并执行
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:300])
            results.append({"type": "tool_result",
                            "tool_use_id": block.id, "content": output})

        # 5. 将工具执行结果作为 user 消息返回给 LLM（让它看到结果并决定下一步）
        messages.append({"role": "user", "content": results})
        # 6. 更新上下文（记忆可能已变化）并重新组装 system prompt
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 主程序入口 (Main) ──

if __name__ == "__main__":
    print("s14: task system")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []                        # 消息历史（跨轮次累积）
    context = update_context({}, [])    # 初始化上下文
    while True:
        # 交互式输入（支持 readline 的行编辑功能）
        try:
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        # 退出命令检测
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 将用户输入加入消息历史，启动 agent 循环
        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        context = update_context(context, history)
        # 打印 LLM 的文本回复（跳过工具调用块）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))
        print()
