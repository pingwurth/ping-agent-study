#!/usr/bin/env python3
"""
s20: 综合 Agent —— 将所有教学组件整合到一个循环中。

运行:  python s20_comprehensive/code.py
依赖:  pip install anthropic python-dotenv pyyaml + .env 中配置 ANTHROPIC_API_KEY

本最终章节有意将前面的教学机制重新组合在一起:
  dispatch (工具分发)、permission (权限检查)、hooks (钩子)、todo (待办)、
  subagent (子 agent)、skills (技能)、compaction (上下文压缩)、
  memory (记忆)、prompt assembly (提示词组装)、error recovery (错误恢复)、
  task graph (任务图)、background tasks (后台任务)、cron (定时调度)、
  teams (团队协作)、protocols (协议)、autonomous agents (自主 agent)、
  worktrees (工作树隔离) 和 MCP (模型上下文协议)。
"""

import ast, json, os, subprocess, time, random, threading, re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
import yaml

# readline 用于终端行编辑（上下箭头翻历史、光标移动等）
# 部分环境没有 readline，所以做优雅降级
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如 ANTHROPIC_API_KEY）
load_dotenv(override=True)
# 如果配置了自定义 BASE_URL（如代理），则移除可能冲突的 AUTH_TOKEN
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# ── 全局配置 ──
WORKDIR = Path.cwd()                                    # 工作目录（当前目录）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))  # Anthropic API 客户端
MODEL = os.environ["MODEL_ID"]                          # 主模型 ID
PRIMARY_MODEL = MODEL                                   # 主模型（用于恢复）
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")         # 备用模型（主模型过载时切换）

SKILLS_DIR = WORKDIR / "skills"                          # 技能文件目录
TRANSCRIPT_DIR = WORKDIR / ".transcripts"                # 对话记录保存目录
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"  # 大型工具输出持久化目录

# ── token / 重试 / 压缩 阈值 ──
DEFAULT_MAX_TOKENS = 8000          # 默认最大输出 token 数
ESCALATED_MAX_TOKENS = 16000       # max_tokens 截断后升级的 token 上限
MAX_RETRIES = 3                    # API 调用最大重试次数
MAX_CONSECUTIVE_529 = 2            # 连续 529 过载错误后切换备用模型
MAX_RECOVERY_RETRIES = 2           # max_tokens 截断后最多追加几次 continuation
BASE_DELAY_MS = 500                # 重试基础延迟（毫秒），指数退避
CONTEXT_LIMIT = 50000              # 上下文大小阈值（超过则触发压缩）
KEEP_RECENT_TOOL_RESULTS = 3       # 微压缩：保留最近 N 条工具结果，其余裁剪
PERSIST_THRESHOLD = 30000          # 工具输出超过此字节数则持久化到磁盘
CONTINUATION_PROMPT = "Continue from the previous response. Do not repeat completed work."
PROMPT = "\033[36ms20 >> \033[0m"  # CLI 提示符（青色）
CLI_ACTIVE = False                 # 是否在 CLI 主循环中（影响终端输出格式）


def terminal_print(text: str):
    """线程安全的终端输出。
    主线程或非 CLI 模式下直接 print；
    后台线程中先清除当前行再输出，避免干扰用户正在输入的内容。
    """
    if threading.current_thread() is threading.main_thread() or not CLI_ACTIVE:
        print(text)
        return
    line = ""
    if READLINE_AVAILABLE:
        try:
            line = readline.get_line_buffer()  # 获取用户正在输入的内容
        except Exception:
            line = ""
    print(f"\r\033[K{text}")   # \r 回到行首，\033[K 清除到行尾
    print(PROMPT + line, end="", flush=True)  # 重新显示提示符和用户输入

# ── 任务系统 (Task System) ──
#
# 任务是轻量级的持久化记录，存储在 .tasks/ 目录下的 JSON 文件中。
# 后续的 ownership（所有权）、dependencies（依赖）、worktrees（工作树）
# 和 teammates（队友）都建立在这套文件状态之上。

TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)
CURRENT_TODOS: list[dict] = []  # 当前会话的待办事项（内存态，不持久化）


@dataclass
class Task:
    """任务数据结构。状态流转: pending → in_progress → completed"""
    id: str                    # 唯一 ID（时间戳 + 随机数）
    subject: str               # 任务标题
    description: str           # 任务描述
    status: str                # 状态: pending / in_progress / completed
    owner: str | None          # 认领者（agent 名称）
    blockedBy: list[str]       # 阻塞依赖的任务 ID 列表
    worktree: str | None = None  # 绑定的工作树名称


def _task_path(task_id: str) -> Path:
    """获取任务 JSON 文件的路径"""
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """创建新任务并持久化到磁盘"""
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject, description=description,
        status="pending", owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    """将任务序列化为 JSON 写入文件"""
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    """从磁盘加载任务"""
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    """列出所有任务（按 ID 排序）"""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task_json(task_id: str) -> str:
    """获取任务的 JSON 字符串表示"""
    return json.dumps(asdict(load_task(task_id)), indent=2)


def can_start(task_id: str) -> bool:
    """检查任务是否可以开始：所有阻塞依赖必须存在且已完成。
    依赖关系故意保持简单——逐个检查即可。
    """
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False  # 依赖任务不存在
        if load_task(dep_id).status != "completed":
            return False  # 依赖任务尚未完成
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领任务：检查状态、所有权、依赖，然后标记为 in_progress"""
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if task.owner:
        return f"Task {task_id} already owned by {task.owner}"
    if not can_start(task_id):
        # 给出详细的阻塞原因
        deps = [d for d in task.blockedBy
                if _task_path(d).exists() and load_task(d).status != "completed"]
        missing = [d for d in task.blockedBy if not _task_path(d).exists()]
        parts = []
        if deps: parts.append(f"blocked by: {deps}")
        if missing: parts.append(f"missing deps: {missing}")
        return "Cannot start — " + ", ".join(parts)
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    """完成任务，并报告因此被解锁的后续任务"""
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
    return msg


# ── 工作树系统 (Worktree System) ──
#
# 工作树名称会成为文件系统路径，因此教学版本保持严格的验证规则，
# 并在 create/remove/keep 三个操作中复用。
# 工作树让多个 agent 可以并行在不同分支上工作，互不干扰。

WORKTREES_DIR = WORKDIR / ".worktrees"
WORKTREES_DIR.mkdir(exist_ok=True)

# 工作树名称正则：仅允许字母、数字、点、下划线、短横线，1-64 字符
VALID_WT_NAME = re.compile(r'^[A-Za-z0-9._-]{1,64}$')


def validate_worktree_name(name: str) -> str | None:
    """验证工作树名称合法性，返回错误消息或 None"""
    if not name:
        return "Worktree name cannot be empty"
    if name in (".", ".."):
        return f"'{name}' is not a valid worktree name"
    if not VALID_WT_NAME.match(name):
        return (f"Invalid worktree name '{name}': "
                "only letters, digits, dots, underscores, dashes (1-64 chars)")
    return None


def run_git(args: list[str]) -> tuple[bool, str]:
    """执行 git 命令，返回 (是否成功, 输出文本)"""
    try:
        r = subprocess.run(["git"] + args, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=30)
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out[:5000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return False, "Error: git timeout"


def log_event(event_type: str, worktree_name: str, task_id: str = ""):
    """将工作树事件追加到 events.jsonl（审计日志）"""
    event = {"type": event_type, "worktree": worktree_name,
             "task_id": task_id, "ts": time.time()}
    events_file = WORKTREES_DIR / "events.jsonl"
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")


def create_worktree(name: str, task_id: str = "") -> str:
    """创建 git 工作树（隔离的分支目录）。
    工具层验证是安全边界的一部分——在 git 看到名称之前就要检查。
    """
    err = validate_worktree_name(name)
    if err:
        return f"Error: {err}"
    if task_id:
        try:
            load_task(task_id)
        except FileNotFoundError:
            return f"Error: task {task_id} not found"
    path = WORKTREES_DIR / name
    if path.exists():
        return f"Worktree '{name}' already exists at {path}"
    # git worktree add: 在指定路径创建新工作树，基于 HEAD 创建新分支 wt/{name}
    ok, result = run_git(["worktree", "add", str(path), "-b", f"wt/{name}", "HEAD"])
    if not ok:
        return f"Git error: {result}"
    if task_id:
        bind_task_to_worktree(task_id, name)  # 将任务绑定到工作树
    log_event("create", name, task_id)
    print(f"  \033[33m[worktree] created: {name} at {path}\033[0m")
    return f"Worktree '{name}' created at {path}"


def bind_task_to_worktree(task_id: str, worktree_name: str):
    """将任务与工作树关联"""
    task = load_task(task_id)
    task.worktree = worktree_name
    save_task(task)


def _count_worktree_changes(path: Path) -> tuple[int, int]:
    """统计工作树中的未提交文件数和未推送提交数"""
    try:
        r1 = subprocess.run(["git", "status", "--porcelain"],
                            cwd=path, capture_output=True, text=True, timeout=10)
        files = len([l for l in r1.stdout.strip().splitlines() if l.strip()])
        r2 = subprocess.run(["git", "log", "@{push}..HEAD", "--oneline"],
                            cwd=path, capture_output=True, text=True, timeout=10)
        commits = len([l for l in r2.stdout.strip().splitlines() if l.strip()])
        return files, commits
    except Exception:
        return -1, -1


def remove_worktree(name: str, discard_changes: bool = False) -> str:
    """删除工作树。如果有未保存的更改，除非 discard_changes=True，否则拒绝删除。"""
    err = validate_worktree_name(name)
    if err:
        return err
    path = WORKTREES_DIR / name
    if not path.exists():
        return f"Worktree '{name}' not found"
    if not discard_changes:
        files, commits = _count_worktree_changes(path)
        if files < 0:
            return "Cannot verify status. Use discard_changes=true to force."
        if files > 0 or commits > 0:
            return (f"Worktree '{name}' has {files} file(s), {commits} commit(s). "
                    "Use discard_changes=true or keep_worktree.")
    # 先移除工作树，再删除分支
    ok1, _ = run_git(["worktree", "remove", str(path), "--force"])
    if not ok1:
        return f"Failed to remove worktree '{name}'"
    run_git(["branch", "-D", f"wt/{name}"])
    log_event("remove", name)
    print(f"  \033[33m[worktree] removed: {name}\033[0m")
    return f"Worktree '{name}' removed"


def keep_worktree(name: str) -> str:
    """保留工作树供人工审查（不删除）"""
    err = validate_worktree_name(name)
    if err:
        return err
    log_event("keep", name)
    return f"Worktree '{name}' kept for review (branch: wt/{name})"


# ── 技能加载 (Skill Loading) ──
#
# 技能是位于 skills/ 目录下的 Markdown 文件，每个技能有一个 SKILL.md 清单文件。
# 清单文件使用 YAML frontmatter 定义 name 和 description。
# Agent 可以通过 load_skill(name) 按需加载技能内容。

SKILL_REGISTRY: dict[str, dict] = {}  # 技能注册表: name → {name, description, content}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 Markdown 文件的 YAML frontmatter（--- 包裹的部分）"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()


def scan_skills():
    """扫描 skills/ 目录，将所有技能注册到 SKILL_REGISTRY"""
    SKILL_REGISTRY.clear()
    if not SKILLS_DIR.exists():
        return
    for directory in sorted(SKILLS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        manifest = directory / "SKILL.md"
        if not manifest.exists():
            continue
        raw = manifest.read_text()
        meta, _ = _parse_frontmatter(raw)
        name = meta.get("name", directory.name)  # 优先用 frontmatter 中的 name
        desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
        SKILL_REGISTRY[name] = {
            "name": name,
            "description": desc,
            "content": raw,  # 保存完整内容，load_skill 时返回
        }


scan_skills()  # 启动时立即扫描


def list_skills() -> str:
    """列出所有已注册技能的名称和描述"""
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(
        f"- {skill['name']}: {skill['description']}"
        for skill in SKILL_REGISTRY.values())


def load_skill(name: str) -> str:
    """按名称加载技能的完整内容"""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        available = ", ".join(SKILL_REGISTRY.keys()) or "(none)"
        return f"Skill not found: {name}. Available: {available}"
    return skill["content"]


# ── Prompt Assembly ──

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, edit_file, glob, "
             "todo_write, task, load_skill, compact, "
             "create_task, list_tasks, get_task, claim_task, complete_task, "
             "schedule_cron, list_crons, cancel_cron, "
             "spawn_teammate, send_message, check_inbox, "
             "request_shutdown, request_plan, review_plan, "
             "create_worktree, remove_worktree, keep_worktree, "
             "connect_mcp. MCP tools are prefixed mcp__{server}__{tool}.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    # The system prompt is rebuilt each turn from live context. This is where
    # memory, skill catalog, MCP state, and active teammates become visible.
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    sections.append(f"Current time: {datetime.now().isoformat(timespec='seconds')}")
    sections.append("Skills catalog:\n" + list_skills() +
                    "\nUse load_skill(name) when a skill is relevant.")
    if context.get("memories"):
        sections.append(f"Relevant memories:\n{context['memories']}")
    mcp_names = list(mcp_clients.keys())
    if mcp_names:
        sections.append(f"Connected MCP servers: {', '.join(mcp_names)}")
    return "\n\n".join(sections)


# ── 基础工具 (Basic Tools) ──
#
# 这些是 agent 最常用的文件和命令操作工具。
# 每个工具都支持可选的 cwd 参数，以支持工作树隔离。

def run_bash(command: str, cwd: Path = None,
             run_in_background: bool = False) -> str:
    """执行 shell 命令。run_in_background 由调度器处理，此处直接执行。"""
    try:
        r = subprocess.run(command, shell=True, cwd=cwd or WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None,
             offset: int = 0, cwd: Path = None) -> str:
    """读取文件内容，支持 offset（跳过前 N 行）和 limit（最多读 N 行）"""
    try:
        base = cwd or WORKDIR
        file_path = (base / path).resolve()
        lines = file_path.read_text().splitlines()
        offset = max(int(offset or 0), 0)
        limit = int(limit) if limit is not None else None
        lines = lines[offset:]
        if limit is not None and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str, cwd: Path = None) -> str:
    """写入文件（自动创建父目录）"""
    try:
        base = cwd or WORKDIR
        fp = (base / path).resolve()
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str,
             cwd: Path = None) -> str:
    """精确替换文件中的文本（只替换第一次出现）"""
    try:
        base = cwd or WORKDIR
        fp = (base / path).resolve()
        text = fp.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        fp.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str, cwd: Path = None) -> str:
    """按 glob 模式查找文件（如 '*.py'、'**/*.md'）"""
    import glob as g
    try:
        base = cwd or WORKDIR
        results = []
        for match in g.glob(pattern, root_dir=base):
            # 安全检查：确保匹配路径不会逃逸出工作目录
            if (base / match).resolve().is_relative_to(base):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def call_tool_handler(handler, args: dict, name: str) -> str:
    """统一的工具调用入口：检查 handler 是否存在，然后传参调用"""
    if not handler:
        return f"Unknown: {name}"
    try:
        return handler(**(args or {}))
    except TypeError as e:
        return f"Error: {e}"


def _normalize_todos(todos):
    """验证并规范化待办事项列表。
    支持 JSON 字符串或 Python 字面量格式，确保每项都有 content 和合法 status。
    """
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, todo in enumerate(todos):
        if not isinstance(todo, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in todo or "status" not in todo:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if todo["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{todo['status']}'"
    return todos, None

def run_todo_write(todos: list) -> str:
    """更新当前会话的待办事项列表（内存态）"""
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    print(f"  \033[33m[todo] updated {len(CURRENT_TODOS)} item(s)\033[0m")
    return f"Updated {len(CURRENT_TODOS)} todos"


# ── 消息总线 (MessageBus) ──
#
# 团队通信使用追加写入的 JSONL 邮箱文件。
# 这种设计让协议消息可在磁盘上检查，也允许后台队友发送消息。
# 每个 agent 有一个 .mailboxes/{agent_name}.jsonl 文件。

MAILBOX_DIR = WORKDIR / ".mailboxes"
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    """基于文件的消息总线，用于 agent 间通信"""

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict = None):
        """发送消息到目标 agent 的邮箱"""
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        terminal_print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
                       f"({msg_type}) {content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        """读取并清空 agent 的邮箱（一次性消费）"""
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]
        inbox.unlink()  # 读取后删除，避免重复消费
        return msgs


BUS = MessageBus()
active_teammates: dict[str, bool] = {}  # 当前活跃的队友线程: name → True

# ── 协议状态 (Protocol State) ──
#
# 协议用于 lead（主 agent）和 teammate（队友）之间的结构化交互：
# - shutdown: lead 请求 teammate 关闭
# - plan_approval: teammate 提交计划，lead 审批/拒绝
# 每个请求有唯一的 request_id，响应回来时通过 ID 匹配。

@dataclass
class ProtocolState:
    """协议请求的状态记录"""
    request_id: str    # 唯一请求 ID
    type: str          # 协议类型: shutdown / plan_approval
    sender: str        # 发送方
    target: str        # 目标方
    status: str        # 状态: pending / approved / rejected
    payload: str       # 载荷（如计划内容）
    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {}  # 待处理的协议请求


def new_request_id() -> str:
    """生成随机请求 ID"""
    return f"req_{random.randint(0, 999999):06d}"


def match_response(response_type: str, request_id: str, approve: bool):
    """将响应匹配到对应的待处理请求。
    通过 request_id 精确匹配，避免一个协议回复意外批准另一个请求。
    """
    state = pending_requests.get(request_id)
    if not state:
        return
    # 类型校验：shutdown 只接受 shutdown_response，plan_approval 只接受 plan_approval_response
    if state.type == "shutdown" and response_type != "shutdown_response":
        return
    if state.type == "plan_approval" and response_type != "plan_approval_response":
        return
    state.status = "approved" if approve else "rejected"


def consume_lead_inbox(route_protocol=True) -> list[dict]:
    """读取 lead 的收件箱，并自动路由协议响应到对应的请求状态"""
    msgs = BUS.read_inbox("lead")
    if route_protocol:
        for msg in msgs:
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            msg_type = msg.get("type", "")
            if req_id and msg_type.endswith("_response"):
                match_response(msg_type, req_id, meta.get("approve", False))
    return msgs


# ── 自主 Agent (Autonomous Agent) ──
#
# 自主队友在空闲时轮询：先检查收件箱消息，再查找未认领的任务。
# 这确保了直接的协议消息（如 shutdown）优先级高于自动认领任务。

IDLE_POLL_INTERVAL = 5   # 空闲轮询间隔（秒）
IDLE_TIMEOUT = 60        # 空闲超时（秒），超时后自动退出


def scan_unclaimed_tasks() -> list[dict]:
    """扫描所有可认领的任务：状态为 pending、无 owner、依赖已满足"""
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and can_start(task["id"])):
            unclaimed.append(task)
    return unclaimed


def idle_poll(agent_name: str, messages: list,
              name: str, role: str,
              worktree_context: dict | None = None) -> str:
    """空闲轮询循环：每 IDLE_POLL_INTERVAL 秒检查一次收件箱和可认领任务。
    返回: "work"（有新消息或认领了任务）、"shutdown"（收到关闭请求）、"timeout"（超时）
    """
    for _ in range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL):
        time.sleep(IDLE_POLL_INTERVAL)
        # 优先级 1: 检查收件箱消息（如 shutdown_request、普通消息）
        inbox = BUS.read_inbox(agent_name)
        if inbox:
            for msg in inbox:
                if msg.get("type") == "shutdown_request":
                    # 收到关闭请求，立即响应并退出
                    req_id = msg.get("metadata", {}).get("request_id", "")
                    BUS.send(name, "lead", "Shutting down.",
                             "shutdown_response",
                             {"request_id": req_id, "approve": True})
                    return "shutdown"
            # 将非关闭消息注入对话历史
            messages.append({"role": "user",
                "content": "<inbox>" + json.dumps(inbox) + "</inbox>"})
            return "work"
        # 优先级 2: 查找并自动认领可执行的任务
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task_data = unclaimed[0]
            result = claim_task(task_data["id"], agent_name)
            if "Claimed" in result:
                wt_info = ""
                if task_data.get("worktree"):
                    wt_path = WORKTREES_DIR / task_data["worktree"]
                    wt_info = f"\nWork directory: {wt_path}"
                    if worktree_context is not None:
                        worktree_context["path"] = str(wt_path)
                # 将认领的任务作为消息注入，让 agent 开始工作
                messages.append({"role": "user",
                    "content": f"<auto-claimed>Task {task_data['id']}: "
                               f"{task_data['subject']}{wt_info}</auto-claimed>"})
                return "work"
    return "timeout"  # 超过 IDLE_TIMEOUT 秒没有活动


# ── 队友线程 (Teammate Thread) ──
#
# 每个队友在独立的守护线程中运行自己的 agent 循环。
# 队友拥有精简的工具集（bash、read、write、send_message、task 管理等），
# 并支持工作树隔离——认领带 worktree 的任务后，所有文件操作自动切换到该目录。

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """启动一个自主队友线程。队友会在后台轮询收件箱和任务。"""
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    # 计划审批是一个真正的门控：submit_plan 之后，队友停止执行，
    # 等待 lead 发送 plan_approval_response 才继续。
    protocol_ctx = {"waiting_plan": None}
    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks. "
              f"If a task has a worktree, work in that directory.")

    def handle_inbox_message(name: str, msg: dict, messages: list):
        """处理收到的协议消息。返回 True 表示应关闭线程。"""
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")
        if msg_type == "shutdown_request":
            # 响应关闭请求，返回 True 触发线程退出
            BUS.send(name, "lead", "Shutting down.",
                     "shutdown_response",
                     {"request_id": req_id, "approve": True})
            return True
        if msg_type == "plan_approval_response":
            # 收到计划审批结果，清除等待状态，注入结果到对话
            approve = meta.get("approve", False)
            if req_id == protocol_ctx["waiting_plan"]:
                protocol_ctx["waiting_plan"] = None
            messages.append({"role": "user",
                "content": "[Plan approved]" if approve
                           else f"[Plan rejected] {msg['content']}"})
        return False

    def run():
        """队友的主循环：调用 LLM → 执行工具 → 检查收件箱 → 空闲轮询"""
        wt_ctx = {"path": None}  # 工作树上下文：认领任务后设置路径

        def _wt_cwd():
            """获取当前工作目录：有工作树则用工作树路径，否则返回 None（用默认目录）"""
            p = wt_ctx["path"]
            return Path(p) if p else None

        def _run_bash(command: str) -> str:
            return run_bash(command, cwd=_wt_cwd())

        def _run_read(path: str) -> str:
            return run_read(path, cwd=_wt_cwd())

        def _run_write(path: str, content: str) -> str:
            return run_write(path, content, cwd=_wt_cwd())

        def _run_list_tasks():
            tasks = list_tasks()
            if not tasks:
                return "No tasks."
            return "\n".join(
                f"  {t.id}: {t.subject} [{t.status}]"
                + (f" (wt:{t.worktree})" if t.worktree else "")
                for t in tasks)

        def _run_claim_task(task_id: str):
            """认领任务，并切换工作目录到绑定的工作树（如果有）"""
            result = claim_task(task_id, owner=name)
            if "Claimed" in result:
                task = load_task(task_id)
                wt_ctx["path"] = (str(WORKTREES_DIR / task.worktree)
                                  if task.worktree else None)
            return result

        def _run_complete_task(task_id: str):
            """完成任务，清除工作树上下文"""
            result = complete_task(task_id)
            wt_ctx["path"] = None
            return result

        messages = [{"role": "user", "content": prompt}]
        # 队友的工具集（比 lead 精简，不含 compact、cron、MCP 等）
        sub_tools = [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "Read file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"},
                                             "limit": {"type": "integer"},
                                             "offset": {"type": "integer"}},
                              "required": ["path"]}},
            {"name": "write_file", "description": "Write file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["path", "content"]}},
            {"name": "send_message",
             "description": "Send message to another agent.",
             "input_schema": {"type": "object",
                              "properties": {"to": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["to", "content"]}},
            {"name": "submit_plan",
             "description": "Submit a plan for Lead approval.",
             "input_schema": {"type": "object",
                              "properties": {"plan": {"type": "string"}},
                              "required": ["plan"]}},
            {"name": "list_tasks",
             "description": "List all tasks.",
             "input_schema": {"type": "object", "properties": {},
                              "required": []}},
            {"name": "claim_task",
             "description": "Claim a pending task.",
             "input_schema": {"type": "object",
                              "properties": {"task_id": {"type": "string"}},
                              "required": ["task_id"]}},
            {"name": "complete_task",
             "description": "Mark an in-progress task as completed.",
             "input_schema": {"type": "object",
                              "properties": {"task_id": {"type": "string"}},
                              "required": ["task_id"]}},
        ]

        sub_handlers = {
            "bash": _run_bash, "read_file": _run_read,
            "write_file": _run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content),
                                                  "Sent")[1],
            "list_tasks": _run_list_tasks,
            "claim_task": _run_claim_task,
            "complete_task": _run_complete_task,
        }

        while True:
            # 每轮开始时注入身份信息（如果消息历史很短）
            if len(messages) <= 3:
                messages.insert(0, {"role": "user",
                    "content": f"<identity>You are '{name}', role: {role}. "
                               f"Continue your work.</identity>"})
            should_shutdown = False
            # 内层循环：最多执行 10 轮 LLM 调用
            for _ in range(10):
                # 检查收件箱
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    stopped = handle_inbox_message(name, msg, messages)
                    if stopped:
                        should_shutdown = True
                        break
                if should_shutdown:
                    break
                if protocol_ctx["waiting_plan"]:
                    # 计划审批门控已关闭：只轮询协议回复，不让模型继续工作
                    time.sleep(IDLE_POLL_INTERVAL)
                    continue
                # 将非协议消息注入对话（如其他队友发来的普通消息）
                if inbox and not should_shutdown:
                    non_protocol = [m for m in inbox
                                    if m.get("type") == "message"]
                    if non_protocol:
                        messages.append({"role": "user",
                            "content": "<inbox>" + json.dumps(non_protocol) + "</inbox>"})
                # 调用 LLM（只保留最近 20 条消息避免上下文溢出）
                try:
                    response = client.messages.create(
                        model=MODEL, system=system, messages=messages[-20:],
                        tools=sub_tools, max_tokens=8000)
                except Exception:
                    break
                messages.append({"role": "assistant", "content": response.content})
                if not has_tool_use(response.content):
                    break  # 模型没有调用工具，本轮结束
                # 执行模型请求的工具
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "submit_plan":
                            # 提交计划需要 lead 审批
                            output = _teammate_submit_plan(
                                name, block.input.get("plan", ""))
                            match = re.search(r"\((req_\d+)\)", output)
                            protocol_ctx["waiting_plan"] = (
                                match.group(1) if match else output)
                        else:
                            handler = sub_handlers.get(block.name)
                            output = call_tool_handler(handler, block.input,
                                                       block.name)
                        results.append({"type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": str(output)})
                        if protocol_ctx["waiting_plan"]:
                            # 提交计划后忽略同一次响应中的其他工具调用
                            # 它们应该在审批通过后再执行
                            break
                messages.append({"role": "user", "content": results})
                if protocol_ctx["waiting_plan"]:
                    break
            if should_shutdown:
                break
            if protocol_ctx["waiting_plan"]:
                continue  # 等待审批，继续轮询
            # 内层循环结束，进入空闲轮询
            idle_result = idle_poll(name, messages, name, role, wt_ctx)
            if idle_result in ("shutdown", "timeout"):
                break

        # 队友退出前提取最后的文本摘要，发送给 lead
        summary = "Done."
        for msg in reversed(messages):
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for b in msg["content"]:
                    if getattr(b, "type", None) == "text":
                        summary = b.text
                        break
                else:
                    continue
                break
        # 退出前将摘要发送给 lead，然后清理活跃队友记录
        BUS.send(name, "lead", summary, "result")
        active_teammates.pop(name, None)

    active_teammates[name] = True
    threading.Thread(target=run, daemon=True).start()  # 守护线程，主进程退出时自动终止
    return f"Teammate '{name}' spawned as {role}"


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """队友提交计划：创建协议请求，发送给 lead 等待审批"""
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="plan_approval",
        sender=from_name, target="lead",
        status="pending", payload=plan)
    BUS.send(from_name, "lead", plan,
             "plan_approval_request",
             {"request_id": req_id})
    return f"Plan submitted ({req_id})"


# ── Lead 协议工具 (Lead Protocol Tools) ──
#
# Lead 使用这些工具与队友进行结构化交互：
# - request_shutdown: 请求队友关闭
# - request_plan: 要求队友提交计划
# - review_plan: 审批或拒绝队友提交的计划

def run_request_shutdown(teammate: str) -> str:
    """向队友发送关闭请求"""
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="shutdown",
        sender="lead", target=teammate,
        status="pending", payload="")
    BUS.send("lead", teammate, "Shut down.", "shutdown_request",
             {"request_id": req_id})
    return f"Shutdown request sent to {teammate}"


def run_request_plan(teammate: str, task: str) -> str:
    """要求队友针对某任务提交计划"""
    BUS.send("lead", teammate, f"Submit plan for: {task}", "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(request_id: str, approve: bool,
                    feedback: str = "") -> str:
    """审批或拒绝队友提交的计划，发送响应"""
    state = pending_requests.get(request_id)
    if not state:
        return f"Request {request_id} not found"
    state.status = "approved" if approve else "rejected"
    BUS.send("lead", state.sender,
             feedback or ("Approved" if approve else "Rejected"),
             "plan_approval_response",
             {"request_id": request_id, "approve": approve})
    return f"Plan {'approved' if approve else 'rejected'}"


# ── 钩子 + 权限管道 (Hooks + Permission Pipeline) ──
#
# 钩子故意放在工具处理器之外。这样循环可以添加权限、日志和停止行为，
# 而无需修改每个单独的工具。
# 四种钩子事件：
#   UserPromptSubmit — 用户提交提示时触发
#   PreToolUse       — 工具执行前触发（可阻止执行）
#   PostToolUse      — 工具执行后触发
#   Stop             — agent 循环结束时触发

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [],
         "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    """注册钩子回调"""
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """触发指定事件的所有钩子。第一个返回非 None 的钩子结果会中断链。"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result  # 钩子返回值表示阻止/中断
    return None


# 危险命令黑名单（直接拒绝）
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
# 破坏性命令（需要用户确认）
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def permission_hook(block):
    """权限检查钩子：在工具执行前检查安全性。
    权限层在工具分发之前看到原始的 tool_use 块，
    可以拒绝、询问用户或允许继续执行。
    """
    if block.name == "bash":
        command = block.input.get("command", "")
        # 检查黑名单：直接拒绝
        for pattern in DENY_LIST:
            if pattern in command:
                return f"Permission denied: '{pattern}' is on the deny list"
        # 检查破坏性命令：需要用户确认
        if any(token in command for token in DESTRUCTIVE):
            print(f"\n\033[33m[permission] destructive command\033[0m")
            print(f"  {command}")
            choice = input("  Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    # 文件操作：检查路径是否逃逸出工作目录
    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"  {block.name}: {path}")
            choice = input("  Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    # MCP 工具：包含 "deploy" 的需要确认
    if block.name.startswith("mcp__") and "deploy" in block.name:
        print(f"\n\033[33m[permission] MCP destructive-looking tool: {block.name}\033[0m")
        choice = input("  Allow? [y/N] ").strip().lower()
        if choice not in ("y", "yes"):
            return "Permission denied by user"
    return None


def log_hook(block):
    """日志钩子：记录每次工具调用"""
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None


def large_output_hook(block, output):
    """大输出警告钩子：工具输出超过 100KB 时发出警告"""
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] large output from {block.name}: "
              f"{len(str(output))} chars\033[0m")
    return None


def user_prompt_hook(query: str):
    """用户提示钩子：记录用户输入"""
    print(f"\033[90m[HOOK] UserPromptSubmit: {WORKDIR}\033[0m")
    return None


def stop_hook(messages: list):
    """停止钩子：统计本轮工具调用次数"""
    tool_count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            tool_count += sum(1 for item in content
                              if isinstance(item, dict)
                              and item.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: {tool_count} tool result(s)\033[0m")
    return None


# 注册所有钩子
register_hook("UserPromptSubmit", user_prompt_hook)
register_hook("PreToolUse", permission_hook)  # 权限检查优先
register_hook("PreToolUse", log_hook)         # 然后记录日志
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", stop_hook)


# ── 子 Agent 工具 (Subagent Tool) ──
#
# 子 agent 是一次性的任务执行器：接收描述，完成工作，返回摘要。
# 与队友不同，子 agent 不轮询、不持久化，执行完就销毁。
# 限制：子 agent 不能再创建子 agent（防止递归）。

SUB_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. "
    "Complete the task, then return a concise final summary. "
    "Do not spawn more agents."
)


SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"},
                                     "offset": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]


SUB_HANDLERS = {
    "bash": run_bash, "read_file": run_read,
    "write_file": run_write, "edit_file": run_edit,
    "glob": run_glob,
}


def extract_text(content) -> str:
    """从模型响应中提取所有 text 块的内容"""
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        getattr(block, "text", "")
        for block in content
        if getattr(block, "type", None) == "text").strip()


def has_tool_use(content) -> bool:
    """检查响应中是否包含 tool_use 块。
    不依赖 stop_reason，因为具体的 tool_use 块才是循环继续的信号。
    """
    return any(getattr(block, "type", None) == "tool_use"
               for block in content)


def spawn_subagent(description: str) -> str:
    """启动子 agent 执行任务，最多 30 轮工具调用，返回最终文本摘要。"""
    messages = [{"role": "user", "content": description}]
    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM, messages=messages,
            tools=SUB_TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        if not has_tool_use(response.content):
            break  # 模型完成，没有更多工具调用
        # 执行子 agent 请求的工具（也经过钩子管道）
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                output = str(blocked)
            else:
                handler = SUB_HANDLERS.get(block.name)
                output = call_tool_handler(handler, block.input, block.name)
                trigger_hooks("PostToolUse", block, output)
            results.append({"type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output)})
        messages.append({"role": "user", "content": results})
    # 从最后一条 assistant 消息中提取文本摘要
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            text = extract_text(msg["content"])
            if text:
                return text
    return "Subagent finished without a text summary."


# ── 上下文压缩 (Context Compaction) ──
#
# 压缩是分层的：
# 1. tool_result_budget — 持久化超大的工具输出到磁盘
# 2. snip_compact      — 裁剪中间的消息范围（保留头尾）
# 3. micro_compact     — 将旧的工具结果替换为占位符
# 4. compact_history   — 调用 LLM 总结整个对话（最重的操作）
# 只有当上下文仍然太大或模型明确请求 compact 时，才调用 LLM 总结。

def estimate_size(messages: list) -> int:
    """估算消息列表的 JSON 大小（字节）"""
    return len(json.dumps(messages, default=str))

def block_type(block):
    """获取消息块的类型（兼容 dict 和对象两种格式）"""
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def message_has_tool_use(message: dict) -> bool:
    """检查消息是否包含 tool_use 块"""
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(block_type(block) == "tool_use" for block in content)


def is_tool_result_message(message: dict) -> bool:
    """检查消息是否是工具结果消息"""
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result"
               for block in content)


def collect_tool_results(messages: list):
    """收集所有工具结果块，返回 (消息索引, 块索引, 块内容) 列表"""
    found = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if msg.get("role") != "user" or not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                found.append((mi, bi, block))
    return found


def persist_large_output(tool_use_id: str, output: str) -> str:
    """将大型工具输出持久化到磁盘，返回预览 + 文件路径"""
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists():
        path.write_text(output)
    return (f"<persisted-output>\nFull output: {path}\n"
            f"Preview:\n{output[:2000]}\n</persisted-output>")


def tool_result_budget(messages: list, max_bytes: int = 200_000) -> list:
    """工具结果预算：如果最近一条消息的工具结果总量超过 max_bytes，
    则将最大的结果持久化到磁盘，直到总量降到预算内。
    """
    if not messages:
        return messages
    last = messages[-1]
    content = last.get("content")
    if last.get("role") != "user" or not isinstance(content, list):
        return messages
    blocks = [(i, b) for i, b in enumerate(content)
              if isinstance(b, dict) and b.get("type") == "tool_result"]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes:
        return messages
    # 从最大的结果开始持久化
    for _, block in sorted(blocks,
                           key=lambda pair: len(str(pair[1].get("content", ""))),
                           reverse=True):
        if total <= max_bytes:
            break
        text = str(block.get("content", ""))
        block["content"] = persist_large_output(
            block.get("tool_use_id", "unknown"), text)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return messages


def snip_compact(messages: list, max_messages: int = 50) -> list:
    """裁剪压缩：保留头部 3 条和尾部消息，中间替换为 [snipped N messages]。
    智能处理 tool_use/tool_result 对，不会拆散它们。
    """
    if len(messages) <= max_messages:
        return messages
    head_end, tail_start = 3, len(messages) - (max_messages - 3)
    # 确保不拆散 tool_use 和对应的 tool_result
    if head_end > 0 and message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and is_tool_result_message(messages[head_end]):
            head_end += 1
    if (tail_start > 0 and tail_start < len(messages)
            and is_tool_result_message(messages[tail_start])
            and message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    if head_end >= tail_start:
        return messages
    snipped = tail_start - head_end
    return (messages[:head_end]
            + [{"role": "user", "content": f"[snipped {snipped} messages]"}]
            + messages[tail_start:])


def micro_compact(messages: list) -> list:
    """微压缩：将旧的工具结果替换为占位符，只保留最近 KEEP_RECENT_TOOL_RESULTS 条"""
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages
    for _, _, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        if len(str(block.get("content", ""))) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


def write_transcript(messages: list) -> Path:
    """将完整对话历史写入 transcript 文件（用于压缩前的备份）"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    """调用 LLM 总结对话历史，保留目标、发现、更改文件和剩余工作"""
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = ("Summarize this coding-agent conversation so work can continue. "
              "Preserve current goal, key findings, changed files, remaining work, "
              "and user constraints.\n\n" + conversation)
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000)
    return extract_text(response.content) or "(empty summary)"


def compact_history(messages: list) -> list:
    """主动压缩：保存 transcript → 总结 → 替换整个历史"""
    transcript = write_transcript(messages)
    print(f"  \033[36m[compact] transcript saved: {transcript}\033[0m")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


def reactive_compact(messages: list) -> list:
    """响应式压缩：当 API 返回 prompt-too-long 错误时触发。
    保留最近 5 条消息，总结其余部分。
    """
    transcript = write_transcript(messages)
    print(f"  \033[31m[reactive compact] transcript saved: {transcript}\033[0m")
    tail_start = max(0, len(messages) - 5)
    # 确保不拆散 tool_use/tool_result 对
    if (tail_start > 0 and tail_start < len(messages)
            and is_tool_result_message(messages[tail_start])
            and message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    try:
        summary = summarize_history(messages[:tail_start])
    except Exception:
        summary = "Earlier conversation was trimmed after a prompt-too-long error."
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"},
            *messages[tail_start:]]


# ── 错误恢复 (Error Recovery) ──
#
# 错误恢复策略：
# - 429 (Rate Limit): 指数退避重试
# - 529 (Overloaded): 指数退避 + 连续 N 次后切换备用模型
# - max_tokens: 升级 token 上限 → 追加 continuation prompt
# - prompt-too-long: 触发响应式压缩

class RecoveryState:
    """跟踪错误恢复的状态"""
    def __init__(self):
        self.has_escalated = False              # 是否已升级过 max_tokens
        self.recovery_count = 0                 # continuation 追加次数
        self.consecutive_529 = 0                # 连续 529 错误计数
        self.has_attempted_reactive_compact = False  # 是否已尝试响应式压缩
        self.current_model = PRIMARY_MODEL      # 当前使用的模型


def retry_delay(attempt: int) -> float:
    """计算重试延迟：指数退避 + 随机抖动（避免雷群效应）"""
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000
    return base + random.uniform(0, base * 0.25)


def with_retry(fn, state: RecoveryState):
    """带重试的 API 调用包装器。
    处理 429（限速）和 529（过载）错误，支持指数退避和模型切换。
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0  # 成功后重置连续 529 计数
            return result
        except Exception as e:
            name = type(e).__name__.lower()
            msg = str(e).lower()
            if "ratelimit" in name or "429" in msg:
                # 限速错误：指数退避重试
                delay = retry_delay(attempt)
                print(f"  \033[33m[429] retry {attempt + 1}/{MAX_RETRIES} "
                      f"after {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue
            if "overloaded" in name or "529" in msg or "overloaded" in msg:
                # 过载错误：连续 N 次后切换到备用模型
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529 and FALLBACK_MODEL:
                    state.current_model = FALLBACK_MODEL
                    state.consecutive_529 = 0
                    print(f"  \033[31m[529] switching to {FALLBACK_MODEL}\033[0m")
                delay = retry_delay(attempt)
                print(f"  \033[33m[529] retry {attempt + 1}/{MAX_RETRIES} "
                      f"after {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue
            raise  # 其他错误直接抛出
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")


def is_prompt_too_long_error(e: Exception) -> bool:
    """检测是否为上下文长度超限错误"""
    msg = str(e).lower()
    return (("prompt" in msg and "long" in msg)
            or "context_length_exceeded" in msg
            or "max_context_window" in msg)


# ── 后台任务 (Background Tasks) ──
#
# 慢速工具（如 install、build、test）立即返回占位符 tool_result，
# 真正的输出稍后作为 task_notification 注入，这样主循环可以继续前进。

_bg_counter = 0                                    # 后台任务计数器
background_tasks: dict[str, dict] = {}             # 运行中的后台任务
background_results: dict[str, str] = {}            # 已完成的结果
background_lock = threading.Lock()                 # 线程安全锁


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """判断是否为慢速操作（基于命令关键词）"""
    if tool_name != "bash":
        return False
    command = tool_input.get("command", "").lower()
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(keyword in command for keyword in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """判断是否应在后台运行：显式标记 run_in_background 或自动检测慢速操作"""
    if tool_name != "bash":
        return False
    return bool(tool_input.get("run_in_background")) or is_slow_operation(tool_name, tool_input)


def start_background_task(block, handlers: dict) -> str:
    """启动后台任务：在独立线程中执行工具，立即返回占位符 ID"""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    command = block.input.get("command", block.name)

    def worker():
        """后台工作线程：执行工具 → 触发钩子 → 存储结果"""
        handler = handlers.get(block.name)
        result = call_tool_handler(handler, block.input, block.name)
        trigger_hooks("PostToolUse", block, result)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = str(result)

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": command,
            "status": "running",
        }
    threading.Thread(target=worker, daemon=True).start()
    print(f"  \033[33m[background] {bg_id}: {str(command)[:60]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    """收集已完成的后台任务结果，格式化为 XML 通知注入对话"""
    with background_lock:
        ready = [bg_id for bg_id, task in background_tasks.items()
                 if task["status"] == "completed"]
    notifications = []
    for bg_id in ready:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
    return notifications


# ── 定时调度器 (Cron Scheduler) ──
#
# Cron 任务独立于对话历史存储。当任务触发时，
# 它变成一个 scheduled prompt 注入回同一个 agent 循环。
# 支持 5 字段 cron 表达式（分 时 日 月 周）。
# durable=True 的任务会持久化到 .scheduled_tasks.json，重启后恢复。

DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"


@dataclass
class CronJob:
    """Cron 任务数据结构"""
    id: str           # 唯一 ID
    cron: str         # 5 字段 cron 表达式（分 时 日 月 周）
    prompt: str       # 触发时注入的提示词
    recurring: bool   # 是否循环执行（False = 一次性）
    durable: bool     # 是否持久化到磁盘


scheduled_jobs: dict[str, CronJob] = {}   # 已注册的 cron 任务
cron_queue: list[CronJob] = []            # 待执行的触发队列
cron_lock = threading.Lock()              # 线程安全锁
_last_fired: dict[str, str] = {}          # 防止同一分钟重复触发


def _cron_field_matches(field: str, value: int) -> bool:
    """检查单个 cron 字段是否匹配当前值。
    支持: * (通配)、*/N (步进)、a-b (范围)、a,b,c (列表)
    """
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(_cron_field_matches(part.strip(), value)
                   for part in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """检查 cron 表达式是否匹配给定的日期时间。
    标准 cron 规则：dom 和 dow 都为 * 时匹配任何日期，
    否则只要其中一个匹配即可（OR 逻辑）。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7  # Python weekday (0=Mon) → cron (0=Sun)
    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)
    if not (m and h and month_ok):
        return False
    # dom 和 dow 的组合逻辑
    if dom == "*" and dow == "*":
        return True
    if dom == "*":
        return dow_ok
    if dow == "*":
        return dom_ok
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """验证单个 cron 字段的合法性"""
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) <= 0:
            return f"Invalid step: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None
    if "-" in field:
        left, right = field.split("-", 1)
        if not left.isdigit() or not right.isdigit():
            return f"Invalid range: {field}"
        a, b = int(left), int(right)
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < lo or value > hi:
        return f"Value {value} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """验证完整的 5 字段 cron 表达式，返回错误消息或 None"""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for field, (lo, hi), name in zip(fields, bounds, names):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    """将 durable=True 的 cron 任务保存到磁盘"""
    durable = [asdict(job) for job in scheduled_jobs.values() if job.durable]
    DURABLE_PATH.write_text(json.dumps(durable, indent=2))


def load_durable_jobs():
    """从磁盘加载持久化的 cron 任务"""
    if not DURABLE_PATH.exists():
        return
    try:
        for item in json.loads(DURABLE_PATH.read_text()):
            job = CronJob(**item)
            if not validate_cron(job.cron):
                scheduled_jobs[job.id] = job
    except Exception:
        pass


def schedule_job(cron: str, prompt: str,
                 recurring: bool = True, durable: bool = True) -> CronJob | str:
    """注册新的 cron 任务"""
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable)
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()
    return job


def cancel_job(job_id: str) -> str:
    """取消 cron 任务"""
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()
    return f"Cancelled {job_id}"


def cron_scheduler_loop():
    """调度器主循环：每秒检查一次，将匹配的任务加入执行队列。
    使用 _last_fired 防止同一分钟内重复触发。
    一次性任务触发后自动移除。
    """
    while True:
        time.sleep(1)
        now = datetime.now()
        marker = now.strftime("%Y-%m-%d %H:%M")  # 精确到分钟的标记
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now) and _last_fired.get(job.id) != marker:
                        cron_queue.append(job)
                        _last_fired[job.id] = marker
                        if not job.recurring:
                            # 一次性任务：触发后移除
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:
                    print(f"  \033[31m[cron error] {job.id}: {e}\033[0m")


def consume_cron_queue() -> list[CronJob]:
    """消费（取出并清空）待执行的 cron 任务队列"""
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


def run_schedule_cron(cron: str, prompt: str,
                      recurring: bool = True, durable: bool = True) -> str:
    """工具入口：注册 cron 任务"""
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' -> {prompt}"


def run_list_crons() -> str:
    """工具入口：列出所有 cron 任务"""
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs."
    return "\n".join(
        f"  {job.id}: '{job.cron}' -> {job.prompt[:40]} "
        f"[{'recurring' if job.recurring else 'one-shot'}, "
        f"{'durable' if job.durable else 'session'}]"
        for job in jobs)


def run_cancel_cron(job_id: str) -> str:
    """工具入口：取消 cron 任务"""
    return cancel_job(job_id)


# 启动时加载持久化的 cron 任务，并启动后台调度线程
load_durable_jobs()
threading.Thread(target=cron_scheduler_loop, daemon=True).start()


# ── MCP 系统 (Model Context Protocol) ──
#
# MCP 被建模为延迟绑定的工具：先连接服务器，然后发现的服务器工具
# 以 mcp__server__tool 的命名格式合并到普通工具池中。
# 这是教学用的 mock 实现，真实场景会连接到实际的 MCP 服务器。

class MCPClient:
    """MCP 客户端：发现并调用 MCP 服务器上的工具（教学用 mock）"""

    def __init__(self, name: str):
        self.name = name
        self.tools: list[dict] = []           # 服务器提供的工具定义
        self._handlers: dict[str, callable] = {}  # 工具名 → 处理函数

    def register(self, tool_defs: list[dict],
                 handlers: dict[str, callable]):
        """注册工具定义和处理函数"""
        self.tools = tool_defs
        self._handlers = handlers

    def call_tool(self, tool_name: str, args: dict) -> str:
        """调用 MCP 工具"""
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return handler(**args)
        except Exception as e:
            return f"MCP error: {e}"


mcp_clients: dict[str, MCPClient] = {}  # 已连接的 MCP 客户端

_DISALLOWED_CHARS = re.compile(r'[^a-zA-Z0-9_-]')


def normalize_mcp_name(name: str) -> str:
    """将 MCP 名称中的非法字符替换为下划线（确保工具名合法）"""
    return _DISALLOWED_CHARS.sub('_', name)


def _mock_server_docs():
    """创建 mock 文档服务器：提供只读的搜索和版本查询工具"""
    client = MCPClient("docs")
    client.register(
        tool_defs=[
            {"name": "search", "description": "Search documentation. (readOnly)",
             "inputSchema": {"type": "object",
                             "properties": {"query": {"type": "string"}},
                             "required": ["query"]}},
            {"name": "get_version", "description": "Get API version. (readOnly)",
             "inputSchema": {"type": "object", "properties": {},
                             "required": []}},
        ],
        handlers={
            "search": lambda query: f"[docs] Found 3 results for '{query}'",
            "get_version": lambda: "[docs] API v2.1.0",
        })
    return client


def _mock_server_deploy():
    """创建 mock 部署服务器：提供部署触发和状态查询工具"""
    client = MCPClient("deploy")
    client.register(
        tool_defs=[
            {"name": "trigger",
             "description": "Trigger a deployment. (destructive — requires approval in real CC)",
             "inputSchema": {"type": "object",
                             "properties": {"service": {"type": "string"}},
                             "required": ["service"]}},
            {"name": "status", "description": "Check deployment status. (readOnly)",
             "inputSchema": {"type": "object",
                             "properties": {"service": {"type": "string"}},
                             "required": ["service"]}},
        ],
        handlers={
            "trigger": lambda service: f"[deploy] Triggered: {service}",
            "status": lambda service: f"[deploy] {service}: running (v1.4.2)",
        })
    return client


MOCK_SERVERS = {
    "docs": _mock_server_docs,
    "deploy": _mock_server_deploy,
}


def connect_mcp(name: str) -> str:
    """连接到 MCP 服务器，发现并注册其工具"""
    if name in mcp_clients:
        return f"MCP server '{name}' already connected"
    factory = MOCK_SERVERS.get(name)
    if not factory:
        available = ", ".join(MOCK_SERVERS.keys())
        return f"Unknown server '{name}'. Available: {available}"
    mcp_client = factory()
    mcp_clients[name] = mcp_client
    tool_names = [t["name"] for t in mcp_client.tools]
    print(f"  \033[31m[mcp] connected: {name} → {tool_names}\033[0m")
    return (f"Connected to MCP server '{name}'. "
            f"Discovered {len(mcp_client.tools)} tools: {', '.join(tool_names)}")


def assemble_tool_pool() -> tuple[list[dict], dict]:
    """合并内置工具 + 所有 MCP 工具到统一的工具池。
    MCP 工具以 mcp__server__tool 格式命名，避免与内置工具冲突。
    """
    tools = list(BUILTIN_TOOLS)
    handlers = dict(BUILTIN_HANDLERS)
    for server_name, mcp_client in mcp_clients.items():
        safe_server = normalize_mcp_name(server_name)
        for tool_def in mcp_client.tools:
            safe_tool = normalize_mcp_name(tool_def["name"])
            prefixed = f"mcp__{safe_server}__{safe_tool}"
            tools.append({
                "name": prefixed,
                "description": tool_def.get("description", ""),
                "input_schema": tool_def.get("inputSchema", {}),
            })
            # 使用默认参数捕获当前的 mcp_client 和 tool 名称
            handlers[prefixed] = (
                lambda *, c=mcp_client, t=tool_def["name"], **kw: c.call_tool(t, kw))
    return tools, handlers


# ── Lead 工作树工具 (Lead Worktree Tools) ──

def run_create_worktree(name: str, task_id: str = "") -> str:
    """工具入口：创建工作树"""
    return create_worktree(name, task_id)

def run_remove_worktree(name: str, discard_changes: bool = False) -> str:
    """工具入口：删除工作树"""
    return remove_worktree(name, discard_changes)

def run_keep_worktree(name: str) -> str:
    """工具入口：保留工作树供审查"""
    return keep_worktree(name)


# ── 工具处理器包装 (Tool Handler Wrappers) ──
#
# 这些是 lead agent 使用的工具处理器。每个函数都是底层实现的薄包装，
# 添加了错误处理和日志输出。

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """工具入口：创建任务"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """工具入口：列出所有任务"""
    tasks = list_tasks()
    if not tasks:
        return "No tasks."
    return "\n".join(
        f"  {t.id}: {t.subject} [{t.status}]"
        + (f" (wt:{t.worktree})" if t.worktree else "")
        for t in tasks)


def run_get_task(task_id: str) -> str:
    """工具入口：获取任务详情"""
    try:
        return get_task_json(task_id)
    except FileNotFoundError:
        return f"Error: task {task_id} not found"

def run_claim_task(task_id: str) -> str:
    """工具入口：认领任务"""
    try:
        return claim_task(task_id, owner="agent")
    except FileNotFoundError:
        return f"Error: task {task_id} not found"

def run_complete_task(task_id: str) -> str:
    """工具入口：完成任务"""
    try:
        return complete_task(task_id)
    except FileNotFoundError:
        return f"Error: task {task_id} not found"

def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    """工具入口：启动队友"""
    return spawn_teammate_thread(name, role, prompt)

def run_send_message(to: str, content: str) -> str:
    """工具入口：发送消息"""
    BUS.send("lead", to, content)
    return f"Sent to {to}"

def run_check_inbox() -> str:
    """工具入口：检查收件箱"""
    msgs = consume_lead_inbox(route_protocol=True)
    if not msgs:
        return "(inbox empty)"
    lines = []
    for m in msgs:
        meta = m.get("metadata", {})
        req_id = meta.get("request_id", "")
        tag = f" [{m['type']} req:{req_id}]" if req_id else f" [{m['type']}]"
        lines.append(f"  [{m['from']}]{tag} {m['content'][:200]}")
    return "\n".join(lines)

def run_connect_mcp(name: str) -> str:
    """工具入口：连接 MCP 服务器"""
    return connect_mcp(name)


# ── 工具定义 (Tool Definitions) ──
#
# 模型看到工具的 JSON Schema（BUILTIN_TOOLS），Python 执行对应的处理函数（BUILTIN_HANDLERS）。
# S20 保持两个表显式列出，这样每个新增的能力都在一个地方可见。
# MCP 工具会在运行时动态添加到这个池中。
BUILTIN_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"},
                                     "run_in_background": {"type": "boolean"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"},
                                     "offset": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
    {"name": "todo_write",
     "description": "Create and manage a task list for the current session.",
     "input_schema": {"type": "object",
                      "properties": {"todos": {"type": "array",
                          "items": {"type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {"type": "string",
                                                   "enum": ["pending", "in_progress", "completed"]}},
                                    "required": ["content", "status"]}}},
                      "required": ["todos"]}},
    {"name": "task",
     "description": "Launch a focused subagent. Returns only its final summary.",
     "input_schema": {"type": "object",
                      "properties": {"description": {"type": "string"}},
                      "required": ["description"]}},
    {"name": "load_skill",
     "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "compact",
     "description": "Summarize earlier conversation and continue with compacted context.",
     "input_schema": {"type": "object",
                      "properties": {"focus": {"type": "string"}},
                      "required": []}},
    {"name": "create_task", "description": "Create a task.",
     "input_schema": {"type": "object",
                      "properties": {"subject": {"type": "string"},
                                     "description": {"type": "string"},
                                     "blockedBy": {"type": "array",
                                                   "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_task", "description": "Get full task details.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task", "description": "Claim a pending task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task", "description": "Complete an in-progress task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "schedule_cron",
     "description": ("Schedule a cron job. cron is 5-field: min hour dom "
                     "month dow. For one-shot reminders, compute the target "
                     "minute and set recurring=false."),
     "input_schema": {"type": "object",
                      "properties": {"cron": {"type": "string"},
                                     "prompt": {"type": "string"},
                                     "recurring": {"type": "boolean"},
                                     "durable": {"type": "boolean"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons", "description": "List registered cron jobs.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "cancel_cron", "description": "Cancel a cron job by ID.",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
    {"name": "spawn_teammate", "description": "Spawn an autonomous teammate.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "role": {"type": "string"},
                                     "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "send_message", "description": "Send message to a teammate.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "check_inbox",
     "description": "Check inbox for messages and protocol responses.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "request_shutdown",
     "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"}},
                      "required": ["teammate"]}},
    {"name": "request_plan",
     "description": "Ask a teammate to submit a plan.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"},
                                     "task": {"type": "string"}},
                      "required": ["teammate", "task"]}},
    {"name": "review_plan",
     "description": "Approve or reject a submitted plan.",
     "input_schema": {"type": "object",
                      "properties": {"request_id": {"type": "string"},
                                     "approve": {"type": "boolean"},
                                     "feedback": {"type": "string"}},
                      "required": ["request_id", "approve"]}},
    {"name": "create_worktree",
     "description": "Create an isolated git worktree.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "task_id": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "remove_worktree",
     "description": "Remove a worktree. Refuses if changes exist.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "discard_changes": {"type": "boolean"}},
                      "required": ["name"]}},
    {"name": "keep_worktree",
     "description": "Keep a worktree for manual review.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "connect_mcp",
     "description": "Connect to an MCP server (docs, deploy) and discover tools.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
]

BUILTIN_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
    "todo_write": run_todo_write, "task": spawn_subagent,
    "load_skill": load_skill,
    "create_task": run_create_task, "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task, "complete_task": run_complete_task,
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
    "spawn_teammate": run_spawn_teammate,
    "send_message": run_send_message, "check_inbox": run_check_inbox,
    "request_shutdown": run_request_shutdown,
    "request_plan": run_request_plan, "review_plan": run_review_plan,
    "create_worktree": run_create_worktree,
    "remove_worktree": run_remove_worktree,
    "keep_worktree": run_keep_worktree,
    "connect_mcp": run_connect_mcp,
}


# ── 上下文管理 (Context) ──
#
# 上下文在每轮调用时从实时状态构建，包括记忆、MCP 连接和活跃队友。

MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"


def update_context(context: dict, messages: list) -> dict:
    """更新上下文：读取记忆文件、MCP 状态、活跃队友列表"""
    memories = ""
    if MEMORY_INDEX.exists():
        memories = MEMORY_INDEX.read_text()[:2000]  # 限制记忆大小
    return {
        "memories": memories,
        "connected_mcp": list(mcp_clients.keys()),
        "active_teammates": list(active_teammates.keys()),
    }


# ── Agent 主循环 (Agent Loop) ──
#
# 这是整个系统的核心。每一轮循环：
# 1. 注入定时任务和后台通知
# 2. 准备上下文（压缩）
# 3. 调用 LLM
# 4. 执行工具
# 5. 将结果反馈给模型
# 6. 重复直到模型不再调用工具

rounds_since_todo = 0        # 距离上次更新 todo 的轮数
agent_lock = threading.Lock()  # 保护 agent 循环的锁（cron 和 CLI 不能同时运行）


def prepare_context(messages: list) -> list:
    """每轮 LLM 调用前的上下文预算管道：按优先级依次压缩"""
    messages[:] = tool_result_budget(messages)  # 1. 持久化大输出
    messages[:] = snip_compact(messages)        # 2. 裁剪中间消息
    messages[:] = micro_compact(messages)       # 3. 替换旧工具结果
    if estimate_size(messages) > CONTEXT_LIMIT:
        messages[:] = compact_history(messages)  # 4. LLM 总结（最重）
    return messages


def build_user_content(results: list[dict]) -> list[dict]:
    """构建发送给模型的用户内容：工具结果 + 后台任务通知"""
    content = list(results)
    for note in collect_background_results():
        content.append({"type": "text", "text": note})
    return content


def inject_background_notifications(messages: list):
    """将已完成的后台任务通知注入消息历史"""
    notes = collect_background_results()
    if notes:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": note} for note in notes]})


def call_llm(messages: list, context: dict, tools: list,
             state: RecoveryState, max_tokens: int):
    """调用 LLM：组装系统提示 → 带重试的 API 调用"""
    system = assemble_system_prompt(context)
    return with_retry(
        lambda: client.messages.create(
            model=state.current_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens),
        state)


def agent_loop(messages: list, context: dict):
    """Agent 主循环：注入任务 → 准备上下文 → 调用 LLM → 执行工具 → 重复"""
    global rounds_since_todo
    tools, handlers = assemble_tool_pool()
    state = RecoveryState()
    max_tokens = DEFAULT_MAX_TOKENS

    while True:
        # ── 阶段 1: 注入定时任务和后台通知 ──
        fired = consume_cron_queue()
        for job in fired:
            messages.append({"role": "user",
                             "content": f"[Scheduled] {job.prompt}"})
            print(f"  \033[35m[cron inject] {job.prompt[:60]}\033[0m")

        inject_background_notifications(messages)

        # 每 3 轮提醒更新 todo
        if rounds_since_todo >= 3:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        # ── 阶段 2: 准备上下文（压缩）和更新工具池 ──
        prepare_context(messages)
        context = update_context(context, messages)
        tools, handlers = assemble_tool_pool()  # MCP 可能有新工具

        # ── 阶段 3: 调用 LLM ──
        try:
            response = call_llm(messages, context, tools, state, max_tokens)
        except Exception as e:
            if is_prompt_too_long_error(e) and not state.has_attempted_reactive_compact:
                messages[:] = reactive_compact(messages)
                state.has_attempted_reactive_compact = True
                continue  # 压缩后重试
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": f"[Error] {type(e).__name__}: {e}"}]})
            return

        # ── 阶段 4: 处理 max_tokens 截断 ──
        if response.stop_reason == "max_tokens":
            if not state.has_escalated:
                # 第一次截断：升级 token 上限
                max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(f"  \033[33m[max_tokens] retry with {max_tokens}\033[0m")
                continue
            messages.append({"role": "assistant", "content": response.content})
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                # 追加 continuation prompt 让模型继续
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                continue
            return  # 超过恢复次数，退出

        # ── 阶段 5: 正常响应处理 ──
        max_tokens = DEFAULT_MAX_TOKENS
        state.has_escalated = False
        messages.append({"role": "assistant", "content": response.content})
        if not has_tool_use(response.content):
            trigger_hooks("Stop", messages)  # 模型完成，触发停止钩子
            return

        # ── 阶段 6: 执行工具 ──
        results = []
        compacted_now = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            # 特殊工具：compact 直接压缩整个历史
            if block.name == "compact":
                messages[:] = compact_history(messages)
                messages.append({"role": "user",
                                 "content": "[Compacted. Continue with summarized context.]"})
                compacted_now = True
                break

            # PreToolUse 钩子（权限检查 + 日志）
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            # 后台任务：慢速操作在后台执行
            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block, handlers)
                output = (f"[Background task {bg_id} started] "
                          "Result will arrive as a task_notification.")
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})
                continue

            # 同步执行工具
            handler = handlers.get(block.name)
            output = call_tool_handler(handler, block.input, block.name)
            trigger_hooks("PostToolUse", block, output)
            print(str(output)[:300])

            # 跟踪 todo 更新轮数
            if block.name == "todo_write":
                rounds_since_todo = 0
            else:
                rounds_since_todo += 1

            results.append({"type": "tool_result",
                            "tool_use_id": block.id, "content": output})

        if compacted_now:
            continue  # compact 后重新开始循环

        # 将工具结果和后台通知一起发送给模型
        messages.append({"role": "user", "content": build_user_content(results)})


def print_turn_assistants(messages: list, turn_start: int):
    """打印本轮 assistant 的文本响应"""
    for msg in messages[turn_start:]:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block_type(block) == "text":
                terminal_print(block["text"] if isinstance(block, dict) else block.text)


def cron_autorun_loop(history: list, context: dict):
    """Cron 自动执行循环：在后台线程中运行，当有定时任务触发时自动执行 agent 循环。
    使用 agent_lock 确保与 CLI 主循环互斥。
    """
    while True:
        time.sleep(1)
        fired = consume_cron_queue()
        if not fired:
            continue
        with agent_lock:
            turn_start = len(history)
            for job in fired:
                history.append({"role": "user",
                                "content": f"[Scheduled] {job.prompt}"})
                terminal_print(
                    f"  \033[35m[cron auto] {job.prompt[:60]}\033[0m")
            agent_loop(history, context)
            context.update(update_context(context, history))
            print_turn_assistants(history, turn_start)


# ── CLI 入口 (Command Line Interface) ──
if __name__ == "__main__":
    CLI_ACTIVE = True
    print("s20: comprehensive agent")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []
    context = update_context({}, [])
    # 启动 cron 自动执行后台线程
    threading.Thread(target=cron_autorun_loop,
                     args=(history, context), daemon=True).start()
    while True:
        try:
            query = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        turn_start = len(history)
        history.append({"role": "user", "content": query})
        with agent_lock:
            agent_loop(history, context)
            context = update_context(context, history)
            print_turn_assistants(history, turn_start)

        # 每轮结束后检查收件箱（队友发来的消息）
        inbox = consume_lead_inbox(route_protocol=True)
        if inbox:
            def inbox_label(msg):
                req_id = msg.get("metadata", {}).get("request_id", "")
                suffix = f" req:{req_id}" if req_id else ""
                return f"{msg.get('type', 'message')}{suffix}"

            inbox_text = "\n".join(
                f"From {m['from']} [{inbox_label(m)}]: "
                f"{m['content'][:200]}" for m in inbox)
            history.append({"role": "user",
                            "content": f"[Inbox]\n{inbox_text}"})
        print()
