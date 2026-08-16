"""
U10 - Context Compaction（上下文压缩）
======================================

在大型语言模型调用之前，需插入四层压缩管道:

    L1: snip_compact      — 当消息数量超过50条时，删除中间的那些消息。
    L2: micro_compact     — 用占位符替换旧的 tool_results 文件。
    L3: tool_result_budget — 将大型结果持久化到磁盘上
    L4: compact_history   — LLM full summary (1 API call)

    Emergency: reactive_compact — 当 API 仍然返回 “prompt_too_long” 时

    ┌─────────────────────────────────────────────────────────────┐
    │  messages[]                                                 │
    │    ↓                                                        │
    │  L3 budget ─→ L1 snip ─→ L2 micro ─→ [token > threshold?]  │
    │                                      ├─ No  → LLM          │
    │                                      └─ Yes → L4 summary   │
    │                                              ↓              │
    │                                          LLM call           │
    │                                    [prompt_too_long?]        │
    │                                      └─ Yes → reactive      │
    └─────────────────────────────────────────────────────────────┘

Core principle: cheap first, expensive last.
Execution order matches CC source: budget → snip → micro → auto.
"""
import ast, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ── 初始化 Anthropic 客户端 ────────────────────────────────
client, MODEL = create_client()
WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
CURRENT_TODOS: list[dict] = []


# ── Frontmatter 解析 ─────────────────────────────────────────
# 用于解析 Markdown 文件头部的 YAML-like 元数据块（由 --- 分隔）
# 返回 (元数据字典, 正文内容) 的元组
def _parse_frontmatter(text: str) -> tuple[dict, str]:
    # 如果文本不以 "---" 开头，说明没有 frontmatter，直接返回原文
    if not text.startswith("---"):
        return {}, text
    # 按 "---" 分割，最多分成 3 部分：[空串, 元数据块, 正文]
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    # 逐行解析元数据块中的 key: value 对
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            # 去除首尾空白和引号
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()


# ── 技能注册表 ──────────────────────────────────────────────
# 存储所有已扫描技能的元数据，格式为 {技能名: {name, description, content}}
SKILL_REGISTRY: dict[str, dict] = {}


def _scan_skills():
    """扫描 skills/ 目录，读取每个技能的 SKILL.md 并注册到 SKILL_REGISTRY"""
    # 如果 skills 目录不存在，直接跳过
    if not SKILLS_DIR.exists():
        return
    # 遍历 skills 目录下的所有子目录（按名称排序）
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            # 读取 SKILL.md 原始内容
            raw = manifest.read_text()
            # 解析 frontmatter 元数据和正文
            meta, body = _parse_frontmatter(raw)
            # 优先使用 frontmatter 中的 name，否则使用目录名
            name = meta.get("name", d.name)
            # 优先使用 frontmatter 中的 description，否则取第一行作为描述
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            # 将技能信息存入注册表
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}


# 模块加载时立即扫描并注册所有技能
_scan_skills()


def list_skills() -> str:
    """列出所有已注册技能的摘要（名称 + 描述），用于系统提示词中的技能目录"""
    # 如果没有注册任何技能，返回占位提示
    if not SKILL_REGISTRY:
        return "(no skills found)"
    # 将每个技能格式化为 Markdown 列表项：**名称**: 描述
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())


def load_skill(name: str) -> str:
    """按名称加载单个技能的完整内容（SKILL.md 原文），供 Agent 按需调用"""
    # 从注册表中查找技能，找不到则返回错误提示
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    # 返回技能的完整 Markdown 内容
    return skill["content"]


def build_system() -> str:
    """构建系统提示词：包含工作目录信息、可用技能目录，以及调用指引"""
    # 获取技能目录的摘要文本
    catelog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}"
        f"Skills available: \n{catelog}\n"
        "Use load_skill to get full details when needed."
    )


# 模块加载时一次性构建系统提示词，后续所有 LLM 调用复用同一份
SYSTEM = build_system()

# subagent gets its own system prompt — no compact, no skill loading
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# ═══════════════════════════════════════════════════════════
#  FROM s02-s07 (unchanged): Basic Tools
# ═══════════════════════════════════════════════════════════
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR): raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines): lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path);
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content);
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text: return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def _normalize_todos(todos):
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
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None


def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"


def extract_text(content) -> str:
    if not isinstance(content, list): return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")


# ═══════════════════════════════════════════════════════════
#  Subagent
# ═══════════════════════════════════════════════════════════
SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                       "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
]
SUB_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write,
                "edit_file": run_edit, "glob": run_glob}


def spawn_subagent(description: str) -> str:
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]
    for _ in range(30):
        response = client.messages.create(model=MODEL, system=SUB_SYSTEM,
                                          messages=messages, tools=SUB_TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": str(blocked)})
                    continue
                handler = SUB_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                trigger_hooks("PostToolUse", block, output)
                print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
    result = extract_text(messages[-1]["content"])
    if not result:
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result


# ═══════════════════════════════════════════════════════════
#  NEW: Four-Layer Compaction Pipeline（四层压缩管道）
# ═══════════════════════════════════════════════════════════
# 核心原则：先廉价后昂贵（cheap first, expensive last）
# 执行顺序：L3 budget → L1 snip → L2 micro → L4 summary → reactive(应急)

CONTEXT_LIMIT = 50000        # 触发 L4 自动摘要的字符数阈值
KEEP_RECENT = 3              # L2 保留最近 N 条 tool_result 不压缩
PERSIST_THRESHOLD = 30000    # L3 单条 tool_result 超过此字节数则落盘


# ── 工具函数 ─────────────────────────────────────────────────

def estimate_size(msgs):
    """估算消息列表的总字符数（作为 token 使用量的粗略代理）"""
    return len(str(msgs))


def _block_type(block):
    """统一获取 content block 的 type 字段，兼容 dict 和对象两种格式"""
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _message_has_tool_use(msg):
    """判断一条 assistant 消息中是否包含 tool_use 类型的 block"""
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(block) == "tool_use" for block in content)


def _is_tool_result_message(msg):
    """判断一条 user 消息中是否包含 tool_result 类型的 block"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result"
               for block in content)


# ── L1: snip_compact —— 消息数量超限时裁剪中间历史 ───────────
# 策略：保留头部 3 条 + 尾部 N-3 条，中间用占位符替代
# 边界对齐：确保不在 tool_use / tool_result 对的中间截断（否则 API 会报错）

def snip_compact(messages, max_messages=50):
    """L1: 消息数超过 max_messages 时，裁掉中间消息，保留头尾"""
    # 消息数未超限，直接返回
    if len(messages) <= max_messages: return messages
    # 计算保留区间：头部保留 3 条，尾部保留 max_messages - 3 条
    keep_head, keep_tail = 3, max_messages - 3
    head_end, tail_start = keep_head, len(messages) - keep_tail

    # ── 边界对齐：头侧 ──
    # 如果头部最后一条是 assistant 的 tool_use，则向后扩展跳过紧随的 tool_result 消息
    # 避免截断后出现孤立的 tool_use（没有对应的 tool_result，API 会拒绝）
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1

    # ── 边界对齐：尾侧 ──
    # 如果尾部第一条是 tool_result，且前一条是 tool_use，将切割点前移一条
    # 避免截断后出现孤立的 tool_result（没有对应的 tool_use，API 会拒绝）
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1

    # 边界对齐后可能无内容可裁，直接返回
    if head_end >= tail_start:
        return messages

    # 执行裁剪：头 + 占位符 + 尾
    snipped = tail_start - head_end
    return messages[:head_end] + [{"role": "user", "content": f"[snipped {snipped} messages]"}] + messages[tail_start:]


# ── L2: micro_compact —— 旧 tool_result 占位符替换 ────────────
# 策略：保留最近 KEEP_RECENT 条 tool_result 的完整内容，更早的替换为短占位符
# 条件：仅替换内容超过 120 字符的 tool_result（短内容压缩收益低）

def collect_tool_results(messages):
    """收集所有 user 消息中的 tool_result block，返回 (msg_index, block_index, block) 列表"""
    blocks = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list): continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))
    return blocks


def micro_compact(messages):
    """L2: 将较早的 tool_result 内容替换为占位符，保留最近 KEEP_RECENT 条不动"""
    tool_results = collect_tool_results(messages)
    # tool_result 总数未超保留阈值，无需压缩
    if len(tool_results) <= KEEP_RECENT: return messages
    # 对 KEEP_RECENT 条之前的所有 tool_result 进行占位符替换
    for _, _, block in tool_results[:-KEEP_RECENT]:
        # 仅压缩内容较长的（>120 字符），短内容保留原样
        if len(block.get("content", "")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


# ── L3: tool_result_budget —— 大结果持久化到磁盘 ──────────────
# 策略：当最近一条 user 消息中所有 tool_result 的总大小超过 max_bytes 时，
#       按大小降序逐个将最大的结果写入磁盘文件，消息中只保留路径 + 预览

def persist_large_output(tool_use_id, output):
    """将单条大型 tool_result 输出写入磁盘，返回包含路径和预览的替代内容"""
    # 输出未超阈值，不落盘
    if len(output) <= PERSIST_THRESHOLD: return output
    # 确保输出目录存在
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # 以 tool_use_id 为文件名，避免冲突
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    # 首次写入（已有文件则不覆盖，避免重复 I/O）
    if not path.exists(): path.write_text(output)
    # 返回精简内容：文件路径 + 前 2000 字符预览
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"


def tool_result_budget(messages, max_bytes=200_000):
    """L3: 检查最近一条 user 消息中 tool_result 的总大小，超限时将最大的结果落盘"""
    # 只检查最后一条消息（新到的工具结果集中在这里）
    last = messages[-1] if messages else None
    if not last or last.get("role") != "user" or not isinstance(last.get("content"), list): return messages

    # 收集该消息中所有 tool_result block 及其索引
    blocks = [(i, b) for i, b in enumerate(last["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]
    # 计算所有 tool_result 的总字符数
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes: return messages

    # 按内容大小降序排列，优先压缩最大的结果
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True)
    for _, block in ranked:
        # 总量已降到阈值内，停止压缩
        if total <= max_bytes: break
        content = str(block.get("content", ""))
        # 单条内容未超落盘阈值，跳过（小内容不值得一次磁盘 I/O）
        if len(content) <= PERSIST_THRESHOLD: continue
        # 将大结果落盘，原地替换 block 内容
        tid = block.get("tool_use_id", "unknown")
        block["content"] = persist_large_output(tid, content)
        # 重新计算总量（blocks 列表中已原地修改）
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return messages


# ── L4: compact_history —— LLM 全量摘要（消耗 1 次 API 调用）──
# 策略：调用 LLM 对整段对话历史生成摘要，然后用摘要替换所有历史消息
# 前置保护：调用前先将完整历史写入磁盘 transcript，防止数据丢失

def write_transcript(messages):
    """将完整对话历史写入 JSONL 文件，作为压缩前的备份"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    # 用 Unix 时间戳命名，保证唯一性
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages: f.write(json.dumps(msg, default=str) + "\n")
    return path


def summarize_history(messages):
    """调用 LLM 对对话历史生成结构化摘要（保留目标、决策、文件、剩余工作、约束）"""
    # 截断到 80000 字符，避免摘要请求本身超 token 限制
    conversation = json.dumps(messages, default=str)[:80000]
    # 摘要提示词：要求保留 5 个关键维度
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)
    response = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    # 从响应中提取所有 text block 并拼接
    return "\n".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text").strip() or "(empty summary)"


def compact_history(messages):
    """L4: 执行全量压缩 — 备份 transcript → 生成摘要 → 用摘要替换全部历史"""
    # 先备份完整历史到磁盘
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    # 调用 LLM 生成摘要
    summary = summarize_history(messages)
    # 用一条包含摘要的 user 消息替换整个消息列表
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# ── Emergency: reactive_compact —— API 报错时的应急压缩 ──────
# 触发条件：API 返回 prompt_too_long 错误
# 策略：比 L4 更激进 — 只对"旧消息"生成摘要，保留最近 5 条消息原文
#       这样即使摘要丢失细节，最近的上下文仍然完整

def reactive_compact(messages):
    """应急压缩：API 返回 prompt_too_long 时，摘要旧消息 + 保留最近 5 条原文"""
    # 先备份完整历史
    transcript = write_transcript(messages)
    # 保留最近 5 条消息的原文（tail）
    tail_start = max(0, len(messages) - 5)
    # 边界对齐：确保不在 tool_use / tool_result 对的中间截断
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    # 仅对 tail 之前的消息生成摘要（比全量摘要更快、更短）
    summary = summarize_history(messages[:tail_start])
    # 摘要 + 原始 tail = 精简后的消息列表
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]


# ═══════════════════════════════════════════════════════════
#  FROM s07: Tool Definitions
# ═══════════════════════════════════════════════════════════

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                       "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "todo_write", "description": "Create and manage a task list for your current coding session.",
     "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object",
                                                                                            "properties": {"content": {
                                                                                                "type": "string"},
                                                                                                "status": {
                                                                                                    "type": "string",
                                                                                                    "enum": [
                                                                                                        "pending",
                                                                                                        "in_progress",
                                                                                                        "completed"]}},
                                                                                            "required": ["content",
                                                                                                         "status"]}}},
                      "required": ["todos"]}},
    {"name": "task", "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
     "input_schema": {"type": "object", "properties": {"description": {"type": "string"}},
                      "required": ["description"]}},
    {"name": "load_skill", "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    # s08 change: new compact tool — triggers compact_history, not a no-op
    {"name": "compact", "description": "Summarize earlier conversation to free context space.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
    "task": spawn_subagent, "load_skill": load_skill,
}

# FROM s04 (unchanged): Hooks
HOOKS = {"PreToolUse": [], "PostToolUse": []}


def trigger_hooks(event, *args):
    for cb in HOOKS[event]:
        r = cb(*args)
        if r is not None: return r
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown"]


def permission_hook(block):
    if block.name == "bash":
        for p in DENY_LIST:
            if p in block.input.get("command", ""): return "Permission denied"
    return None


def log_hook(block):
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None


HOOKS["PreToolUse"].append(permission_hook)
HOOKS["PreToolUse"].append(log_hook)

# ═══════════════════════════════════════════════════════════
#  agent_loop — run compaction pipeline before LLM
# ═══════════════════════════════════════════════════════════
MAX_REACTIVE_RETRIES = 1  # Reactive Compact 的重试次数限制


def agent_loop(messages: list):
    """Agent 主循环：发送消息 → 处理工具调用 → 循环直到模型停止调用工具"""
    reactive_retries = 0  # Reactive Compact 的当前重试计数

    while True:
        # ── 阶段 1: 三层预处理（零 API 调用，先廉价后昂贵）──
        messages[:] = tool_result_budget(messages)  # L3: 将大型 tool_result 持久化到磁盘，用路径占位
        messages[:] = snip_compact(messages)        # L1: 消息数超阈值时裁剪中间历史
        messages[:] = micro_compact(messages)       # L2: 将旧 tool_result 替换为简短占位符

        # ── 阶段 2: 自动摘要（仅在预处理后仍然超限时触发，消耗 1 次 API 调用）──
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)  # L4: 调用 LLM 对整段历史做摘要压缩

        # ── 阶段 3: 调用 LLM ──
        try:
            response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS,
                                              max_tokens=8000)
            reactive_retries = 0  # API 调用成功，重置重试计数
        except Exception as e:
            # ── 阶段 4: 应急压缩（API 返回 prompt_too_long 时的最后一道防线）──
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(
                    e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                messages[:] = reactive_compact(messages)  # 激进压缩：只保留摘要 + 最近几轮
                reactive_retries += 1
                continue  # 压缩后重新进入循环，重试 API 调用
            raise  # 非 token 超限错误，或重试次数已用完，向上抛出

        # ── 阶段 5: 处理模型响应 ──
        # 将模型的回复追加到消息历史
        messages.append({"role": "assistant", "content": response.content})

        # 如果模型没有请求调用任何工具（stop_reason == "end_turn"），对话结束，退出循环
        if response.stop_reason != "tool_use": return

        # ── 阶段 6: 执行工具调用 ──
        results = []
        for block in response.content:
            # 跳过文本块，只处理 tool_use 类型的块
            if block.type != "tool_use": continue
            # 青色打印正在执行的工具名称，便于调试观察
            print(f"\033[36m> {block.name}\033[0m")

            # 特殊处理：模型主动请求 compact 工具 → 执行 L4 摘要并中断当前轮次
            if block.name == "compact":
                messages[:] = compact_history(messages)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "[Compacted. Conversation history has been summarized.]"})
                messages.append({"role": "user", "content": results})
                break  # 跳出工具循环，回到主循环重新开始（已压缩过的消息）

            # PreToolUse 钩子：执行前拦截（权限校验、参数修改等）
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                # 钩子返回阻止原因，作为工具结果反馈给模型，不实际执行
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(blocked)})
                continue

            # 从工具注册表查找处理器，找不到则返回未知工具提示
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknowwn: {block.name}"

            # PostToolUse 钩子：执行后处理（日志记录、格式化等）
            trigger_hooks("PostToolUse", block, output)
            # 截断打印输出（最多 200 字符），避免刷屏
            print(str(output)[:200])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        else:
            # for 循环正常结束（没有 break）→ 将工具结果作为 user 消息发给模型，继续下一轮
            messages.append({"role": "user", "contnet": results})
            continue
        # for 循环被 break 中断（compact 工具触发）→ 也继续下一轮，但结果已在 break 内处理
        continue


if __name__ == "__main__":
    print("s08: Context Compact — four-layer compaction pipeline")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []
    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""): break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text": print(block.text)
        print()
