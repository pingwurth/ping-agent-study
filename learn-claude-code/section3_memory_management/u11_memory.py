"""
 Memory System

Persistent, cross-session knowledge for the coding agent.

Storage:
    .memory/
      MEMORY.md          ← index (one line per memory, ≤200 lines)
      feedback_tabs.md    ← individual memory files (Markdown + YAML frontmatter)
      user_profile.md
      project_facts.md

Flow in agent_loop:
    1. Load MEMORY.md index into SYSTEM prompt (cheap, always present)
    2. Select relevant memories by filename/description → inject content
    3. Run compression pipeline from s08
    4. After each turn ends → extract new memories from original messages
    5. Periodically consolidate (Dream)
"""
import ast, json, os, subprocess, sys, time
import re
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ── 初始化 Anthropic 客户端 ────────────────────────────────
client, MODEL = create_client()
WORKDIR = Path.cwd()

# ── Memory 系统目录结构 ────────────────────────────────────
# .memory/
#   ├── MEMORY.md              ← 索引文件（每条记忆一行，注入 system prompt）
#   ├── user-preference.md     ← 单个记忆文件（Markdown + YAML frontmatter）
#   ├── project-fact.md
#   └── ...
MEMORY_DIR = WORKDIR / ".memory";
MEMORY_DIR.mkdir(exist_ok=True)
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"  # 索引文件：轻量级，始终注入 system prompt

# 其他工作目录（与 s08 共用）
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"

# ═══════════════════════════════════════════════════════════
#  NEW: Memory System（记忆系统）
# ═══════════════════════════════════════════════════════════
# 三层记忆架构:
#   1. System Prompt  — 每轮注入，轻量索引（MEMORY.md）
#   2. Context Window — 按需加载相关记忆全文（select_relevant_memories）
#   3. Memory Files   — 永久存储在磁盘（.memory/*.md）
#
# 设计原则: 轻量、可读、可编辑 — 纯 Markdown 文件，不是向量数据库

# 记忆类型分类（用于 extract_memories 的 LLM 提示词）
MEMORY_TYPES = ["user", "feedback", "project", "reference"]
#   user      — 用户偏好（如"我喜欢用 tab 缩进"）
#   feedback  — 对 Agent 行为的反馈（如"不要自动提交"）
#   project   — 项目事实（如"本项目使用 Python 3.12"）
#   reference — 外部资源指针（如"API 文档在 docs.example.com"）


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 Markdown 文件头部的 YAML-like frontmatter（由 --- 分隔）"""
    # 没有 frontmatter 标记，直接返回空元数据 + 原文
    if not text.startswith("---"):
        return {}, text
    # 按 "---" 分割为 [空串, 元数据块, 正文] 三部分
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    # 逐行解析 key: value 对
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()


def write_memory_file(name: str, mem_type: str, description: str, body: str):
    """写入单个记忆文件（含 YAML frontmatter），然后重建索引"""
    # 将名称转为 kebab-case 文件名（空格和斜杠都转连字符）
    slug = name.lower().replace(" ", "-").replace("/", "-")
    filename = f"{slug}.md"
    filepath = MEMORY_DIR / filename
    # 写入 frontmatter（name/description/type）+ 正文
    filepath.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n"
    )
    # 每次写入后重建索引，保持 MEMORY.md 与磁盘文件同步
    _rebuild_index()
    return filepath


def _rebuild_index():
    """从所有记忆文件重建 MEMORY.md 索引（每条一行：名称 + 描述）"""
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        # 跳过索引文件本身
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", f.stem)
        # 描述优先取 frontmatter，否则取正文第一行（截断到 80 字符）
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}]({f.name}) — {desc}")
    # 有内容则写入索引，否则写空文件
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "")


def read_memory_index() -> str:
    """读取 MEMORY.md 索引内容（每轮注入 system prompt，成本极低）"""
    if not MEMORY_INDEX.exists():
        return ""
    text = MEMORY_INDEX.read_text().strip()
    return text if text else ""


def read_memory_file(filename: str) -> str | None:
    """按文件名读取单个记忆文件的完整内容（含 frontmatter）"""
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()


def list_memory_files() -> list[dict]:
    """列出所有记忆文件及其元数据，返回 [{filename, name, description, type, body}]"""
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        # 跳过索引文件
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        result.append({
            "filename": f.name,                    # 磁盘文件名（如 "user-tabs.md"）
            "name": meta.get("name", f.stem),      # 记忆名称（frontmatter 中的 name）
            "description": meta.get("description", ""),  # 一行摘要
            "type": meta.get("type", "user"),      # 类型：user/feedback/project/reference
            "body": body,                          # 正文内容
        })
    return result


def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    """根据最近对话内容，选择相关的记忆文件名列表。
    策略：优先用 LLM 从目录中挑选，失败则回退到关键词匹配。
    """
    files = list_memory_files()
    if not files:
        return []

    # ── 收集最近的用户消息文本（最多 3 条）作为匹配上下文 ──
    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # content 可能是 list[Block]，需要提取 text block
            if isinstance(content, list):
                content = " ".join(
                    str(getattr(b, "text", "")) for b in content
                    if getattr(b, "type", None) == "text"
                )
            if isinstance(content, str):
                recent_texts.append(content)
            if len(recent_texts) >= 3:
                break
    # 反转为时间顺序，截断到 2000 字符避免 prompt 过长
    recent = " ".join(reversed(recent_texts))[:2000]

    if not recent.strip():
        return []

    # ── 构建记忆目录（索引 + 描述），供 LLM 选择 ──
    catalog_lines = []
    for i, f in enumerate(files):
        catalog_lines.append(f"{i}: {f['name']} — {f['description']}")
    catalog = "\n".join(catalog_lines)

    # ── LLM 选择：让模型返回相关记忆的索引数组 ──
    prompt = (
        "Given the recent conversation and the memory catalog below, "
        "select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n\n"
        f"Recent conversation:\n{recent}\n\n"
        f"Memory catalog:\n{catalog}"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = extract_text(response.content).strip()
        # 从响应中提取 JSON 数组（如 [0, 3]）
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            indices = json.loads(match.group())
            selected = []
            for idx in indices:
                # 校验索引合法性
                if isinstance(idx, int) and 0 <= idx < len(files):
                    selected.append(files[idx]["filename"])
                    if len(selected) >= max_items:
                        break
            return selected
    except Exception:
        pass  # LLM 调用失败，静默回退到关键词匹配

    # ── 回退策略：关键词匹配（提取 >3 字符的词，在 name+description 中搜索）──
    keywords = [w.lower() for w in recent.split() if len(w) > 3]
    selected = []
    for f in files:
        text = (f["name"] + " " + f["description"]).lower()
        if any(kw in text for kw in keywords):
            selected.append(f["filename"])
            if len(selected) >= max_items:
                break
    return selected


def load_memories(messages: list) -> str:
    """加载相关记忆内容，用 XML 标签包裹后注入上下文"""
    # 通过 LLM 选择（或关键词匹配）获取相关记忆文件名
    selected_files = select_relevant_memories(messages)
    if not selected_files:
        return ""

    # 用 <relevant_memories> 标签包裹，便于模型识别这是外部记忆而非用户输入
    parts = ["<relevant_memories>"]
    for filename in selected_files:
        content = read_memory_file(filename)
        if content:
            parts.append(content)
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)


def extract_memories(messages: list):
    """从最近对话中提取新记忆。每轮对话结束后调用（消耗 1 次 API 调用）"""
    # ── 收集最近 10 条消息的文本内容 ──
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        # content 可能是 list[Block]，提取 text block
        if isinstance(content, list):
            content = " ".join(
                str(getattr(b, "text", "")) for b in content
                if getattr(b, "type", None) == "text"
            )
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    dialogue = "\n".join(dialogue_parts)

    if not dialogue.strip():
        return

    # ── 获取已有记忆列表，避免重复提取 ──
    existing = list_memory_files()
    existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"

    # ── LLM 提示词：要求返回 JSON 数组，每项包含 name/type/description/body ──
    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (user preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=800
        )
        text = extract_text(response.content).strip()
        # 从响应中提取 JSON 数组
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        # ── 逐条写入新记忆文件 ──
        count = 0
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")  # 无名称则用时间戳
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass  # 提取失败不影响主流程


# ── Dream: 记忆合并/整理（类似人类睡眠时的记忆整理）─────────
CONSOLIDATE_THRESHOLD = 10  # 记忆文件数达到此阈值时触发合并


def consolidate_memories():
    """合并重复/过时的记忆。当记忆文件数 ≥ CONSOLIDATE_THRESHOLD 时触发。
    类似"睡眠整理"：LLM 审视所有记忆，合并重复项，删除过时信息。
    """
    files = list_memory_files()
    # 未达阈值，跳过合并
    if len(files) < CONSOLIDATE_THRESHOLD:
        return

    # ── 构建所有记忆的完整目录（含正文），供 LLM 审视 ──
    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )

    # ── LLM 合并提示词：4 条规则 ──
    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"           # 合并重复记忆
        "2. Remove outdated/contradicted memories\n" # 删除过时或矛盾的记忆
        "3. Keep the total under 30 memories\n"     # 总数控制在 30 条以内
        "4. Preserve important user preferences above all\n"  # 用户偏好优先保留
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"  # 截断到 16K 字符，避免 prompt 过长
    )

    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=3000
        )
        text = extract_text(response.content).strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())

        # ── 删除所有旧记忆文件（保留 MEMORY.md 索引）──
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()

        # ── 写入合并后的新记忆 ──
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)

        print(f"\n\033[33m[Memory: consolidated {len(files)} → {len(items)} memories]\033[0m")
    except Exception:
        pass  # 合并失败不影响主流程


# ── System Prompt 构建 ────────────────────────────────────────

def build_system() -> str:
    """构建主 Agent 的 system prompt，注入记忆索引作为轻量级上下文"""
    # 读取 MEMORY.md 索引（每条记忆一行，成本极低）
    index = read_memory_index()
    memories_section = f"\n\nMemories available:\n{index}" if index else ""
    return (
        f"You are a coding agent at {WORKDIR}."
        f"{memories_section}\n"
        "Relevant memories are injected below. Respect user preferences from memory.\n"
        "When the user says 'remember' or expresses a clear preference, extract it as a memory."
    )


# 子 Agent 的 system prompt（简化版，不注入记忆，不递归委派）
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# ═══════════════════════════════════════════════════════════
#  FROM s02-s08 (skeleton): Basic tools（基础工具集）
# ═══════════════════════════════════════════════════════════
# 这些是 Agent 可调用的基础文件/命令操作工具（与 s02-s08 共用）


def safe_path(p: str) -> Path:
    """路径安全校验：确保解析后的路径仍在工作目录内（防止路径遍历攻击）"""
    path = (WORKDIR / p).resolve()
    # 检查解析后的路径是否是 WORKDIR 的子路径
    if not path.is_relative_to(WORKDIR): raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令，返回 stdout+stderr（截断到 50K 字符，超时 120s）"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容，可选限制行数"""
    try:
        lines = safe_path(path).read_text().splitlines()
        # 限制行数时，末尾附加省略提示
        if limit and limit < len(lines): lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件内容（自动创建父目录）"""
    try:
        file_path = safe_path(path);
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content);
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的文本（只替换第一次出现）"""
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        # 查找目标文本不存在则报错
        if old_text not in text: return f"Error: text not found in {path}"
        # 只替换第一次出现（count=1），避免误替换多处
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """查找匹配 glob 模式的文件（限制在工作目录内）"""
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            # 二次校验：确保匹配结果仍在工作目录内
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def extract_text(content) -> str:
    """从 API 响应的 content 列表中提取所有 text block 并拼接"""
    if not isinstance(content, list): return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")


# ── Subagent（子 Agent，简化版）──────────────────────────────
# 子 Agent 拥有独立的 system prompt（SUB_SYSTEM）和精简工具集
# 不能递归委派（不包含 "task" 工具），避免无限嵌套

# 子 Agent 可用工具：仅 bash/read/write 三个基础工具
SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
]
SUB_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write}


def spawn_subagent(description: str) -> str:
    """启动子 Agent 执行子任务，返回其最终文本结果"""
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]
    # 最多运行 30 轮，防止子 Agent 陷入死循环
    for _ in range(30):
        response = client.messages.create(model=MODEL, system=SUB_SYSTEM,
                                          messages=messages, tools=SUB_TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        # 模型停止调用工具 → 任务完成
        if response.stop_reason != "tool_use": break
        # 执行子 Agent 请求的工具调用
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = SUB_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                # 灰色打印子 Agent 的工具调用（区分于主 Agent 的青色）
                print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

    # ── 提取子 Agent 的最终结果 ──
    # 优先取最后一条消息的 text block
    result = extract_text(messages[-1]["content"])
    if not result:
        # 最后一条没有文本（可能是工具结果），向前查找最后一条 assistant 的文本
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result: break
        # 30 轮后仍无结果，返回超时提示
        if not result: result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result


# ═══════════════════════════════════════════════════════════
#  FROM s08 (skeleton): Compaction pipeline（四层压缩管道）
# ═══════════════════════════════════════════════════════════
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
    """判断 assistant 消息中是否包含 tool_use block"""
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(block) == "tool_use" for block in content)


def _is_tool_result_message(msg):
    """判断 user 消息中是否包含 tool_result block"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


# ── L1: snip_compact —— 消息数量超限时裁剪中间历史 ───────────

def snip_compact(msgs, mx=50):
    """L1: 消息数超过 mx 时，保留头 3 条 + 尾 (mx-3) 条，中间用占位符替代"""
    if len(msgs) <= mx: return msgs
    head_end, tail_start = 3, len(msgs) - (mx - 3)
    # ── 边界对齐：头侧 ──
    # 如果头部最后一条是 tool_use，向后跳过紧随的 tool_result（防止孤立）
    if head_end > 0 and _message_has_tool_use(msgs[head_end - 1]):
        while head_end < len(msgs) and _is_tool_result_message(msgs[head_end]):
            head_end += 1
    # ── 边界对齐：尾侧 ──
    # 如果尾部第一条是 tool_result，且前一条是 tool_use，切割点前移（防止孤立）
    if (tail_start > 0 and tail_start < len(msgs)
            and _is_tool_result_message(msgs[tail_start])
            and _message_has_tool_use(msgs[tail_start - 1])):
        tail_start -= 1
    # 边界对齐后可能无内容可裁
    if head_end >= tail_start:
        return msgs
    # 执行裁剪：头 + 占位符 + 尾
    return msgs[:head_end] + [{"role": "user", "content": f"[snipped {tail_start - head_end} msgs]"}] + msgs[
        tail_start:]


# ── L2: micro_compact —— 旧 tool_result 占位符替换 ────────────

def collect_tool_results(msgs):
    """收集所有 user 消息中的 tool_result block，返回 (msg_index, block_index, block) 列表"""
    blocks = []
    for mi, msg in enumerate(msgs):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list): continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result": blocks.append((mi, bi, block))
    return blocks


def micro_compact(msgs):
    """L2: 将较早的 tool_result 内容替换为占位符，保留最近 KEEP_RECENT 条不动"""
    tr = collect_tool_results(msgs)
    # tool_result 总数未超保留阈值，无需压缩
    if len(tr) <= KEEP_RECENT: return msgs
    # 对 KEEP_RECENT 条之前的所有 tool_result 进行占位符替换（仅压缩 >120 字符的）
    for _, _, b in tr[:-KEEP_RECENT]:
        if len(b.get("content", "")) > 120: b["content"] = "[Earlier tool result compacted.]"
    return msgs


# ── L3: tool_result_budget —— 大结果持久化到磁盘 ──────────────

def persist_large(tid, out):
    """将单条大型 tool_result 输出写入磁盘，返回包含路径和预览的替代内容"""
    if len(out) <= PERSIST_THRESHOLD: return out
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = TOOL_RESULTS_DIR / f"{tid}.txt"
    if not p.exists(): p.write_text(out)  # 首次写入，已有文件不覆盖
    return f"<persisted-output>\nFull: {p}\nPreview:\n{out[:2000]}\n</persisted-output>"


def tool_result_budget(msgs, mx=200_000):
    """L3: 检查最近一条 user 消息中 tool_result 总大小，超限时将最大的结果落盘"""
    last = msgs[-1] if msgs else None
    if not last or last.get("role") != "user" or not isinstance(last.get("content"), list): return msgs
    # 收集该消息中所有 tool_result block
    blocks = [(i, b) for i, b in enumerate(last["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= mx: return msgs
    # 按大小降序排列，优先压缩最大的结果
    for _, block in sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True):
        if total <= mx: break
        c = str(block.get("content", ""))
        if len(c) <= PERSIST_THRESHOLD: continue  # 小内容不值得一次磁盘 I/O
        block["content"] = persist_large(block.get("tool_use_id", "?"), c)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return msgs


# ── L4: compact_history —— LLM 全量摘要（消耗 1 次 API 调用）──

def write_transcript(msgs):
    """将完整对话历史写入 JSONL 文件，作为压缩前的备份"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    p = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with p.open("w") as f:
        for m in msgs: f.write(json.dumps(m, default=str) + "\n")
    return p


def summarize_history(msgs):
    """调用 LLM 对对话历史生成结构化摘要（保留目标、发现、文件变更、剩余工作、约束）"""
    conv = json.dumps(msgs, default=str)[:80000]  # 截断到 80K 字符
    r = client.messages.create(model=MODEL, messages=[{"role": "user", "content":
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve: 1. current goal, 2. key findings, 3. files changed, 4. remaining work, 5. user constraints.\n\n" + conv}],
                               max_tokens=2000)
    return extract_text(r.content).strip()


def compact_history(msgs):
    """L4: 执行全量压缩 — 备份 transcript → 生成摘要 → 用摘要替换全部历史"""
    write_transcript(msgs)
    summary = summarize_history(msgs)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# ── Emergency: reactive_compact —— API 报错时的应急压缩 ──────

def reactive_compact(msgs):
    """应急压缩：API 返回 prompt_too_long 时，摘要旧消息 + 保留最近 5 条原文"""
    write_transcript(msgs)
    tail_start = max(0, len(msgs) - 5)
    # 边界对齐：确保不在 tool_use/tool_result 对的中间截断
    if (tail_start > 0 and tail_start < len(msgs)
            and _is_tool_result_message(msgs[tail_start])
            and _message_has_tool_use(msgs[tail_start - 1])):
        tail_start -= 1
    # 仅对旧消息生成摘要，保留最近 5 条原文
    summary = summarize_history(msgs[:tail_start])
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *msgs[tail_start:]]


# ═══════════════════════════════════════════════════════════
#  Tool Definitions（工具定义，精简版 — 聚焦记忆系统）
# ═══════════════════════════════════════════════════════════
# 与 s02-s08 相比，减少了工具数量，专注于记忆 + 基础操作

TOOLS = [
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
    {"name": "task", "description": "Launch a subagent to handle a subtask.",
     "input_schema": {"type": "object", "properties": {"description": {"type": "string"}},
                      "required": ["description"]}},
]

# 工具名 → 处理函数的映射表
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "task": spawn_subagent,
}

# ═══════════════════════════════════════════════════════════
#  agent_loop — s09: 注入记忆 + 每轮结束后提取新记忆
# ═══════════════════════════════════════════════════════════
# 与 s08 的 agent_loop 相比，新增两个关键步骤：
#   1. 循环开始前：加载相关记忆注入上下文（load_memories）
#   2. 循环结束后：从对话中提取新记忆（extract_memories）+ 合并整理（consolidate）

MAX_REACTIVE_RETRIES = 1  # 应急压缩的重试次数限制


def agent_loop(messages: list):
    """Agent 主循环：记忆注入 → 压缩 → LLM 调用 → 工具执行 → 记忆提取"""
    reactive_retries = 0

    # ── s09 新增：会话开始时加载相关记忆 ──
    # 根据最近对话内容，选择相关记忆文件并加载全文
    memories_content = load_memories(messages)
    # 记住最后一条 user 消息的位置，后续注入记忆内容到该位置
    memory_turn = len(messages) - 1 if messages and isinstance(messages[-1].get("content"), str) else None
    # 构建 system prompt（包含记忆索引摘要）
    system = build_system()

    while True:
        # ── s09 新增：保存压缩前的快照，用于事后提取记忆 ──
        # 压缩会丢失细节，所以用原始消息提取记忆更准确
        pre_compress = [m if isinstance(m, dict) else {"role": m.get("role", ""),
                                                       "content": str(m.get("content", ""))} for m in messages]

        # ── s08 压缩管道（budget → snip → micro）──
        messages[:] = tool_result_budget(messages)  # L3: 大结果落盘
        messages[:] = snip_compact(messages)        # L1: 裁掉中间
        messages[:] = micro_compact(messages)       # L2: 旧结果占位符替换

        # 如果预处理后仍然超限，触发 L4 全量摘要
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        try:
            # ── s09 新增：将记忆内容注入到最后一条 user 消息中 ──
            # 不修改原始 messages，而是创建 request_messages 副本
            request_messages = messages
            if memories_content and memory_turn is not None and memory_turn < len(messages):
                request_messages = messages.copy()
                request_messages[memory_turn] = {
                    **messages[memory_turn],
                    "content": memories_content + "\n\n" + messages[memory_turn]["content"],
                }
            # 调用 LLM
            response = client.messages.create(
                model=MODEL, system=system, messages=request_messages, tools=TOOLS, max_tokens=8000
            )
            reactive_retries = 0  # API 成功，重置重试计数
        except Exception as e:
            # ── 应急压缩（prompt_too_long 的最后防线）──
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(
                    e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                messages[:] = reactive_compact(messages)
                reactive_retries += 1
                continue  # 压缩后重试
            raise  # 非 token 错误或重试耗尽，向上抛出

        # ── 处理模型响应 ──
        messages.append({"role": "assistant", "content": response.content})

        # 模型停止调用工具 → 对话结束
        if response.stop_reason != "tool_use":
            # ── s09 新增：会话结束时提取新记忆 ──
            # 使用压缩前的快照提取（保留完整细节）
            extract_memories(pre_compress)
            # 如果记忆文件数超阈值，触发合并整理
            consolidate_memories()
            return

        # ── 执行工具调用 ──
        results = []
        for block in response.content:
            if block.type != "tool_use": continue
            # 青色打印工具名称
            print(f"\033[36m> {block.name}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:200])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


# ═══════════════════════════════════════════════════════════
#  入口：交互式 REPL
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("s09: Memory — persistent cross-session knowledge")
    print(f"Memory directory: {MEMORY_DIR}")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []
    while True:
        try:
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""): break
        # 将用户输入追加到对话历史
        history.append({"role": "user", "content": query})
        # 运行 Agent 循环（包含记忆注入 + 压缩 + 工具执行 + 记忆提取）
        agent_loop(history)
        # 打印模型的最后一条文本回复
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text": print(block.text)
        print()
