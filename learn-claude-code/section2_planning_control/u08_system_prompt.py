"""
U08 - System Prompt（系统提示词）
=================================
本文件演示 **System Prompt** 机制：如何通过系统提示词定义 Agent 的行为。
使用原生 Anthropic SDK 实现。

核心概念：
  1. System Prompt 定义了 Agent 的角色、能力和行为规范
  2. 它在每轮对话中都会被发送给模型，但不会显示在对话历史中
  3. Claude Code 的 System Prompt 非常复杂，包含多个层次

Claude Code 的 System Prompt 结构：
  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code 的 system prompt 包含以下部分：              │
  │                                                          │
  │  1. 角色定义：你是谁，能做什么                            │
  │  2. 工具指南：如何使用各种工具                            │
  │  3. 行为规范：回复风格、安全约束                          │
  │  4. 环境信息：当前目录、平台、git 状态                    │
  │  5. 规则（Rules）：从 CLAUDE.md 加载的项目规则            │
  │  6. 技能（Skills）：当前激活的技能                        │
  │  7. MCP 服务器：可用的外部工具                            │
  │  8. 日期和上下文：当前日期、会话信息                      │
  │                                                          │
  │  这些部分在每次 API 调用时动态组装。                      │
  └──────────────────────────────────────────────────────────┘

为什么 System Prompt 如此重要？
  - 它是 Agent 行为的"宪法"，定义了所有行为边界
  - 好的 system prompt 能让 Agent 更准确、更安全地工作
  - Claude Code 的 system prompt 经过精心设计和反复迭代
"""

import os, json
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ── readline 中文修复 ──────────────────────────────────────
# macOS 默认使用 libedit 而非 GNU readline，在处理中文输入时
# 退格键（Backspace）会出错。以下配置可修复该问题。
try:
    import readline

    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    # Windows 或精简环境可能没有 readline，忽略即可
    pass

# ============================================================
# 初始化客户端
# ============================================================
WORKDIR = Path.cwd()
client, MODEL = create_client()
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

# ═══════════════════════════════════════════════════════════
#  Prompt Sections
# ═══════════════════════════════════════════════════════════
PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
}


def assemble_system_prompt(context: dict) -> str:
    """Select and join prompt sections based on current context."""
    sections = []

    # Always loaded — identity
    sections.append(PROMPT_SECTIONS["identity"])

    # Dynamic — tools and workspace from context
    tools = ", ".join(context.get("enabled_tools", []))
    if tools:
        sections.append(f"Available tools: {tools}.")
    sections.append(f"Working directory: {context.get("workspace", WORKDIR)}")

    # Conditional — memory loaded when MEMORY.md exists and has content
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")

    return "\n\n".join(sections)


_last_context_key = None
_last_prompt = None


def get_system_prompt(context: dict) -> str:
    """Cache wrapper — reassemble only when context changes.

    Uses json.dumps for deterministic serialization, not Python's hash()
    which has process randomization and fails on nested dicts/lists.
    This cache only avoids redundant string assembly within a process.
    Real Claude Code additionally protects API-level prompt cache via
    stable section ordering and SYSTEM_PROMPT_DYNAMIC_BOUNDARY.
    """
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)

    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt


# ═══════════════════════════════════════════════════════════
#   Tool Definitions & Dispatch
# ═══════════════════════════════════════════════════════════
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string"
                }
            },
            "required": [
                "command"
            ]
        }
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "limit": {
                    "type": "integer"
                }
            },
            "required": [
                "path"
            ]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "content": {
                    "type": "string"
                }
            },
            "required": [
                "path",
                "content"
            ]
        }
    }
]

TOOL_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write}


# ═══════════════════════════════════════════════════════════
#  Context
# ═══════════════════════════════════════════════════════════
def update_context(context: dict, messages: list) -> dict:
    """Derive context from real state: which tools exist, whether memory files exist."""
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


def print_response_content(content):
    """
    打印模型回复的内容块。

    Anthropic API 返回的 content 是一个列表，每个元素是一个 content block：
      - type="text"：普通文本，模型的自然语言回复
    """
    if isinstance(content, list):
        for block in content:
            if block.type == "text":
                print(f"\n{block.text}")


def agent_loop(messages: list) -> None:
    """
    Agent 循环的核心实现。

    流程：
      1. 将消息历史发送给模型（连同工具定义）
      2. 模型返回回复，检查 stop_reason
      3. 如果是 "end_turn" -- 对话结束，退出循环
      4. 如果包含 tool_use 块 -- 执行工具，将结果追加到消息中，继续循环
      5. 重复步骤 1

    关键 API 字段说明：
      - response.stop_reason: "end_turn"（模型说完）或 "tool_use"（请求执行工具）
      - response.content: 内容块列表（text 和/或 tool_use）
      - tool_use.id: 每个工具调用的唯一 ID，tool_result 必须用这个 ID 来匹配
    """
    system = get_system_prompt(context)
    while True:
        # 调用模型 API
        # tools 参数传入工具定义列表，模型就知道它有哪些能力可用
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        # 将助手回复追加到消息历史
        # 注意：content 是 block 列表，不是字符串
        messages.append({"role": "assistant", "content": response.content})

        # 关键判断：模型是否要结束对话？
        if response.stop_reason == "end_turn":
            break

        # 如果 stop_reason 包含 tool_use，处理工具调用
        # 一个回复中可能包含多个 tool_use 块，需要逐个处理
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n[执行工具] {block.name}...")

                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"

                print(f"[工具输出]\n{output}")

                # 构建 tool_result
                # tool_use_id 必须与 tool_use 的 id 匹配，这是 API 的要求
                # content 是工具执行的输出文本
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # 将工具结果作为 user 消息追加到历史
        # 下一轮循环时，模型会看到这些工具结果并决定下一步
        messages.append({"role": "user", "content": tool_results})
        # 更新上下文，重新组装系统提示词
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    """
    条件判断：if __name__ == "__main__": 检查当前模块是否作为主程序运行
    程序入口：只有当文件直接执行时，条件才为真，代码块内的内容才会执行
    模块导入保护：当文件被其他模块导入时，条件为假，代码块内的内容不会执行

    __name__ 是Python的一个内置变量：
    当文件直接运行时，__name__ 的值为 "__main__"
    当文件被导入时，__name__ 的值为模块名（如 "u01_agent_loop"）
    """

    print("输入问题，回车发送。输入 q 退出。")

    # 消息历史贯穿整个会话，模型能看到之前所有对话
    history = []
    context = update_context({}, [])
    while True:
        try:
            # 用户输入
            user_input = input("\033[36ms01 >> \033[0m").strip()
            if user_input.lower() in ("q", "exit", "quit", ""): break
            # 输入追加到消息历史
            history.append({"role": "user", "content": user_input})
        except (EOFError, KeyboardInterrupt):
            break  # Ctrl+D 退出, Ctrl+C 退出

        # Agent Loop = 智能体的 “思考 — 行动 — 观察” 循环
        # 是大模型 Agent（智能体）最核心的工作机制
        # 让 AI 能像人一样自主解决复杂任务，而不是只做一次性问答
        agent_loop(history, context)

        context = update_context(context, history)
        print()
        print("-" * 100)
        print_response_content(history[-1]['content'])
