"""
U09 - Error Recovery（错误恢复）
=================================
本文件演示 **错误恢复** 机制：Agent 如何处理和从错误中恢复。
使用原生 Anthropic SDK 实现。

核心概念：
  1. Agent 在执行过程中会遇到各种错误
  2. 错误恢复能力是 Agent 智能的重要体现
  3. 不同类型的错误需要不同的恢复策略
  4. 最自然的恢复方式是将错误信息反馈给模型

错误恢复的工作原理：
  ┌──────────────────────────────────────────────────────────┐
  │  当工具执行失败时：                                       │
  │                                                          │
  │  1. 不要崩溃或抛出异常                                    │
  │  2. 将错误信息作为工具结果返回给模型                       │
  │  3. 模型看到错误后，会自动调整策略                         │
  │  4. 模型可能尝试不同的方法，或向用户求助                   │
  │                                                          │
  │  这种"将错误反馈给模型"的方式是最自然的错误恢复机制：      │
  │  模型本身就是错误处理的决策者。                            │
  └──────────────────────────────────────────────────────────┘

Claude Code 的错误恢复策略：
  - 工具错误 → 将错误信息反馈给模型，让它调整命令
  - 权限错误 → 提示用户授予权限
  - 文件不存在 → 搜索相似文件名
  - 速率限制 → 等待后重试
  - 上下文溢出 → 压缩上下文或重启会话
  - 多次失败 → 向用户求助
"""

import os, json
import random
import sys
import subprocess
import time
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

PRIMARY_MODEL = os.environ["MODEL_ID"]
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")

# ── Constants ──

ESCALATED_MAX_TOKENS = 64000
DEFAULT_MAX_TOKENS = 8000
MAX_RECOVERY_RETRIES = 3
MAX_RETRIES = 10
BASE_DELAY_MS = 500
MAX_CONSECUTIVE_529 = 3
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)

# ═══════════════════════════════════════════════════════════
#  Prompt Assembly
# ═══════════════════════════════════════════════════════════
PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)


_last_context_key, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    """带缓存的 system prompt 获取函数。

    核心思想：避免在 context 未变化时重复组装 system prompt 字符串。
    使用 json.dumps 将 context 序列化为确定性字符串作为缓存 key，
    而非 Python 内置 hash()——因为 hash() 存在进程级随机化（PYTHONHASHSEED），
    且无法直接处理嵌套的 dict/list。

    缓存策略说明：
    - 这是进程内缓存，仅避免同一次运行中的重复字符串拼接。
    - 真实的 Claude Code 还通过稳定 section 顺序 + SYSTEM_PROMPT_DYNAMIC_BOUNDARY
      来保护 API 层面的 prompt cache（跨请求复用前缀 token），
      这里不做模拟。

    Args:
        context: 上下文字典，包含 memories 等可选字段。

    Returns:
        组装好的 system prompt 字符串。
    """
    global _last_context_key, _last_prompt
    # 将 context 确定性序列化为字符串，作为缓存比对的 key
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    # 缓存命中：context 未变化且已有缓存，直接返回上次结果
    if key == _last_context_key and _last_prompt:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt
    # 缓存未命中：更新 key 并重新组装 prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)

    # 打印本次组装加载了哪些 section，便于调试
    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt


# ═══════════════════════════════════════════════════════════
# 工具定义：以 JSON Schema 形式描述工具的能力
# ═══════════════════════════════════════════════════════════
# 这是 Anthropic Tool Use 的标准格式：
#   - name: 工具名称，模型会用这个名字来调用工具
#   - description: 工具描述，帮助模型理解何时使用该工具
#   - input_schema: JSON Schema 格式的参数定义
#
# Claude Code 中定义了类似但更多的工具：
#   Read、Write、Edit、Grep、Bash 等
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


TOOL_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write}


# ═══════════════════════════════════════════════════════════
#  Error Recovery (s11 new)
# ═══════════════════════════════════════════════════════════
class RecoveryState:
    """Track recovery attempts across the loop."""

    def __init__(self):
        self.has_escalated = False
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.has_attempted_reactive_compact = False
        self.current_model = PRIMARY_MODEL


def retry_delay(attempt, retry_after=None):
    """Exponential backoff with jitter. Retry-After takes priority."""
    if retry_after:
        return retry_after
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000
    jitter = random.uniform(0, base * 0.25)
    return base + jitter


def with_retry(fn, state: RecoveryState):
    """Exponential backoff for transient errors (429/529).
    Non-transient errors are re-raised for the outer handler."""
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0
            return result
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()

            # 429 rate limit -> exponential backoff
            if "ratelimit" in name.lower() or "429" in msg:
                delay = retry_delay(attempt)
                print(f"  \033[33m[429 rate limit] retry {attempt + 1}/{MAX_RETRIES},"
                      f" wait {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue

            # 529 overloaded -> exponential backoff + fallback model
            if "overloaded" in name.lower() or "529" in msg or "overloaded" in msg:
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529:
                    if FALLBACK_MODEL:
                        state.current_model = FALLBACK_MODEL
                        state.consecutive_529 = 0
                        print(f"  \033[31m[529 x{MAX_CONSECUTIVE_529}]"
                              f" switching to {FALLBACK_MODEL}\033[0m")
                    else:
                        state.consecutive_529 = 0
                        print(f"  \033[31m[529 x{MAX_CONSECUTIVE_529}]"
                              f" no FALLBACK_MODEL_ID configured, continuing retry\033[0m")
                delay = retry_delay(attempt)
                print(f"  \033[33m[529 overloaded] retry {attempt + 1}/{MAX_RETRIES},"
                      f" wait {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue

            # Not transient -> re-raise for outer try/except
            raise
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")


def is_prompt_too_long_error(e: Exception) -> bool:
    """Check whether an API error indicates prompt/context too long."""
    msg = str(e).lower()
    return (("prompt" in msg and "long" in msg)
            or "prompt_is_too_long" in msg
            or "context_length_exceeded" in msg
            or "max_context_window" in msg)


def reactive_compact(messages: list) -> list:
    """Emergency compact — teaching version keeps last N messages.
    Real CC generates a compact summary via LLM, then retries with
    the compacted message list. Teaching version simplifies to tail
    retention since s08/s09 already cover LLM-based compact."""
    print("  \033[31m[reactive compact] trimming to last 5 messages\033[0m")
    tail = messages[-5:]
    return [{"role": "user",
             "content": "[Reactive compact] Earlier conversation trimmed. "
                        "Continue from where you left off."}, *tail]


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


def agent_loop(messages: list, context: dict):
    """包含错误恢复机制的主循环，用于包裹大语言模型的调用。

    整体流程：
    ┌──────────────────────────────────────────────────────┐
    │  while True:                                         │
    │    ① 调用 LLM（with_retry 处理 429/529 速率限制）   │
    │    ② 处理调用层面的异常（prompt_too_long 等）        │
    │    ③ 处理 max_tokens 截断 → 提升上限 / 续写         │
    │    ④ 正常完成 → 返回                                 │
    │    ⑤ stop_reason == tool_use → 执行工具 → 追加结果   │
    └──────────────────────────────────────────────────────┘

    Args:
        messages: 对话历史，会被原地修改（追加 assistant / user 消息）
        context: 上下文字典，包含 memories 等字段，每轮工具执行后更新
    """
    system = get_system_prompt(context)
    state = RecoveryState()  # 追踪恢复状态（escalation、compact 尝试、529 连续次数等）
    max_tokens = DEFAULT_MAX_TOKENS  # 初始输出上限 8000 token

    while True:
        # ── ① 调用 LLM ──
        # with_retry 负责 429（速率限制）和 529（服务过载）的指数退避重试；
        # 其余异常会直接抛出，由外层 try/except 捕获处理。
        try:
            response = with_retry(
                lambda: client.messages.create(
                    model=state.current_model, system=system,
                    messages=messages, tools=TOOLS,
                    max_tokens=max_tokens),
                state)
        except Exception as e:
            # ── ② 调用层面的错误处理 ──

            # Path 2: prompt_too_long —— 上下文超过模型窗口限制
            # 策略：用 reactive_compact 压缩消息历史（仅尝试一次）
            if is_prompt_too_long_error(e):
                if not state.has_attempted_reactive_compact:
                    messages[:] = reactive_compact(messages)  # 原地替换消息列表
                    state.has_attempted_reactive_compact = True
                    continue  # 压缩后重试调用
                # 压缩后仍然太长 → 不可恢复，向用户报告错误
                print("  \033[31m[unrecoverable] still too long after compact\033[0m")
                messages.append({"role": "assistant", "content": [
                    {"type": "text",
                     "text": "[Error] Context too large, cannot continue."}]})
                return

            # 其他不可恢复的异常（网络中断、认证失败等）→ 直接终止
            name = type(e).__name__
            print(f"  \033[31m[unrecoverable] {name}: {str(e)[:100]}\033[0m")
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": f"[Error] {name}: {str(e)[:200]}"}]})
            return

        # ── ③ 输出被 max_tokens 截断的处理 ──
        if response.stop_reason == "max_tokens":
            # 第一次截断：不保存截断的输出，直接提升上限到 64K 重试
            # 这样做是因为提升上限后模型会重新生成完整输出，避免拼接碎片
            if not state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS  # 8000 → 64000
                state.has_escalated = True
                print(f"  \033[33m[max_tokens] escalating"
                      f" {DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}\033[0m")
                continue
            # 64K 仍然截断：保存截断输出 + 追加续写提示，让模型继续
            # 这是"continuation"模式——模型看到之前的内容后接着写
            messages.append({"role": "assistant", "content": response.content})
            if state.recovery_count < MAX_RECOVERY_RETRIES:  # 最多重试 3 次
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                print(f"  \033[33m[max_tokens] continuation"
                      f" {state.recovery_count}/{MAX_RECOVERY_RETRIES}\033[0m")
                continue
            # 超过续写次数上限 → 放弃
            print("  \033[31m[max_tokens] recovery limit reached\033[0m")
            return

        # ── ④ 正常完成：追加 assistant 响应 ──
        messages.append({"role": "assistant", "content": response.content})

        # 如果 stop_reason 不是 tool_use，说明模型已完成回复（end_turn）→ 退出循环
        if response.stop_reason != "tool_use":
            return

        # ── ⑤ 工具执行阶段 ──
        # 遍历响应中的所有 tool_use 块，逐个调用对应的 handler
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            # handler 未找到时返回错误提示而非崩溃——这就是"错误反馈给模型"策略
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:200])
            results.append({"type": "tool_result",
                            "tool_use_id": block.id, "content": output})
        # 工具结果作为 user 消息追加（Anthropic API 要求 tool_result 放在 user role 中）
        messages.append({"role": "user", "content": results})

        # 工具执行后更新上下文（可能注入新的 memories），重新组装 system prompt
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
        except (EOFError, KeyboardInterrupt):
            break  # Ctrl+D 退出, Ctrl+C 退出

        turn_start = len(history)
        # 输入追加到消息历史
        history.append({"role": "user", "content": user_input})

        # Agent Loop = 智能体的 “思考 — 行动 — 观察” 循环
        # 是大模型 Agent（智能体）最核心的工作机制
        # 让 AI 能像人一样自主解决复杂任务，而不是只做一次性问答
        agent_loop(history, context)

        context = update_context(context, history)
        print()
        print("-" * 100)
        for msg in history[turn_start:]:
            if msg.get("role") != "assistant":
                continue
            for block in msg["content"]:
                if getattr(block, "type", None) == "text":
                    print(block.text)
