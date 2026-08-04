#!/usr/bin/env python3
"""
u04_hooks.py - Hooks（钩子系统）实现
======================================
本文件演示 **Hooks（钩子系统）** 机制：在工具执行的特定时机自动运行代码。

核心概念：
  1. Hooks 是在工具执行生命周期中自动触发的脚本
  2. 三种类型：
     - PreToolUse: 工具执行前触发（可用于安全检查、参数验证）
     - PostToolUse: 工具执行后触发（可用于格式化、日志记录）
     - Stop: 会话结束时触发（可用于清理、最终验证）
  3. PreToolUse 的非零退出码会阻止工具执行（关键安全机制）
  4. Hooks 通过环境变量接收上下文信息（HOOK_TYPE, TOOL_NAME, FILE_PATH）

Hooks — move extension logic out of the loop, onto hooks.

  User types query
       │
       ▼
  ┌──────────────────┐
  │ UserPromptSubmit │ ── trigger_hooks() before LLM
  └────────┬─────────┘
           ▼
  ┌────────────┐     ┌────────────────────────────┐
  │  messages  │────▶│ LLM (stop_reason=tool_use?)│
  └────────────┘     │  No ──▶ Stop hooks ──▶ exit│
                     │  Yes ──▶ tool_use block ───│─┐
                     └────────────────────────────┘ │
                                                    ▼
                                          ┌──────────────────┐
                                          │ trigger_hooks()  │
                                          │  PreToolUse:     │
                                          │   permission_hook│
                                          │   log_hook       │
                                          └───────┬──────────┘
                                                  │ (not blocked)
                                          ┌───────▼──────────┐
                                          │ TOOL_HANDLERS[x] │
                                          └───────┬──────────┘
                                                  │
                                          ┌───────▼──────────┐
                                          │ trigger_hooks()  │
                                          │  PostToolUse:    │
                                          │   large_output   │
                                          └───────┬──────────┘
                                                  │
                                          results ──▶ back to messages

"""

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ── readline 中文修复 ──────────────────────────────────────
try:
    import readline

    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════
# 初始化客户端
# ═══════════════════════════════════════════════════════════
client, MODEL = create_client()
WORKDIR = Path.cwd()
SYSTEM = f"You are a coding agent at {WORKDIR}. All destructive operations require user approval."


# ═══════════════════════════════════════════════════════════
#  Tool Implementations
# ═══════════════════════════════════════════════════════════

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
        lines = (WORKDIR / path).resolve().read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
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


# ═══════════════════════════════════════════════════════════
#   Tool Definitions & Dispatch
# ═══════════════════════════════════════════════════════════

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
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "old_text": {
                    "type": "string"
                },
                "new_text": {
                    "type": "string"
                }
            },
            "required": [
                "path",
                "old_text",
                "new_text"
            ]
        }
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string"
                }
            },
            "required": [
                "pattern"
            ]
        }
    }
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

# ═══════════════════════════════════════════════════════════
#  Hook System
# ═══════════════════════════════════════════════════════════
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


# s03 permission check logic, now wrapped as a hook
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def permission_hook(block):
    """PreToolUse: s03 check_permission() logic moved here."""
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                print(f"\n\033[31m⛔ Blocked: '{pattern}'\033[0m")
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command", ""):
                print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
                print(f"   Tool: {block.name}({block.input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m⚠  Access outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None


def log_hook(block):
    """PreToolUse: log every tool call."""
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None


def large_output_hook(block, output):
    """PostToolUse: warn on large output."""
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] ⚠ Large output from {block.name}: {len(str(output))} chars\033[0m")
    return None


# UserPromptSubmit hook: log user input before it reaches the LLM
def context_inject_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


# Stop hook: print summary when loop is about to exit
def summary_hook(messages: list):
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# ═══════════════════════════════════════════════════════════
#  agent_loop — with check_permission() inserted
# ═══════════════════════════════════════════════════════════
def agent_loop(messages: list, is_auto_approve: bool = False) -> bool:
    """
    带权限控制的 Agent 循环。

    与 u01 的区别：
      - u01 直接执行所有工具调用
      - 这里在执行前进行权限检查
      - 危险命令会询问用户确认
      - 用户可以选择自动允许（auto_approve）

    auto_approve 参数：
      - False: 每次危险命令都询问（默认）
      - True: 跳过所有权限检查（用户之前选择了 'a'）
    """
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=8096, system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return is_auto_approve

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 工具执行前 HOOK: 权限检查放在这里实现
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(blocked)})
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"

            # 工具执行后 HOOK
            trigger_hooks("PostToolUse", block, output)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        messages.append({"role": "user", "content": tool_results})

    return is_auto_approve


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


# ============================================================
# 程序入口：交互式 REPL
# ============================================================
if __name__ == "__main__":
    print("输入问题，回车发送。输入 q 退出。")

    history = []
    auto_approve = False
    while True:
        try:
            # 用户输入
            user_input = input("\033[36ms01 >> \033[0m").strip()
            if user_input.lower() in ("q", "exit", "quit", ""): break
        except (EOFError, KeyboardInterrupt):
            break  # Ctrl+D 退出, Ctrl+C 退出

        # 触发 HOOK: 用户输入提交后、进入 LLM 前
        trigger_hooks("UserPromptSubmit", user_input)
        # 输入追加到消息历史
        history.append({"role": "user", "content": user_input})

        auto_approve = agent_loop(history, is_auto_approve=auto_approve)
        print()
        print("-" * 100)
        print_response_content(history[-1]['content'])
