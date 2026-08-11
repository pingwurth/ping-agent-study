"""
U05 - TodoWrite（任务管理工具）
================================
本文件演示 **TodoWrite** 工具：如何在 Agent 运行中管理任务列表。
使用原生 Anthropic SDK 实现。

核心概念：
  1. TodoWrite 让 Agent 能够将复杂任务分解为可跟踪的子任务
  2. 每个任务有三种状态：pending（待处理）、in_progress（进行中）、completed（完成）
  3. Agent 可以动态添加、更新、删除任务
  4. 任务列表帮助 Agent 保持方向感，避免在长对话中迷失

实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  TodoWrite 工具的 JSON Schema 定义：                      │
  │                                                          │
  │  {                                                       │
  │    "name": "TodoWrite",                                  │
  │    "input_schema": {                                     │
  │      "type": "object",                                   │
  │      "properties": {                                     │
  │        "todos": {                                        │
  │          "type": "array",                                │
  │          "items": { ... }                                │
  │        }                                                 │
  │      }                                                   │
  │    }                                                     │
  │  }                                                       │
  │                                                          │
  │  Agent 调用此工具 → 后端更新任务列表 → 返回进度摘要            │
  └──────────────────────────────────────────────────────────┘

为什么需要 TodoWrite？
  - 长对话中 Agent 容易"忘记"自己在做什么
  - 任务列表提供结构化的进度追踪
  - 用户可以看到 Agent 的工作进展
  - Agent 可以根据进度动态调整策略

  +---------+      +-------+      +------------------+
  |  User   | ---> |  LLM  | ---> | TOOL_HANDLERS    |
  | prompt  |      |       |      |  bash            |
  +---------+      +---+---+      |  read_file       |
                        ^         |  write_file      |
                        | result  |  edit_file       |
                        +---------+  glob            |
                                      todo_write ← NEW
                                   +------------------+
                                        |
                         in-memory current_todos
                                        |
                        if rounds_since_todo >= 3:
                          inject <reminder>
"""
import ast
import json
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
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)
CURRENT_TODOS: list[dict] = []


# ═══════════════════════════════════════════════════════════
#  Tool Implementations
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


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
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
    },
    {
        "name": "todo_write",
        "description": "Create and manage a task list for your current coding session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string"
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed"
                                ]
                            }
                        },
                        "required": [
                            "content",
                            "status"
                        ]
                    }
                }
            },
            "required": [
                "todos"
            ]
        }
    }
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
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
    带 Hook 系统和 Todo 提醒的 Agent 循环。

    在 u04 Hook 系统的基础上新增 todo_write 工具的调度逻辑：

      1. 每次模型发起工具调用，rounds_since_todo 自增 1
      2. 若连续 3 轮未调用 todo_write → 注入 <reminder> 提醒 LLM 更新任务列表
      3. 调用 todo_write 后 rounds_since_todo 重置为 0

    目的：长对话中防止 Agent "迷路"，通过周期性提醒保持任务列表的时效性。

    Args:
        messages: 对话消息列表（会被原地追加 assistant/tool_result 消息）
        is_auto_approve: 是否自动允许所有工具调用（默认 False）

    Returns:
        is_auto_approve 的当前值，供调用方决定后续交互是否继续跳过确认
    """
    rounds_since_todo = 0
    while True:
        # 超过 3 轮提示 LLM 更新 todo list
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

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

        rounds_since_todo += 1

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

            if block.name == "todo_write":
                rounds_since_todo = 0
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
