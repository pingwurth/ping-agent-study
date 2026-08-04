#!/usr/bin/env python3
"""
u03_permission.py - 权限控制系统
==================================
本文件演示 **权限控制（Permission System）** 机制：如何在工具执行前进行安全审查。

核心概念：
  1. 不是所有工具调用都应该自动执行
  2. 危险操作（如 rm -rf、写入系统文件）需要用户确认
  3. 权限系统在工具执行前拦截，询问用户是否允许
  4. Claude Code 支持三种权限模式：自动允许、需确认、禁止

Claude Code 中的权限实现：
  - 每个工具有权限级别（auto、confirm、deny）
  - 用户可以通过 "always allow" 跳过后续同类确认
  - 某些操作（如删除文件）始终需要确认

权限检查流程：
  模型请求工具调用 → 检查是否危险 → 危险则询问用户 → 允许则执行，拒绝则跳过

Three gates inserted before tool execution:

    Gate 1: Hard deny list (rm -rf /, sudo, ...)
    Gate 2: Rule matching (write outside workspace? destructive cmd?)
    Gate 3: User approval (pause and wait for confirmation)

    +-------+    +--------+    +--------+    +--------+    +------+
    | Tool  | -> | Gate 1 | -> | Gate 2 | -> | Gate 3 | -> | Exec |
    | call  |    | deny?  |    | match? |    | allow? |    |      |
    +-------+    +--------+    +--------+    +--------+    +------+
         |            |             |             |
         v            v             v             v
      (normal)     (blocked)    (ask user)   (user says no?)
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
#  Three-Gate Permission Pipeline
# ═══════════════════════════════════════════════════════════

# Gate 1: Hard deny list — always forbidden
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]


def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# Gate 2: Rule matching — context-dependent checks
PERMISSION_RULES = [
    {"tools": ["read_file", "write_file", "edit_file"],
     "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
     "message": "Writing outside workspace"},
    {"tools": ["bash"],
     "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
     "message": "Potentially destructive command"},
]


def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# Gate 3: User approval — wait for confirmation after rule match
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# Pipeline: all three gates chained
def check_permission(block) -> bool:
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False
    return True


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

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 在执行前运行权限管道
            if not check_permission(block):
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Permission denied."
                })
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"

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
            # 输入追加到消息历史
            history.append({"role": "user", "content": user_input})
        except (EOFError, KeyboardInterrupt):
            break  # Ctrl+D 退出, Ctrl+C 退出

        auto_approve = agent_loop(history, is_auto_approve=auto_approve)
        print()
        print("-" * 100)
        print_response_content(history[-1]['content'])
