#!/usr/bin/env python3
"""
u02_tool_use.py - 多工具调用实现
==================================
本文件演示 **多工具（Multi-Tool）** 支持：如何同时定义多个工具，
让模型根据任务自主选择最合适的工具。

核心概念：
  1. 工具定义：每个工具以 JSON Schema 描述名称、用途和参数
  2. 工具选择：模型根据任务需求自主决定调用哪个工具
  3. 工具分发：根据工具名将调用路由到对应的执行函数
  4. 多工具协作：一个任务可能需要多个工具配合完成

Claude Code 中定义的工具示例：
  - Read: 读取文件内容
  - Write: 写入文件
  - Edit: 编辑文件（精确替换）
  - Grep: 搜索文件内容
  - Bash: 执行 shell 命令

工具调用的完整流程：
  模型输出 tool_use block → 解析工具名和参数 → 执行对应函数 → 返回结果
"""

import os
import sys
import json
import subprocess

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

# ============================================================
# 初始化客户端
# ============================================================
client, MODEL = create_client()

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use tools to solve tasks. Act, don't explain."

# ============================================================
# 工具定义：四个工具，各自有不同的用途
# ============================================================
# 每个工具的 description 至关重要 —— 模型根据描述判断何时使用哪个工具
# 好的描述 = 模型能正确选择工具

BASH_TOOL = {
    "name": "bash",
    "description": "在终端中执行 bash 命令并返回输出结果。",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 bash 命令",
            }
        },
        "required": ["command"],
    },
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "读取指定路径的文件内容并返回。",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文件路径（绝对路径或相对路径）",
            }
        },
        "required": ["path"],
    },
}

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": "将内容写入指定路径的文件。如果文件不存在则创建，存在则覆盖。",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要写入的文件路径",
            },
            "content": {
                "type": "string",
                "description": "要写入的内容",
            }
        },
        "required": ["path", "content"],
    },
}

EDIT_FILE_TOOL = {
    "name": "edit_file",
    "description": "仅替换文件中的精确文本一次。",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要写入的文件路径",
            },
            "old_text": {
                "type": "string",
                "description": "要替换的文本",
            },
            "new_text": {
                "type": "string",
                "description": "要替换的新文本",
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
}

# 所有工具列表 —— 传给模型让它知道有哪些工具可用
ALL_TOOLS = [BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, EDIT_FILE_TOOL]


# ============================================================
# 工具执行函数
# ============================================================

def execute_bash(command: str) -> str:
    """执行 bash 命令。"""
    dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){:|:&};:"]
    if any(d in command.lower() for d in dangerous):
        return f"[BLOCKED] 危险命令被拦截: {command}"
    try:
        result = subprocess.run(
            command, shell=True,
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(命令执行成功，无输出)"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT] 命令执行超过 120 秒超时"
    except Exception as e:
        return f"[ERROR] 执行失败: {e}"


def execute_read_file(path: str) -> str:
    """读取文件内容。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[ERROR] 文件不存在: {path}"
    except Exception as e:
        return f"[ERROR] 读取失败: {e}"


def execute_write_file(path: str, content: str) -> str:
    """写入文件内容。"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入 {path}"
    except Exception as e:
        return f"[ERROR] 写入失败: {e}"


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    工具分发器：根据工具名调用对应的执行函数。

    这是 Agent 的"手臂" —— 模型决定做什么（tool_use），
    这个函数负责实际执行。
    """
    if tool_name == "bash":
        return execute_bash(tool_input["command"])
    elif tool_name == "read_file":
        return execute_read_file(tool_input["path"])
    elif tool_name == "write_file":
        return execute_write_file(tool_input["path"], tool_input["content"])
    elif tool_name == "edit_file":
        return execute_write_file(tool_input["path"],
                                  tool_input["old_text"].replace(tool_input["old_text"], tool_input["new_text"]))
    else:
        return f"[ERROR] 未知工具: {tool_name}"


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
# Agent 循环（多工具版本）
# ============================================================
def agent_loop(messages: list) -> None:
    """
    带多工具支持的 Agent 循环。

    与 u01 的区别：
      - u01 只有 bash 一个工具，硬编码执行
      - 这里有三个工具，通过 execute_tool() 统一分发
      - 模型会根据任务自动选择最合适的工具

    多工具协作示例：
      用户: "读取 config.json，修改版本号为 2.0，然后写回"
      模型:
        ① 调用 read_file("config.json") → 获取内容
        ② 调用 write_file("config.json", 新内容) → 写回修改
        ③ 回复 "已完成修改"
    """
    while True:
        # 将所有工具定义传给模型
        response = client.messages.create(
            model=MODEL, max_tokens=8096, system=SYSTEM,
            tools=ALL_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        # 处理所有工具调用
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n[执行工具] {block.name}...")
                # 使用统一分发器，而非硬编码 if/else
                output = execute_tool(block.name, block.input)
                print(f"[工具输出]\n{output}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user", "content": tool_results})


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
    messages = []

    while True:
        try:
            # 用户输入
            user_input = input("\033[36ms01 >> \033[0m").strip()
            if user_input.lower() in ("q", "exit", "quit", ""): break
            # 输入追加到消息历史
            messages.append({"role": "user", "content": user_input})
        except (EOFError, KeyboardInterrupt):
            break  # Ctrl+D 退出, Ctrl+C 退出

        # Agent Loop = 智能体的 “思考 — 行动 — 观察” 循环
        # 是大模型 Agent（智能体）最核心的工作机制
        # 让 AI 能像人一样自主解决复杂任务，而不是只做一次性问答
        agent_loop(messages)

        print()
        print("-" * 100)
        print_response_content(messages[-1]['content'])
