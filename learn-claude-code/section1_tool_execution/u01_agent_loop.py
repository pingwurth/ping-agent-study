#!/usr/bin/env python3
"""
u01_agent_loop.py - 最小化 Agent 循环实现
===========================================
本文件演示 Agent 最核心的运行机制：**消息循环（Agent Loop）**。
直接使用 Anthropic SDK 实现，不依赖任何框架。

核心概念：
  1. 工具（Tool）：以 JSON Schema 形式定义，告诉模型可以调用什么能力
  2. 消息循环（Agent Loop）：模型生成回复 -> 检测工具调用 -> 执行工具 -> 将结果送回模型 -> 重复
  3. stop_reason：模型返回 "end_turn" 表示对话结束，"tool_use" 表示需要执行工具

Claude Code 的本质就是一个 while True 循环：
  ┌──────────────────────────────────────────────────┐
  │  ① 将消息历史 + 工具定义发送给模型                │
  │  ② 模型返回回复                                  │
  │  ③ 如果 stop_reason == "end_turn" → 退出循环     │
  │  ④ 如果包含 tool_use 块 → 执行工具               │
  │  ⑤ 将 tool_result 追加到消息历史                  │
  │  ⑥ 回到步骤 ①                                    │
  └──────────────────────────────────────────────────┘

.env 配置项：
  - ANTHROPIC_BASE_URL: API 代理地址（可选，用于自定义端点）
  - MODEL_ID: 模型 ID（必需）
"""

import os
import sys
import subprocess

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
client, MODEL = create_client()

# ============================================================
# System Prompt（系统提示词）
# ============================================================
# 告诉模型它的角色：一个在当前目录工作的编码 Agent
# "Act, don't explain" 让模型直接行动而非解释意图
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ============================================================
# 工具定义：以 JSON Schema 形式描述工具的能力
# ============================================================
# 这是 Anthropic Tool Use 的标准格式：
#   - name: 工具名称，模型会用这个名字来调用工具
#   - description: 工具描述，帮助模型理解何时使用该工具
#   - input_schema: JSON Schema 格式的参数定义
#
# Claude Code 中定义了类似但更多的工具：
#   Read、Write、Edit、Grep、Bash 等
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

# 危险命令黑名单 —— 防止模型执行破坏性操作
# 实际的 Claude Code 有更完善的沙箱和权限系统
DANGEROUS_COMMANDS = [
    "rm -rf /",
    "rm -rf ~",
    "sudo",
    "shutdown",
    "reboot",
    "> /dev/"
    "mkfs",
    "dd if=",
    "chmod -R 777 /",
]


def is_dangerous(command: str) -> bool:
    """检查命令是否在危险命令黑名单中。"""
    cmd_lower = command.lower().strip()
    return any(danger in cmd_lower for danger in DANGEROUS_COMMANDS)


def execute_bash(command: str) -> str:
    """
    执行 bash 命令并捕获输出。

    实际的 Claude Code 会在沙箱中执行命令，这里为了教学简化为直接 subprocess.run。
    超时设置为 120 秒，防止命令挂起。
    """
    if is_dangerous(command):
        return f"[BLOCKED] 危险命令被拦截: {command}"

    try:
        result = subprocess.run(
            command,              # 要执行的命令字符串
            shell=True,           # 通过 shell 执行，支持管道、重定向等语法
            cwd=os.getcwd(),      # 工作目录设为当前目录（与 Agent 运行目录一致）
            capture_output=True,  # 捕获 stdout 和 stderr，而非直接打印到终端
            text=True,            # 以字符串（而非 bytes）返回输出，自动解码
            timeout=120,          # 超时 120 秒，防止命令挂起导致 Agent 卡死
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
    while True:
        # 调用模型 API
        # tools 参数传入工具定义列表，模型就知道它有哪些能力可用
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            system=SYSTEM,
            tools=[BASH_TOOL],
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

                # 根据工具名称分发执行
                if block.name == "bash":
                    output = execute_bash(block.input["command"])
                else:
                    output = f"[ERROR] 未知工具: {block.name}"

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
            if user_input.lower() in ("q" "exit", "quit", ""): break
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
