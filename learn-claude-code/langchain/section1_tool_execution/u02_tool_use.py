"""
U02 - Tool Use（工具调用）
==========================
本文件演示 LangChain 的 **Tool Use** 机制：如何定义工具、模型如何自主选择工具。

核心概念：
  1. 工具定义：使用 @tool 装饰器，自动从函数签名生成 JSON Schema
  2. 工具调用：模型根据任务需要，自主决定调用哪个工具
  3. 工具执行：LangChain 自动执行工具，将结果返回给模型
  4. 多工具：可以同时定义多个工具，模型会选择最合适的

LangChain 的工具定义 vs 手动 JSON Schema：
  - 手动方式（Anthropic SDK）：手写 JSON Schema 字典
  - LangChain 方式（@tool 装饰器）：从函数签名自动生成
  - 装饰器自动从 docstring 和类型注解生成 JSON Schema

Claude Code 中定义的工具示例：
  - Read: 读取文件内容
  - Write: 写入文件
  - Edit: 编辑文件（精确替换）
  - Grep: 搜索文件内容
  - Bash: 执行 shell 命令
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

model = get_model()


# ── 定义多个工具 ──────────────────────────────────────────
# 使用 @tool 装饰器定义工具
# 每个工具的：
#   - 名称 = 函数名（或在装饰器中指定）
#   - 描述 = docstring 的第一行
#   - 参数 = 函数签名 + 类型注解
#
# 模型会根据 docstring 判断何时使用哪个工具

@tool
def bash(command: str) -> str:
    """Run a shell command and return its output.

    Args:
        command: The shell command to execute
    """
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )
    return result.stdout + result.stderr


@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path.

    Args:
        path: Absolute or relative file path
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist.

    Args:
        path: File path to write to
        content: Content to write into the file
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


# ── Agent Loop（多工具版本）───────────────────────────────
# 使用 LangGraph 的 create_react_agent，自动处理多工具选择
from langgraph.prebuilt import create_react_agent

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use tools to solve tasks. Act, don't explain."


def agent_loop(query: str) -> str:
    """
    带多工具支持的 Agent。

    与 u01 的区别：
      - u01 只有 bash 一个工具
      - 这里有 bash、read_file、write_file 三个工具
      - 模型会根据任务自动选择最合适的工具

    LangGraph 内部流程：
      ① 模型收到用户消息 + 3 个工具的 schema
      ② 模型决定调用哪个工具（可能同时调用多个）
      ③ LangGraph 自动执行工具，将结果返回模型
      ④ 模型根据结果决定是否继续调用工具
      ⑤ 最终输出文本回答
    """
    agent = create_react_agent(
        model,
        [bash, read_file, write_file],
        prompt=SYSTEM,
    )

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # 提取最终回答
    last_message = result["messages"][-1]
    return last_message.content


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("多工具 Agent - 输入问题，回车发送。输入 q 退出。\n")

    while True:
        try:
            query = input("\033[036mu02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit"):
            print("bye!")
            break

        try:
            response = agent_loop(query)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print()
