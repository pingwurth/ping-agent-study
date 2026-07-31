"""
U20 - Comprehensive Agent Turn（完整的代理轮次）
==================================================
本文件演示一个 **完整的代理轮次** 是如何工作的。
整合所有前面章节的概念，使用 LangGraph 实现。

核心概念：
  1. 一个 Agent Turn = 接收用户输入 → 处理 → 输出响应
  2. 每个 turn 可能包含多轮工具调用
  3. 涉及前面所有章节的概念：工具、权限、Hook、任务管理等

LangGraph 完整实现：
  ┌──────────────────────────────────────────────────────────┐
  │  使用 LangGraph 的 create_react_agent 构建完整 Agent：    │
  │                                                          │
  │  agent = create_react_agent(                             │
  │      model,                                              │
  │      tools=[bash, read_file, write_file, todo_write],   │
  │      prompt=system_prompt,                               │
  │      checkpointer=memory,  # 持久化状态                  │
  │  )                                                       │
  │                                                          │
  │  result = agent.invoke(                                  │
  │      {"messages": [HumanMessage(content=query)]},        │
  │      config={"configurable": {"thread_id": "session-1"}},│
  │  )                                                       │
  └──────────────────────────────────────────────────────────┘

完整的 Agent Turn 流程：
  ① 接收用户输入
  ② 构建请求（system prompt + tools + messages）
  ③ 调用模型
  ④ 处理响应（文本 or 工具调用）
  ⑤ 执行工具（权限检查 + Hook）
  ⑥ 循环回到 ③
  ⑦ 输出最终响应
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END

model = get_model()


# ── Agent Turn 的配置 ─────────────────────────────────────
@dataclass
class AgentConfig:
    """
    Agent 的完整配置。

    Claude Code 在每个 turn 开始时会组装这些配置：
      - system_prompt: 系统提示词（含规则、环境信息）
      - tools: 可用工具列表（内置 + MCP）
      - max_tokens: 最大输出 token 数
      - auto_approve: 是否自动批准所有工具调用
    """
    system_prompt: str = ""
    tools: list = field(default_factory=list)
    max_tokens: int = 8000
    auto_approve: bool = False


# ── Agent Turn 实现 ───────────────────────────────────────
class AgentTurn:
    """
    一个完整的 Agent Turn。

    使用 LangGraph 的 create_react_agent 实现，
    整合了所有前面章节的概念：
      - 工具执行（U01-U02）
      - 权限控制（U03）
      - Hook 系统（U04）
      - 任务管理（U05）
      - 上下文管理（U10）
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.tool_call_count = 0
        self.total_tokens = 0
        self.start_time = 0

        # 使用 LangGraph 构建 Agent
        self.agent = create_react_agent(
            model,
            config.tools,
            prompt=config.system_prompt,
        )

    def process(self, user_input: str) -> str:
        """
        处理一个完整的用户输入。

        Args:
            user_input: 用户的输入文本

        Returns:
            str: Agent 的最终文本响应
        """
        self.start_time = time.time()

        result = self.agent.invoke({
            "messages": [HumanMessage(content=user_input)],
        })

        # 统计工具调用次数
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                self.tool_call_count += len(msg.tool_calls)

        # 提取最终回答
        last_message = result["messages"][-1]
        return last_message.content

    def get_stats(self) -> dict:
        """获取本轮的统计信息。"""
        elapsed = time.time() - self.start_time
        return {
            "tool_calls": self.tool_call_count,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(elapsed, 2),
        }


# ── 工具定义 ──────────────────────────────────────────────
import subprocess

@tool
def bash(command: str) -> str:
    """Run a shell command.

    Args:
        command: 要执行的 shell 命令
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr


@tool
def read_file(path: str) -> str:
    """Read file contents.

    Args:
        path: 文件路径
    """
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file.

    Args:
        path: 文件路径
        content: 要写入的内容
    """
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote to {path}"
    except Exception as e:
        return f"Error: {e}"


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Comprehensive Agent Turn 完整代理轮次演示\n")

    # System prompt（模拟 Claude Code 的完整 system prompt）
    system_prompt = f"""You are a coding agent at {os.getcwd()}.
Use tools to solve tasks.

Available tools:
- bash: Execute shell commands
- read_file: Read file contents
- write_file: Write content to files

Be concise. Act, don't explain."""

    # 创建 Agent 配置
    config = AgentConfig(
        system_prompt=system_prompt,
        tools=[bash, read_file, write_file],
        max_tokens=8000,
    )

    # 创建 Agent Turn
    turn = AgentTurn(config)

    print("── Agent Turn 流程 ──")
    print("""
    ┌─────────────────────────────────────────────┐
    │  用户输入: "列出当前目录的文件"               │
    ├─────────────────────────────────────────────┤
    │  ① LangGraph create_react_agent 处理         │
    │     - system: coding agent prompt            │
    │     - tools: [bash, read_file, write_file]   │
    │     - messages: [user: "列出文件"]            │
    ├─────────────────────────────────────────────┤
    │  ② 模型返回 tool_calls: bash("ls -la")       │
    ├─────────────────────────────────────────────┤
    │  ③ LangGraph 自动执行工具                     │
    │     - 执行: ls -la                           │
    │     - 返回 ToolMessage                        │
    ├─────────────────────────────────────────────┤
    │  ④ 模型收到工具结果，生成最终回答              │
    ├─────────────────────────────────────────────┤
    │  ⑤ 输出给用户: "当前目录包含以下文件..."      │
    └─────────────────────────────────────────────┘
    """)

    print("── 统计信息示例 ──")
    stats = turn.get_stats()
    print(f"  工具调用次数: {stats['tool_calls']}")
    print(f"  Token 使用量: {stats['total_tokens']}")
    print(f"  耗时: {stats['elapsed_seconds']} 秒")

    print("\n── 全课程总结 ──")
    print("""
    Section 1 - 工具执行:
      U01: Agent Loop → LangGraph StateGraph / create_react_agent
      U02: Tool Use → @tool 装饰器
      U03: Permission → LangGraph 条件节点
      U04: Hooks → LangChain Callbacks

    Section 2 - 规划控制:
      U05: TodoWrite → @tool + State
      U06: Sub-Agent → LangGraph 子图
      U07: Skills → ChatPromptTemplate
      U08: System Prompt → SystemMessagePromptTemplate
      U09: Error Recovery → 条件边重试

    Section 3 - 记忆管理:
      U10: Context Compaction → trim_messages
      U11: Memory → BaseChatMessageHistory

    Section 4 - 并发调度:
      U12: Background Tasks → asyncio
      U13: Cron Scheduler → 后台线程 + Agent 执行器

    Section 5 - 多代理:
      U14: Task System → StateGraph DAG
      U15: Agent Teams → 多代理编排
      U16: Team Protocols → 消息传递 + Send API
      U17: Autonomous Agents → 循环图 + 条件边
      U18: Worktree Isolation → 子图封装
      U19: MCP Tools → langchain-mcp-adapters
      U20: Comprehensive Turn → create_react_agent 整合
    """)

    print("── LangChain/LangGraph 核心优势 ──")
    print("""
    1. @tool 装饰器: 自动从函数签名生成 JSON Schema
    2. create_react_agent: 一行代码创建 ReAct Agent
    3. StateGraph: 图结构编排，支持条件边、并行、循环
    4. ChatPromptTemplate: 模板化的提示词管理
    5. BaseChatMessageHistory: 标准化的记忆接口
    6. Callbacks: 无侵入的 Hook 机制
    7. MCP 适配器: 标准化的外部工具集成
    """)
