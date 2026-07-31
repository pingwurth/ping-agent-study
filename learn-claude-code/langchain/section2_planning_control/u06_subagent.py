"""
U06 - Sub-Agent（子代理）
==========================
本文件演示 **Sub-Agent（子代理）** 机制：主代理如何委派任务给子代理。
使用 LangGraph 子图实现。

核心概念：
  1. 子代理是主代理启动的独立 Agent 实例
  2. 每个子代理有自己的对话上下文，不污染主代理的上下文窗口
  3. 子代理完成后将结果返回给主代理
  4. 多个子代理可以并行运行，提高效率

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  子代理 = 独立的 LangGraph 子图                            │
  │                                                          │
  │  主图节点 "delegate":                                     │
  │    sub_agent = create_react_agent(model, tools, prompt)  │
  │    result = sub_agent.invoke({"messages": [task]})       │
  │    return {"messages": [result]}                         │
  │                                                          │
  │  优点：                                                   │
  │    - 子代理有独立的 message history                       │
  │    - 可以有不同的 system prompt（专家角色）                │
  │    - 可以有不同的工具集                                   │
  └──────────────────────────────────────────────────────────┘
"""

import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END

model = get_model()


# ── 子代理定义 ────────────────────────────────────────────
class SubAgent:
    """
    子代理：一个独立的 LangGraph Agent，执行特定任务后返回结果。

    在 LangGraph 中，子代理可以用 create_react_agent 创建：
      sub_agent = create_react_agent(model, tools, prompt=system_prompt)
      result = sub_agent.invoke({"messages": [task]})

    子代理的关键特性：
      - 独立的对话上下文（messages）
      - 独立的 system prompt（可以是专家角色）
      - 共享工具集（可以读写文件、执行命令）
      - 结果以单一消息返回给主代理
    """

    def __init__(self, name: str, system_prompt: str, tools: list = None):
        """
        初始化子代理。

        Args:
            name: 子代理名称（用于日志和调试）
            system_prompt: 子代理的系统提示词（定义其角色和行为）
            tools: 可用工具列表
        """
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.agent = create_react_agent(
            model,
            self.tools,
            prompt=system_prompt,
        )

    def run(self, task: str) -> str:
        """
        执行任务并返回结果。

        Args:
            task: 任务描述

        Returns:
            str: 子代理的最终回答
        """
        result = self.agent.invoke({
            "messages": [HumanMessage(content=task)],
        })

        # 提取最后一条 AI 消息的文本
        last_message = result["messages"][-1]
        return last_message.content


# ── 主代理：使用子代理 ───────────────────────────────────
class MainAgent:
    """
    主代理：协调多个子代理完成复杂任务。

    Claude Code 的实际工作方式：
      - 主代理接收用户的复杂请求
      - 分析任务，决定是否需要子代理
      - 启动一个或多个子代理并行工作
      - 收集子代理的结果
      - 综合结果，给用户最终回答

    在 LangGraph 中，主代理可以：
      1. 使用工具委派任务给子代理
      2. 使用 Send API 并行分发任务
      3. 使用子图调用独立的子代理图
    """

    def __init__(self):
        self.sub_agents: dict[str, SubAgent] = {}

    def create_agent(self, name: str, system_prompt: str, tools: list = None) -> SubAgent:
        """创建并注册一个子代理。"""
        agent = SubAgent(name=name, system_prompt=system_prompt, tools=tools)
        self.sub_agents[name] = agent
        return agent

    def delegate_task(self, agent_name: str, task: str) -> str:
        """将任务委派给指定的子代理。"""
        agent = self.sub_agents.get(agent_name)
        if not agent:
            return f"Error: agent '{agent_name}' not found"
        return agent.run(task)

    def parallel_tasks(self, tasks: list[dict]) -> list[str]:
        """
        并行执行多个任务。

        在 LangGraph 中可以使用 Send API 实现真正的并行：
          from langgraph.types import Send
          def fan_out(state):
              return [Send("agent", task) for task in tasks]

        这里简化为顺序执行演示。

        Args:
            tasks: [{"agent": str, "task": str}, ...]

        Returns:
            list[str]: 各任务的结果
        """
        results = []
        for task_info in tasks:
            result = self.delegate_task(task_info["agent"], task_info["task"])
            results.append(result)
        return results


# ── 使用 LangGraph 构建主代理图 ──────────────────────────
# 将子代理委派封装为 LangGraph 工具
def create_delegate_tool(main_agent: MainAgent):
    """创建一个委派任务的工具，供主代理使用。"""

    @tool
    def delegate_to_agent(agent_name: str, task: str) -> str:
        """Delegate a task to a specialized sub-agent.

        Available agents: code-searcher, security-reviewer, test-writer

        Args:
            agent_name: Name of the sub-agent to delegate to
            task: Task description for the sub-agent
        """
        return main_agent.delegate_task(agent_name, task)

    return delegate_to_agent


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Sub-Agent 演示\n")

    # 创建主代理
    main = MainAgent()

    # 创建专家子代理
    main.create_agent(
        "code-searcher",
        "You are a code search specialist. Find relevant code files and summarize their purpose.",
    )
    main.create_agent(
        "security-reviewer",
        "You are a security expert. Analyze code for security vulnerabilities.",
    )
    main.create_agent(
        "test-writer",
        "You are a test engineer. Write comprehensive test cases.",
    )

    # 演示：主代理将任务委派给子代理
    print("── 主代理收到任务：'分析项目的认证模块' ──\n")

    print("启动 3 个子代理...")
    results = main.parallel_tasks([
        {"agent": "code-searcher", "task": "找到项目中所有与认证相关的文件"},
        {"agent": "security-reviewer", "task": "分析认证模块的安全性"},
        {"agent": "test-writer", "task": "为认证模块编写测试用例"},
    ])

    for i, result in enumerate(results):
        print(f"\n子代理 {i+1} 结果: {result[:200]}...")

    print("\n\n── 使用 LangGraph 主代理图 ──")
    print("主代理可以将子代理委派封装为工具，通过 LangGraph 自动调度。\n")

    # 创建带委派工具的主代理
    delegate_tool = create_delegate_tool(main)
    main_agent_graph = create_react_agent(
        model,
        [delegate_tool],
        prompt="You are a project manager. Delegate tasks to specialized agents.",
    )

    print("(演示完成。实际使用需要配置 API 密钥)")
