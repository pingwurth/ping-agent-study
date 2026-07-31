"""
U15 - Agent Teams（代理团队）
==============================
本文件演示 **Agent Teams** 机制：多个专业化代理协同工作。
使用 LangGraph 多代理编排实现。

核心概念：
  1. Agent Team 是一组分工明确的专业化代理
  2. 每个代理有专长的领域（安全、测试、前端等）
  3. 主代理（Orchestrator）协调团队工作
  4. 代理之间通过消息传递协作

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  每个角色 = 一个 LangGraph 子图或 Agent                   │
  │  协调器 = 主图，控制执行顺序                              │
  │                                                          │
  │  graph.add_node("searcher", searcher_agent)              │
  │  graph.add_node("coder", coder_agent)                    │
  │  graph.add_node("tester", tester_agent)                  │
  │                                                          │
  │  串行: graph.add_edge("searcher", "coder")               │
  │  并行: 使用 Send API fan-out                              │
  │  审查: 条件边循环                                         │
  └──────────────────────────────────────────────────────────┘
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END

model = get_model()


# ── 代理角色定义 ──────────────────────────────────────────
@dataclass
class AgentRole:
    """
    代理的角色定义。

    每个代理角色包含：
      - name: 角色名称
      - system_prompt: 代理的系统提示词
      - tools: 可用工具列表
      - description: 角色描述
    """
    name: str
    system_prompt: str
    tools: list = field(default_factory=list)
    description: str = ""


# 预定义的代理角色
AGENT_ROLES = {
    "searcher": AgentRole(
        name="Code Searcher",
        system_prompt="""You are a code search specialist.
Your job is to find relevant files, functions, and patterns in the codebase.
Be thorough but concise. Report file paths and key findings.""",
        tools=["glob", "grep", "read"],
        description="搜索和定位代码",
    ),
    "coder": AgentRole(
        name="Code Writer",
        system_prompt="""You are an expert programmer.
Write clean, well-structured code following best practices.
Always include type annotations and handle errors explicitly.""",
        tools=["read", "write", "edit", "bash"],
        description="编写和修改代码",
    ),
    "tester": AgentRole(
        name="Test Engineer",
        system_prompt="""You are a test engineer.
Write comprehensive tests covering edge cases.
Follow AAA pattern (Arrange-Act-Assert).
Aim for >80% coverage.""",
        tools=["read", "write", "bash"],
        description="编写和运行测试",
    ),
    "reviewer": AgentRole(
        name="Code Reviewer",
        system_prompt="""You are a senior code reviewer.
Check for: security issues, code quality, performance, best practices.
Categorize issues as CRITICAL/HIGH/MEDIUM/LOW.
Be constructive but thorough.""",
        tools=["read", "grep"],
        description="代码审查",
    ),
    "security": AgentRole(
        name="Security Analyst",
        system_prompt="""You are a security expert.
Analyze code for vulnerabilities following OWASP Top 10.
Check for: injection, XSS, auth issues, data exposure.
Provide specific remediation steps.""",
        tools=["read", "grep"],
        description="安全分析",
    ),
}


# ── 团队代理 ──────────────────────────────────────────────
class TeamAgent:
    """
    团队中的单个代理。

    每个代理是一个独立的 LangGraph Agent，有自己的：
      - 角色和专长（通过 system prompt 定义）
      - 工具集
      - 对话上下文
    """

    def __init__(self, role: AgentRole):
        self.role = role
        self.agent = create_react_agent(
            model,
            [],  # 工具列表（演示中为空，实际使用时添加具体工具）
            prompt=role.system_prompt,
        )
        self.results = []

    def execute(self, task: str) -> str:
        """执行任务并返回结果。"""
        result = self.agent.invoke({
            "messages": [HumanMessage(content=task)],
        })

        last_message = result["messages"][-1]
        text = last_message.content
        self.results.append(text)
        return text


# ── 团队协调器 ────────────────────────────────────────────
class TeamOrchestrator:
    """
    团队协调器（主代理）。

    职责：
      ① 分析用户请求，确定需要哪些代理
      ② 分配任务给各个代理
      ③ 管理代理间的依赖关系
      ④ 收集和整合结果

    在 LangGraph 中，协调器可以用主图实现：
      - 每个角色 = 一个节点
      - 执行顺序 = 边
      - 条件边 = 动态路由
    """

    def __init__(self):
        self.agents: dict[str, TeamAgent] = {}
        self.task_history = []

    def add_agent(self, role_name: str) -> TeamAgent:
        """添加一个代理到团队。"""
        role = AGENT_ROLES.get(role_name)
        if not role:
            raise ValueError(f"Unknown role: {role_name}")
        agent = TeamAgent(role)
        self.agents[role_name] = agent
        return agent

    def sequential(self, tasks: list[dict]) -> list[str]:
        """
        串行执行任务（前一个的输出作为后一个的输入）。

        LangGraph 实现：
          graph.add_edge("searcher", "coder")
          graph.add_edge("coder", "tester")
        """
        results = []
        context = ""

        for task_info in tasks:
            role = task_info["role"]
            task = task_info["task"]

            if role not in self.agents:
                self.add_agent(role)

            full_task = task
            if context:
                full_task = f"前一个代理的结果:\n{context}\n\n你的任务:\n{task}"

            result = self.agents[role].execute(full_task)
            results.append(result)
            context = result

        return results

    def parallel(self, tasks: list[dict]) -> dict[str, str]:
        """
        并行执行任务（每个代理独立工作）。

        LangGraph 实现：使用 Send API fan-out。
        """
        results = {}

        for task_info in tasks:
            role = task_info["role"]
            task = task_info["task"]

            if role not in self.agents:
                self.add_agent(role)

            results[role] = self.agents[role].execute(task)

        return results

    def review_cycle(self, code_task: str, max_rounds: int = 2) -> str:
        """
        编码-审查循环。

        LangGraph 实现：使用条件边循环。
        """
        if "coder" not in self.agents:
            self.add_agent("coder")
        if "reviewer" not in self.agents:
            self.add_agent("reviewer")

        code = self.agents["coder"].execute(code_task)

        for round_num in range(max_rounds):
            review = self.agents["reviewer"].execute(
                f"审查以下代码，列出所有问题:\n\n{code}"
            )

            if "CRITICAL" not in review and "HIGH" not in review:
                return code

            code = self.agents["coder"].execute(
                f"根据审查反馈修改代码:\n\n原代码:\n{code}\n\n审查反馈:\n{review}"
            )

        return code


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Agent Teams 演示\n")

    orchestrator = TeamOrchestrator()

    print("── 串行任务流示例 ──")
    print("  任务: 实现用户认证功能")
    print("  流程: 搜索 → 编码 → 测试 → 审查")
    print()

    print("── 并行任务示例 ──")
    print("  任务: 多维度分析代码库")
    print("  并行: 搜索 + 安全分析 + 测试覆盖检查")
    print()

    print("── 审查循环示例 ──")
    print("  流程: 编码 → 审查 → 修改 → 审查 → 完成")
    print()

    print("── LangGraph 实现方式 ──")
    print("""
    # 串行: 使用边连接
    graph.add_edge("searcher", "coder")
    graph.add_edge("coder", "tester")
    graph.add_edge("tester", "reviewer")

    # 并行: 使用 Send API
    def fan_out(state):
        return [
            Send("searcher", state),
            Send("security", state),
            Send("tester", state),
        ]

    # 审查循环: 使用条件边
    def should_continue(state):
        if "CRITICAL" in state["review"]:
            return "coder"
        return END
    """)

    print("(演示完成。实际使用需要配置 API 密钥)")
