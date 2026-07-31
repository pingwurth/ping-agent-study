"""
U15 - Agent Teams（代理团队）
==============================
本文件演示 Claude Code 的 **Agent Teams** 机制：多个专业化代理协同工作。

核心概念：
  1. Agent Team 是一组分工明确的专业化代理
  2. 每个代理有专长的领域（安全、测试、前端等）
  3. 主代理（Orchestrator）协调团队工作
  4. 代理之间通过消息传递协作

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code 使用 Agent 工具创建子代理：                   │
  │                                                          │
  │  Agent(                                                  │
  │      description="安全分析",                              │
  │      prompt="分析代码安全性...",                          │
  │      subagent_type="security-reviewer"                   │
  │  )                                                       │
  │                                                          │
  │  多个代理可以并行运行：                                   │
  │  - Agent 1: 搜索代码                                     │
  │  - Agent 2: 安全分析                                     │
  │  - Agent 3: 性能审查                                     │
  └──────────────────────────────────────────────────────────┘

协作模式：
  ┌──────────────────────────────────────────────────────────┐
  │  1. 串行模式（Sequential）                                │
  │     搜索 → 编码 → 测试 → 审查                            │
  │     前一个代理的输出作为后一个的输入                       │
  │                                                          │
  │  2. 并行模式（Parallel）                                  │
  │     搜索 ─┬→ 安全分析                                    │
  │           ├→ 性能审查                                    │
  │           └→ 测试覆盖                                    │
  │     多个代理同时工作                                      │
  │                                                          │
  │  3. 审查循环（Review Cycle）                              │
  │     编码 → 审查 → 修改 → 审查 → 完成                     │
  │     反复迭代直到审查通过                                  │
  └──────────────────────────────────────────────────────────┘

本文件使用 anthropic SDK 实现 Agent 代理。
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client


# ══════════════════════════════════════════════════════════════
# 第一部分：代理角色定义
# ══════════════════════════════════════════════════════════════

@dataclass
class AgentRole:
    """
    代理的角色定义。

    每个代理角色包含：
      - name:          角色名称（如 "Code Searcher"）
      - system_prompt: 代理的系统提示词（定义专长和行为）
      - tools:         可用工具列表（如 ["read", "write"]）
      - description:   角色的中文描述

    Claude Code 中的代理角色通过 Agent 工具的参数定义：
      Agent(
          description="搜索和定位代码",
          prompt="You are a code search specialist...",
          subagent_type="researcher"
      )
    """
    name: str
    system_prompt: str
    tools: list = field(default_factory=list)
    description: str = ""


# 预定义的代理角色
# Claude Code 内置了多种专业代理
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


# ══════════════════════════════════════════════════════════════
# 第二部分：团队代理
# ══════════════════════════════════════════════════════════════

class TeamAgent:
    """
    团队中的单个代理。

    每个代理是一个独立的 Claude API 调用：
      - 使用 anthropic SDK 创建客户端
      - 通过 system_prompt 定义角色和专长
      - 通过 agent loop 处理工具调用

    Claude Code 的 Agent 工具实现原理：
      1. 创建独立的 Claude API 调用
      2. 设置角色特定的 system prompt
      3. 提供角色特定的工具集
      4. 运行 agent loop 直到任务完成
    """

    def __init__(self, role: AgentRole):
        """
        初始化代理。

        Args:
            role: 代理角色定义
        """
        self.role = role
        self.client, self.model = create_client()
        self.results = []

    def execute(self, task: str, max_turns: int = 3) -> str:
        """
        执行任务（Agent Loop）。

        Agent Loop 流程：
          1. 发送任务给 Claude（带 system prompt）
          2. 如果返回 tool_use，执行工具并继续循环
          3. 如果返回 text，返回最终结果

        Args:
            task:      要执行的任务描述
            max_turns: 最大循环次数

        Returns:
            str: 代理的最终响应
        """
        messages = [{"role": "user", "content": task}]

        for turn in range(max_turns):
            # 调用 Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.role.system_prompt,
                messages=messages,
            )

            # 提取文本响应
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)

            result = "\n".join(text_parts)
            self.results.append(result)

            # 如果没有工具调用，返回结果
            if response.stop_reason == "end_turn":
                return result

            # 处理工具调用（简化实现）
            if response.stop_reason == "tool_use":
                # 在实际实现中，这里会执行工具并继续循环
                # 这里简化为直接返回
                return result

        return result


# ══════════════════════════════════════════════════════════════
# 第三部分：团队协调器
# ══════════════════════════════════════════════════════════════

class TeamOrchestrator:
    """
    团队协调器（主代理）。

    职责：
      ① 分析用户请求，确定需要哪些代理
      ② 分配任务给各个代理
      ③ 管理代理间的依赖关系
      ④ 收集和整合结果

    在 Claude Code 中，协调器是主 Agent：
      - 接收用户输入
      - 决定是否需要创建子代理
      - 使用 Agent 工具启动子代理
      - 收集子代理的结果
      - 整合并返回给用户
    """

    def __init__(self):
        # 已创建的代理字典
        self.agents: dict[str, TeamAgent] = {}
        # 任务历史
        self.task_history = []

    def add_agent(self, role_name: str) -> TeamAgent:
        """
        添加一个代理到团队。

        Args:
            role_name: 角色名称（必须在 AGENT_ROLES 中定义）

        Returns:
            TeamAgent: 创建的代理实例
        """
        role = AGENT_ROLES.get(role_name)
        if not role:
            raise ValueError(f"Unknown role: {role_name}")
        agent = TeamAgent(role)
        self.agents[role_name] = agent
        return agent

    def sequential(self, tasks: list[dict]) -> list[str]:
        """
        串行执行任务。

        前一个代理的输出作为后一个代理的上下文。
        适合有依赖关系的工作流。

        示例流程：搜索 → 编码 → 测试 → 审查

        Args:
            tasks: 任务列表，每个任务包含 role 和 task

        Returns:
            list[str]: 每个代理的执行结果
        """
        results = []
        context = ""

        for task_info in tasks:
            role = task_info["role"]
            task = task_info["task"]

            # 如果代理不存在，自动创建
            if role not in self.agents:
                self.add_agent(role)

            # 将前一个代理的结果作为上下文
            full_task = task
            if context:
                full_task = f"前一个代理的结果:\n{context}\n\n你的任务:\n{task}"

            result = self.agents[role].execute(full_task)
            results.append(result)
            context = result

        return results

    def parallel(self, tasks: list[dict]) -> dict[str, str]:
        """
        并行执行任务。

        每个代理独立工作，互不干扰。
        适合无依赖关系的分析任务。

        示例：搜索 + 安全分析 + 测试覆盖检查

        注意：这里简化为顺序执行，实际 Claude Code 会并行启动多个 Agent。

        Args:
            tasks: 任务列表

        Returns:
            dict[str, str]: 每个角色的执行结果
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

        流程：
          1. 编码代理生成代码
          2. 审查代理审查代码
          3. 如果有 CRITICAL/HIGH 问题，编码代理修改
          4. 重复直到审查通过或达到最大轮次

        Args:
            code_task:  编码任务描述
            max_rounds: 最大审查轮次

        Returns:
            str: 最终的代码
        """
        # 确保编码和审查代理存在
        if "coder" not in self.agents:
            self.add_agent("coder")
        if "reviewer" not in self.agents:
            self.add_agent("reviewer")

        # 第一步：编码
        code = self.agents["coder"].execute(code_task)

        # 审查循环
        for round_num in range(max_rounds):
            # 审查
            review = self.agents["reviewer"].execute(
                f"审查以下代码，列出所有问题:\n\n{code}"
            )

            # 如果没有严重问题，返回
            if "CRITICAL" not in review and "HIGH" not in review:
                return code

            # 根据审查反馈修改代码
            code = self.agents["coder"].execute(
                f"根据审查反馈修改代码:\n\n原代码:\n{code}\n\n审查反馈:\n{review}"
            )

        return code


# ══════════════════════════════════════════════════════════════
# 第四部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U15 - Agent Teams 代理团队演示")
    print("=" * 60)

    orchestrator = TeamOrchestrator()

    # ── 模式 1：串行任务流 ────────────────────────────────
    print("\n── 模式 1: 串行任务流 ──")
    print("  流程: 搜索 → 编码 → 测试 → 审查")
    print("  每个代理的输出作为下一个代理的输入")
    print()
    print("  代码示例:")
    print("  results = orchestrator.sequential([")
    print('      {"role": "searcher", "task": "查找认证相关代码"},')
    print('      {"role": "coder", "task": "实现 JWT token 刷新"},')
    print('      {"role": "tester", "task": "编写单元测试"},')
    print('      {"role": "reviewer", "task": "审查代码质量"},')
    print("  ])")

    # ── 模式 2：并行任务 ──────────────────────────────────
    print("\n── 模式 2: 并行任务 ──")
    print("  任务: 多维度分析代码库")
    print("  并行: 搜索 + 安全分析 + 测试覆盖检查")
    print()
    print("  代码示例:")
    print("  results = orchestrator.parallel([")
    print('      {"role": "searcher", "task": "查找所有 API 端点"},')
    print('      {"role": "security", "task": "分析安全漏洞"},')
    print('      {"role": "tester", "task": "检查测试覆盖率"},')
    print("  ])")

    # ── 模式 3：审查循环 ──────────────────────────────────
    print("\n── 模式 3: 审查循环 ──")
    print("  流程: 编码 → 审查 → 修改 → 审查 → 完成")
    print()
    print("  代码示例:")
    print('  code = orchestrator.review_cycle(')
    print('      "实现用户登录功能，包括密码哈希和 JWT token",')
    print('      max_rounds=2')
    print("  )")

    # ── Claude Code Agent 工具说明 ────────────────────────
    print("\n── Claude Code Agent 工具说明 ──")
    print("""
    Claude Code 使用 Agent 工具创建子代理：

    1. 创建子代理：
       Agent(
           description="安全分析",
           prompt="分析代码中的安全漏洞...",
           subagent_type="security-reviewer"
       )

    2. 可用的代理类型：
       - researcher      → 代码搜索和研究
       - planner         → 实现规划
       - tdd-guide       → 测试驱动开发
       - code-reviewer   → 代码审查
       - security-reviewer → 安全分析

    3. 并行执行：
       - 多个 Agent 可以同时运行
       - 每个 Agent 有独立的上下文
       - 结果通过通知机制返回

    4. 隔离执行：
       - 使用 worktree 隔离工作目录
       - 每个 Agent 在独立的分支工作
       - 完成后可以合并或丢弃
    """)
