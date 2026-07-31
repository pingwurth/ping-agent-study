"""
U06 - Sub-Agent（子代理）
==========================
本文件演示 **Sub-Agent（子代理）** 机制：主代理如何委派任务给子代理。
使用原生 Anthropic SDK 实现。

核心概念：
  1. 子代理是主代理启动的独立 Claude 实例
  2. 每个子代理有自己的消息历史、系统提示词和工具集
  3. 子代理完成后将结果返回给主代理
  4. 多个子代理可以并行运行，提高效率

为什么需要子代理？
  ┌──────────────────────────────────────────────────────────┐
  │  问题：上下文窗口有限                                     │
  │                                                          │
  │  如果所有任务都在同一个对话中进行：                        │
  │    - 上下文窗口会被大量中间结果填满                        │
  │    - 模型容易"忘记"早期的重要信息                          │
  │    - 无法并行处理多个独立任务                              │
  │                                                          │
  │  解决方案：子代理                                         │
  │    - 每个子代理有独立的上下文窗口                          │
  │    - 只有最终结果返回给主代理                              │
  │    - 多个子代理可以同时运行                                │
  │    - 每个子代理可以有不同的专业角色                        │
  └──────────────────────────────────────────────────────────┘

Claude Code 中的子代理：
  Claude Code 使用 Claude Agent SDK 的 Agent 类来创建子代理。
  每个子代理是一个独立的 Claude 实例，有自己的：
    - system_prompt（定义角色和行为）
    - tools（可用工具列表）
    - messages（独立的对话历史）
"""

import os
import sys
import json
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ============================================================
# 初始化客户端
# ============================================================
client, MODEL = create_client()


# ════════════════════════════════════════════════════════════
# 第一部分：SubAgent - 独立的 Claude 实例
# ════════════════════════════════════════════════════════════

class SubAgent:
    """
    子代理：一个独立的 Claude 实例，专门处理特定类型的任务。

    每个子代理的核心属性：
      - name:          代理名称（如 "code-searcher"）
      - system_prompt: 系统提示词（定义角色和专业领域）
      - tools:         可用工具列表
      - messages:      独立的对话历史（不与主代理共享）

    与主代理的关键区别：
      - 子代理的 system_prompt 更加专业化（如"你是安全专家"）
      - 子代理的 tools 可能与主代理不同
      - 子代理完成后，只有最终结果返回给主代理
      - 子代理的中间过程对主代理不可见

    这种设计的好处：
      - 主代理不需要了解子代理的执行细节
      - 子代理可以自由探索，不受主代理上下文的约束
      - 多个子代理可以并行运行，互不干扰
    """

    def __init__(self, name: str, system_prompt: str, tools: list = None):
        """
        初始化子代理。

        Args:
            name:          子代理名称，用于日志和调试
            system_prompt: 系统提示词，定义子代理的角色和行为
            tools:         可用工具列表（JSON Schema 格式）
        """
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        # 每个子代理维护自己的消息历史
        # 这是子代理独立性的核心：它们的对话上下文完全隔离
        self.messages: list[dict] = []

    def run(self, task: str, max_turns: int = 10) -> str:
        """
        执行任务并返回结果。

        内部使用 agent loop（代理循环）：
          1. 将任务作为用户消息发送
          2. 模型返回响应（文本或工具调用）
          3. 如果有工具调用 → 执行工具 → 将结果发回 → 重复步骤 2
          4. 如果是文本 → 返回最终结果

        Args:
            task:      任务描述
            max_turns: 最大循环次数，防止无限循环

        Returns:
            str: 子代理的最终回答文本
        """
        # 初始化对话：只包含当前任务
        self.messages = [{"role": "user", "content": task}]

        for turn in range(max_turns):
            # 调用 Claude API
            kwargs = {
                "model": MODEL,
                "max_tokens": 4096,
                "system": self.system_prompt,
                "messages": self.messages,
            }
            # 只有在有工具时才传入 tools 参数
            if self.tools:
                kwargs["tools"] = self.tools

            response = client.messages.create(**kwargs)

            # 如果模型要调用工具
            if response.stop_reason == "tool_use" and self.tools:
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        # 简单的工具执行：返回模拟结果
                        # 在实际应用中，这里应该执行真实的工具逻辑
                        result = f"[{self.name}] 工具 {block.name} 执行完成"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                self.messages.append({"role": "assistant", "content": response.content})
                self.messages.append({"role": "user", "content": tool_results})
                continue

            # 模型给出了最终回答
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text

        return f"[{self.name}] 达到最大轮次 {max_turns}，任务未完成"


# ════════════════════════════════════════════════════════════
# 第二部分：MainAgent - 协调多个子代理
# ════════════════════════════════════════════════════════════

class MainAgent:
    """
    主代理：协调多个子代理完成复杂任务。

    主代理的职责：
      1. 接收用户的复杂请求
      2. 分析任务，决定需要哪些子代理
      3. 启动一个或多个子代理
      4. 收集子代理的结果
      5. 综合结果，给用户最终回答

    子代理的注册和管理：
      - create_agent():    创建并注册子代理
      - delegate_task():   将任务委派给指定子代理
      - parallel_tasks():  并行执行多个任务

    在 Claude Code 中，主代理通过 Agent SDK 的 handoff 机制
    来委派任务给子代理。这里我们用更直观的方式实现。
    """

    def __init__(self):
        # 子代理注册表：name → SubAgent 实例
        self.sub_agents: dict[str, SubAgent] = {}

    def create_agent(self, name: str, system_prompt: str, tools: list = None) -> SubAgent:
        """
        创建并注册一个子代理。

        Args:
            name:          子代理名称
            system_prompt: 系统提示词
            tools:         可用工具列表

        Returns:
            SubAgent: 创建的子代理实例
        """
        agent = SubAgent(name=name, system_prompt=system_prompt, tools=tools)
        self.sub_agents[name] = agent
        return agent

    def delegate_task(self, agent_name: str, task: str) -> str:
        """
        将任务委派给指定的子代理。

        这是主代理调度子代理的核心方法。
        主代理根据任务类型选择合适的子代理。

        Args:
            agent_name: 子代理名称
            task:       任务描述

        Returns:
            str: 子代理的执行结果
        """
        agent = self.sub_agents.get(agent_name)
        if not agent:
            return f"Error: 未找到名为 '{agent_name}' 的子代理"
        return agent.run(task)

    def parallel_tasks(self, tasks: list[dict]) -> list[str]:
        """
        并行执行多个任务。

        使用 ThreadPoolExecutor 实现真正的并行执行。
        每个子代理在独立的线程中运行，互不阻塞。

        这对应 Claude Code 中的 parallel tool use（并行工具调用）：
          当主代理识别到多个独立任务时，可以同时启动多个子代理。

        Args:
            tasks: 任务列表，每个任务是 {"agent": str, "task": str}

        Returns:
            list[str]: 各任务的执行结果（顺序与输入一致）
        """
        results = [None] * len(tasks)

        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # 提交所有任务
            future_to_index = {}
            for i, task_info in enumerate(tasks):
                future = executor.submit(
                    self.delegate_task,
                    task_info["agent"],
                    task_info["task"]
                )
                future_to_index[future] = i

            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = f"Error: {e}"

        return results


# ════════════════════════════════════════════════════════════
# 第三部分：专家子代理的定义
# ════════════════════════════════════════════════════════════

# 以下定义了三个专家子代理的系统提示词。
# 在实际 Claude Code 中，这些提示词会更加详细，
# 包含具体的工具使用指南和行为规范。

CODE_SEARCHER_PROMPT = """你是一个代码搜索专家。

你的职责：
- 在项目中查找与任务相关的代码文件
- 分析代码结构和依赖关系
- 总结关键文件的功能和作用

工作方式：
1. 先理解任务需求
2. 搜索相关的文件和代码片段
3. 分析代码的结构和逻辑
4. 输出简洁的代码摘要

输出格式：
- 列出找到的相关文件
- 每个文件的主要功能
- 关键代码片段（如果有）"""

SECURITY_REVIEWER_PROMPT = """你是一个安全审查专家。

你的职责：
- 分析代码中的安全漏洞
- 检查 OWASP Top 10 风险
- 提供修复建议

审查清单：
1. 注入攻击（SQL/NoSQL/OS 注入）
2. 认证和授权问题
3. 敏感数据暴露
4. XSS 漏洞
5. CSRF 保护
6. 安全配置错误
7. 不安全的反序列化
8. 使用已知漏洞的组件
9. 日志和监控不足

输出格式：
- 发现的问题（按严重程度排序）
- 每个问题的位置和描述
- 修复建议"""

TEST_WRITER_PROMPT = """你是一个测试工程师。

你的职责：
- 为代码编写全面的测试用例
- 确保测试覆盖关键路径
- 包含边界情况和错误处理

测试原则：
1. AAA 模式：Arrange（准备）- Act（执行）- Assert（断言）
2. 测试应该独立且可重复
3. 测试应该快速执行
4. 测试应该有清晰的命名

输出格式：
- 测试文件的代码
- 测试用例的说明
- 覆盖的功能点"""


# ════════════════════════════════════════════════════════════
# 第四部分：程序入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  U06 - Sub-Agent（子代理）演示")
    print("=" * 60)

    # 创建主代理
    main = MainAgent()

    # 创建三个专家子代理
    # 每个子代理有不同的系统提示词，定义了不同的专业领域
    main.create_agent("code-searcher", CODE_SEARCHER_PROMPT)
    main.create_agent("security-reviewer", SECURITY_REVIEWER_PROMPT)
    main.create_agent("test-writer", TEST_WRITER_PROMPT)

    print("\n已创建 3 个专家子代理：")
    print("  1. code-searcher     - 代码搜索专家")
    print("  2. security-reviewer - 安全审查专家")
    print("  3. test-writer       - 测试工程师")

    # ── 演示：并行任务委派 ──
    print("\n\n── 演示：主代理将任务并行委派给 3 个子代理 ──\n")
    print("场景：用户要求 '分析项目的认证模块'\n")
    print("主代理将任务分解为 3 个子任务，并行执行：")
    print("  → code-searcher: 找到认证相关文件")
    print("  → security-reviewer: 分析安全漏洞")
    print("  → test-writer: 编写测试用例\n")

    results = main.parallel_tasks([
        {"agent": "code-searcher", "task": "找到项目中所有与用户认证相关的文件，包括登录、注册、JWT、session 等"},
        {"agent": "security-reviewer", "task": "分析认证模块的安全性，检查常见的安全漏洞"},
        {"agent": "test-writer", "task": "为认证模块编写单元测试和集成测试用例"},
    ])

    # 输出结果
    agent_names = ["code-searcher", "security-reviewer", "test-writer"]
    for name, result in zip(agent_names, results):
        print(f"\n── {name} 的结果 ──")
        # 只显示前 300 个字符，避免输出过长
        print(result[:300] + "..." if len(result) > 300 else result)

    # ── 架构说明 ──
    print("\n\n── 子代理架构说明 ──\n")
    print("""
    主代理 (MainAgent)
        │
        ├── 创建子代理 ──→ SubAgent("code-searcher", ...)
        │                     ├── 独立的 system_prompt
        │                     ├── 独立的 messages 历史
        │                     └── 独立的 agent loop
        │
        ├── 委派任务 ────→ SubAgent.run(task)
        │                     ├── 1. 发送任务给 Claude API
        │                     ├── 2. 模型分析任务
        │                     ├── 3. 可能调用工具（搜索、分析等）
        │                     ├── 4. 循环直到完成
        │                     └── 5. 返回最终结果
        │
        └── 收集结果 ────→ 综合所有子代理的结果
                            └── 给用户最终回答

    关键特性：
      - 子代理有独立的上下文窗口（不会污染主代理）
      - 多个子代理可以并行运行（使用线程池）
      - 每个子代理可以有不同的工具集
      - 子代理的中间过程对主代理不可见（只有最终结果）
    """)
