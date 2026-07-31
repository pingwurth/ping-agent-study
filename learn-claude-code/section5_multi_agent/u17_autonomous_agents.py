"""
U17 - Autonomous Agents（自主代理）
====================================
本文件演示 Claude Code 的 **自主代理** 机制：代理如何自主循环执行任务。

核心概念：
  1. 自主代理能够在没有人类干预的情况下持续工作
  2. 通过 "思考—行动—观察" 循环实现自主性
  3. 有明确的终止条件，避免无限循环
  4. 有质量门控（quality gates），确保输出质量
  5. 支持检测质量停滞（plateau detection）

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code 的 Agent Loop 就是一个自主循环：             │
  │                                                          │
  │  while not done:                                         │
  │      ① 思考：分析当前状态，规划下一步                    │
  │      ② 行动：调用工具执行操作                            │
  │      ③ 观察：获取工具执行结果                            │
  │      ④ 评估：检查是否达到目标                            │
  │                                                          │
  │  终止条件：                                              │
  │    - 任务完成（stop_reason == "end_turn"）               │
  │    - 达到最大循环次数                                    │
  │    - 用户中断                                            │
  │    - 质量停滞                                            │
  └──────────────────────────────────────────────────────────┘

思考-行动-观察循环：
  ┌────────┐     ┌────────┐     ┌──────────┐
  │ 思考   │ ──→ │ 行动   │ ──→ │ 观察/评估│
  │ Think  │     │ Act    │     │ Observe  │
  └────────┘     └────────┘     └──────────┘
       ↑                              │
       │          continue            │
       └──────────────────────────────┘
                                    │
                                    ↓ stop
                                   END

本文件使用 anthropic SDK 实现自主代理的 agent loop。
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client


# ══════════════════════════════════════════════════════════════
# 第一部分：终止条件
# ══════════════════════════════════════════════════════════════

@dataclass
class TerminationCondition:
    """
    定义自主循环的终止条件。

    Claude Code 的终止条件：
      - max_iterations:          最大循环次数（防止无限循环）
      - goal_reached:            目标是否达成
      - quality_plateau_threshold: 质量停滞阈值（连续 N 次无改善则停止）
      - user_interrupt:          用户是否中断

    终止优先级：
      1. 用户中断（最高优先级）
      2. 目标达成
      3. 质量停滞
      4. 达到最大循环次数
    """
    max_iterations: int = 20
    goal_reached: bool = False
    quality_plateau_threshold: int = 3
    user_interrupt: bool = False


# ══════════════════════════════════════════════════════════════
# 第二部分：质量评估器
# ══════════════════════════════════════════════════════════════

@dataclass
class QualityMetrics:
    """
    质量指标，用于评估代理的工作质量。

    Claude Code 会跟踪多种质量指标：
      - tests_passing:  测试通过率 (0.0 ~ 1.0)
      - coverage:       测试覆盖率 (0.0 ~ 1.0)
      - lint_errors:    代码规范错误数
      - build_success:  构建是否成功

    综合分数计算（加权平均）：
      score = tests_passing * 0.4
            + coverage * 0.3
            + build_success * 0.2
            + lint_score * 0.1
    """
    tests_passing: float = 0.0
    coverage: float = 0.0
    lint_errors: int = 0
    build_success: bool = False
    score: float = 0.0

    def calculate_score(self) -> float:
        """
        计算综合质量分数。

        权重分配：
          - 测试通过率: 40%（最重要）
          - 测试覆盖率: 30%
          - 构建成功:   20%
          - 代码规范:   10%

        Returns:
            float: 综合质量分数 (0.0 ~ 1.0)
        """
        self.score = (
            self.tests_passing * 0.4
            + self.coverage * 0.3
            + (1.0 if self.build_success else 0.0) * 0.2
            + max(0, 1.0 - self.lint_errors / 100) * 0.1
        )
        return self.score


# ══════════════════════════════════════════════════════════════
# 第三部分：自主代理
# ══════════════════════════════════════════════════════════════

class AutonomousAgent:
    """
    自主代理：能够独立循环执行任务直到完成。

    Agent Loop 流程：
      while not terminated:
          ① think()    - 分析当前状态，规划下一步
          ② act()      - 执行具体操作（调用 Claude API）
          ③ observe()  - 获取执行结果
          ④ evaluate() - 评估质量，检查终止条件

    Claude Code 的 Agent Loop 实现：
      - 使用 anthropic SDK 的 messages.create()
      - 处理 tool_use 响应
      - 循环直到 stop_reason == "end_turn"
      - 或达到 max_turns 限制
    """

    def __init__(self, name: str, system_prompt: str):
        """
        初始化自主代理。

        Args:
            name:          代理名称
            system_prompt: 系统提示词
        """
        self.name = name
        self.system_prompt = system_prompt
        self.client, self.model = create_client()
        self.iteration = 0
        self.quality_history: list[QualityMetrics] = []
        self.action_log: list[dict] = []
        self.messages: list[dict] = []  # 对话历史

    def evaluate_quality(self) -> QualityMetrics:
        """
        评估当前工作质量。

        简化实现：随着迭代次数增加，质量逐步提升。
        实际 Claude Code 中，会通过以下方式评估：
          - 运行测试套件
          - 检查测试覆盖率
          - 运行 linter
          - 尝试构建

        Returns:
            QualityMetrics: 质量指标
        """
        metrics = QualityMetrics(
            tests_passing=min(1.0, self.iteration * 0.2),
            coverage=min(1.0, self.iteration * 0.15),
            lint_errors=max(0, 10 - self.iteration * 2),
            build_success=self.iteration >= 3,
        )
        metrics.calculate_score()
        return metrics

    def detect_plateau(self, threshold: int = 3) -> bool:
        """
        检测质量是否停滞。

        如果最近 N 次迭代的质量改善都小于 0.05，
        则认为质量已经停滞，应该停止循环。

        Args:
            threshold: 检测窗口大小

        Returns:
            bool: 是否停滞
        """
        if len(self.quality_history) < threshold + 1:
            return False

        recent = self.quality_history[-threshold:]
        scores = [m.score for m in recent]
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        return all(imp < 0.05 for imp in improvements)

    def run(self, goal: str, termination: TerminationCondition = None) -> dict:
        """
        运行自主循环。

        这是自主代理的主循环，实现了完整的
        "思考—行动—观察—评估" 循环。

        Args:
            goal:        目标描述
            termination: 终止条件（使用默认值如果未指定）

        Returns:
            dict: 执行结果
                - iterations:     实际迭代次数
                - final_score:    最终质量分数
                - goal_reached:   是否达成目标
                - quality_history: 质量变化历史
                - action_log:     行动日志
        """
        if termination is None:
            termination = TerminationCondition()

        # 重置状态
        self.iteration = 0
        self.quality_history = []
        self.action_log = []
        self.messages = []

        # 主循环
        while self.iteration < termination.max_iterations:
            self.iteration += 1

            # ① 思考：分析状态，规划行动
            think_result = self._think(goal)

            # ② 行动：执行操作（调用 Claude API）
            action_result = self._act(think_result)

            # ③ 评估：检查质量
            quality = self.evaluate_quality()
            self.quality_history.append(quality)

            # 记录日志
            self.action_log.append({
                "iteration": self.iteration,
                "thought": think_result,
                "action": action_result[:100] if action_result else "",
                "quality_score": quality.score,
            })

            # ④ 检查终止条件
            # 条件 1：质量达标
            if quality.score >= 0.9:
                termination.goal_reached = True
                break

            # 条件 2：质量停滞
            if self.detect_plateau(termination.quality_plateau_threshold):
                break

            # 条件 3：用户中断
            if termination.user_interrupt:
                break

        return {
            "iterations": self.iteration,
            "final_score": self.quality_history[-1].score if self.quality_history else 0,
            "goal_reached": termination.goal_reached,
            "quality_history": [m.score for m in self.quality_history],
            "action_log": self.action_log,
        }

    def _think(self, goal: str) -> str:
        """
        思考阶段：分析当前状态，规划下一步行动。

        使用 Claude API 分析当前进展并制定计划。

        Args:
            goal: 目标描述

        Returns:
            str: 思考结果和行动计划
        """
        # 构建思考提示
        prompt = f"""当前目标: {goal}
当前迭代: {self.iteration}
质量历史: {[f'{m.score:.2f}' for m in self.quality_history[-3:]]}

分析当前状态，规划下一步行动。简短回答。"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception:
            return f"第 {self.iteration} 轮: 分析状态，规划下一步行动"

    def _act(self, plan: str) -> str:
        """
        行动阶段：根据计划执行操作。

        使用 Claude API 执行具体的编码任务。

        Args:
            plan: 行动计划

        Returns:
            str: 执行结果
        """
        self.messages.append({"role": "user", "content": plan})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=self.messages[-10:],  # 保留最近 10 条消息
            )
            result = response.content[0].text
            self.messages.append({"role": "assistant", "content": result})
            return result
        except Exception:
            return f"执行: {plan}"


# ══════════════════════════════════════════════════════════════
# 第四部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U17 - Autonomous Agents 自主代理演示")
    print("=" * 60)

    # ── 创建自主代理 ──────────────────────────────────────
    print("\n── 创建自主代理 ──")

    agent = AutonomousAgent(
        name="coder",
        system_prompt="You are an autonomous coding agent. Write clean Python code.",
    )

    termination = TerminationCondition(
        max_iterations=10,
        quality_plateau_threshold=3,
    )

    print(f"  代理名称: {agent.name}")
    print(f"  目标: 实现用户认证模块")
    print(f"  最大迭代: {termination.max_iterations}")
    print(f"  停滞阈值: {termination.quality_plateau_threshold}")

    # ── 运行自主循环 ──────────────────────────────────────
    print("\n── 运行自主循环 ──")
    print("  流程: think → act → observe → evaluate → (continue/stop)")

    result = agent.run(
        goal="实现用户认证模块，包括登录、注册、JWT token 管理",
        termination=termination,
    )

    # ── 显示结果 ──────────────────────────────────────────
    print("\n── 执行结果 ──")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  最终分数: {result['final_score']:.2f}")
    print(f"  目标达成: {result['goal_reached']}")
    print(f"  质量变化: {['%.2f' % s for s in result['quality_history']]}")

    # ── 行动日志 ──────────────────────────────────────────
    print("\n── 行动日志 ──")
    for log in result["action_log"]:
        print(f"  迭代 {log['iteration']}: 质量={log['quality_score']:.2f}")
        print(f"    思考: {log['thought'][:60]}...")

    # ── Claude Code Agent Loop 说明 ───────────────────────
    print("\n── Claude Code Agent Loop 机制说明 ──")
    print("""
    Claude Code 的 Agent Loop 是一个自主循环：

    1. 循环流程：
       while not done:
           ① 构建请求（system prompt + tools + messages）
           ② 调用 Claude API (messages.create)
           ③ 处理响应：
              - text → 返回给用户
              - tool_use → 执行工具，继续循环
           ④ 检查终止条件

    2. 终止条件：
       - stop_reason == "end_turn" → 代理完成任务
       - stop_reason == "max_tokens" → 达到 token 限制
       - 达到 max_turns → 达到最大轮次
       - 用户中断 → Ctrl+C

    3. 质量保证：
       - 每轮迭代评估代码质量
       - 运行测试检查正确性
       - 检测质量停滞，避免无效循环

    4. 与本文件的对应：
       - _think()  → 分析状态，规划行动
       - _act()    → 调用 Claude API 执行
       - evaluate_quality() → 检查质量指标
       - detect_plateau() → 检测停滞
    """)
