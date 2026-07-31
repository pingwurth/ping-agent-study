"""
U17 - Autonomous Agents（自主代理）
====================================
本文件演示 **自主代理** 机制：代理如何自主循环执行任务。
使用 LangGraph 循环图 + 条件边实现。

核心概念：
  1. 自主代理能够在没有人类干预的情况下持续工作
  2. 通过 "思考—行动—观察" 循环实现自主性
  3. 有明确的终止条件，避免无限循环
  4. 有质量门控（quality gates），确保输出质量

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  使用 LangGraph 的条件边实现自主循环：                    │
  │                                                          │
  │  graph.add_node("think", think_node)                     │
  │  graph.add_node("act", act_node)                         │
  │  graph.add_node("evaluate", evaluate_node)               │
  │                                                          │
  │  graph.add_conditional_edges("evaluate", should_continue │
  │      {"continue": "think", "stop": END})                 │
  │                                                          │
  │  终止条件在 should_continue 函数中判断                    │
  └──────────────────────────────────────────────────────────┘

终止条件：
  - 任务完成（达到目标状态）
  - 达到最大循环次数
  - 质量门控失败（连续多次未改善）
  - 用户中断
"""

import os
import time
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

model = get_model()


# ── 终止条件 ──────────────────────────────────────────────
@dataclass
class TerminationCondition:
    """
    定义自主循环的终止条件。

    Claude Code 的终止条件：
      - max_iterations: 最大循环次数
      - goal_reached: 目标是否达成
      - quality_plateau: 质量是否停滞
      - user_interrupt: 用户是否中断
    """
    max_iterations: int = 20
    goal_reached: bool = False
    quality_plateau_threshold: int = 3
    user_interrupt: bool = False


# ── 质量评估器 ────────────────────────────────────────────
@dataclass
class QualityMetrics:
    """
    质量指标，用于评估代理的工作质量。

    Claude Code 会跟踪多种质量指标：
      - tests_passing: 测试通过率
      - coverage: 测试覆盖率
      - lint_errors: 代码规范错误数
      - build_success: 构建是否成功
    """
    tests_passing: float = 0.0
    coverage: float = 0.0
    lint_errors: int = 0
    build_success: bool = False
    score: float = 0.0

    def calculate_score(self):
        """计算综合质量分数。"""
        self.score = (
            self.tests_passing * 0.4
            + self.coverage * 0.3
            + (1.0 if self.build_success else 0.0) * 0.2
            + max(0, 1.0 - self.lint_errors / 100) * 0.1
        )
        return self.score


# ── 自主代理 ──────────────────────────────────────────────
class AutonomousAgent:
    """
    自主代理：能够独立循环执行任务直到完成。

    在 LangGraph 中，自主代理用循环图实现：
      - 每个迭代 = 图的一次完整执行
      - 终止条件 = 条件边的判断
      - 质量评估 = State 中的质量字段
    """

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.iteration = 0
        self.quality_history: list[QualityMetrics] = []
        self.action_log: list[dict] = []

    def evaluate_quality(self) -> QualityMetrics:
        """评估当前工作质量（简化实现）。"""
        metrics = QualityMetrics(
            tests_passing=min(1.0, self.iteration * 0.2),
            coverage=min(1.0, self.iteration * 0.15),
            lint_errors=max(0, 10 - self.iteration * 2),
            build_success=self.iteration >= 3,
        )
        metrics.calculate_score()
        return metrics

    def detect_plateau(self, threshold: int = 3) -> bool:
        """检测质量是否停滞。"""
        if len(self.quality_history) < threshold + 1:
            return False

        recent = self.quality_history[-threshold:]
        scores = [m.score for m in recent]
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        return all(imp < 0.05 for imp in improvements)

    def run(self, goal: str, termination: TerminationCondition = None) -> dict:
        """运行自主循环。"""
        if termination is None:
            termination = TerminationCondition()

        self.iteration = 0
        self.quality_history = []
        self.action_log = []

        while self.iteration < termination.max_iterations:
            self.iteration += 1

            # ① 思考
            think_result = self._think(goal)

            # ② 行动
            action_result = self._act(think_result)

            # ③ 评估
            quality = self.evaluate_quality()
            self.quality_history.append(quality)

            self.action_log.append({
                "iteration": self.iteration,
                "thought": think_result,
                "action": action_result,
                "quality_score": quality.score,
            })

            # ④ 检查终止条件
            if quality.score >= 0.9:
                termination.goal_reached = True
                break

            if self.detect_plateau(termination.quality_plateau_threshold):
                break

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
        return f"第 {self.iteration} 轮: 分析状态，规划下一步行动"

    def _act(self, plan: str) -> str:
        return f"执行: {plan}"


# ── 使用 LangGraph 构建自主代理图 ────────────────────────
def build_autonomous_agent_graph():
    """
    使用 LangGraph 构建自主代理的循环图。

    图结构：
      ┌────────┐     ┌────────┐     ┌──────────┐
      │ think  │ ──→ │  act   │ ──→ │ evaluate │
      └────────┘     └────────┘     └──────────┘
           ↑                              │
           │          continue            │
           └──────────────────────────────┘
                                        │
                                        ↓ stop
                                       END
    """
    # 定义 State
    from typing import TypedDict, Annotated
    import operator

    class AutonomousState(TypedDict):
        messages: Annotated[list, operator.add]
        iteration: int
        quality_score: float
        goal_reached: bool

    def think_node(state: AutonomousState):
        """思考节点：分析当前状态，规划下一步。"""
        iteration = state["iteration"] + 1
        return {
            "messages": [SystemMessage(content=f"第 {iteration} 轮思考")],
            "iteration": iteration,
        }

    def act_node(state: AutonomousState):
        """行动节点：执行具体操作。"""
        return {
            "messages": [HumanMessage(content=f"执行第 {state['iteration']} 轮行动")],
        }

    def evaluate_node(state: AutonomousState):
        """评估节点：检查结果质量。"""
        # 简化：每轮质量提升 0.15
        new_score = min(1.0, state["quality_score"] + 0.15)
        return {"quality_score": new_score}

    def should_continue(state: AutonomousState) -> str:
        """条件边：判断是否继续循环。"""
        if state["quality_score"] >= 0.9:
            return "stop"
        if state["iteration"] >= 10:
            return "stop"
        return "continue"

    # 构建图
    graph = StateGraph(AutonomousState)

    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("evaluate", evaluate_node)

    graph.add_edge(START, "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue, {
        "continue": "think",
        "stop": END,
    })

    return graph.compile()


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Autonomous Agent 自主代理演示\n")

    # 方式 1：使用简单的循环类
    print("── 方式 1: 简单循环 ──")
    agent = AutonomousAgent(
        name="coder",
        system_prompt="You are an autonomous coding agent.",
    )

    termination = TerminationCondition(
        max_iterations=10,
        quality_plateau_threshold=3,
    )

    print(f"  目标: 实现用户认证模块")
    print(f"  最大迭代: {termination.max_iterations}")
    print()

    result = agent.run(
        goal="实现用户认证模块，包括登录、注册、JWT token 管理",
        termination=termination,
    )

    print("── 执行结果 ──")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  最终分数: {result['final_score']:.2f}")
    print(f"  目标达成: {result['goal_reached']}")
    print(f"  质量变化: {['%.2f' % s for s in result['quality_history']]}")

    # 方式 2：使用 LangGraph 循环图
    print("\n── 方式 2: LangGraph 循环图 ──")
    print("  图结构: think → act → evaluate → (continue/stop)")
    print("  终止条件: quality_score >= 0.9 或 iteration >= 10")
    print()

    graph = build_autonomous_agent_graph()
    result = graph.invoke({
        "messages": [],
        "iteration": 0,
        "quality_score": 0.0,
        "goal_reached": False,
    })

    print(f"  最终迭代: {result['iteration']}")
    print(f"  最终质量: {result['quality_score']:.2f}")
