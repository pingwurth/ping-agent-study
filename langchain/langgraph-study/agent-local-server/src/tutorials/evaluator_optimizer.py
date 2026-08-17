"""模式 4：评估-优化循环（Evaluator-Optimizer）

来自 LangGraph 官方文档：
> One LLM call generates a response while another provides evaluation and feedback
> in a loop until the quality meets a certain threshold.

核心思想：
- 生成器生成内容
- 评估器评估质量
- 如果质量不达标，生成器根据反馈优化
- 循环直到质量达标或达到最大迭代次数

适用场景：
- 内容优化（文章、代码、设计）
- 答案改进（问答系统）
- 方案迭代（产品设计）

图结构：
    START
      ↓
   generate（生成器）
      ↓
   evaluate（评估器）
      ↓
   ┌────────────────┐
   ↓                │
needs_improvement   │
   ↓                │
   generate ←───────┘
   ↓
approved
   ↓
    END

实现方式：
使用条件边实现循环，评估器决定是否继续优化。
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from tutorials.models import EvaluatorState

# ============================================================================
# 辅助函数（模拟 LLM 调用）
# ============================================================================


def _simulate_llm(prompt: str) -> str:
    """模拟 LLM 调用。"""
    return "[模拟 LLM 响应] 基于提示词生成的内容"


def _simulate_evaluation(content: str, task: str) -> tuple[str, str]:
    """模拟评估器评估。

    实际项目中应使用结构化输出：
        class Evaluation(BaseModel):
            verdict: Literal["approved", "needs_improvement"]
            feedback: str
        structured_llm = llm.with_structured_output(Evaluation)
        result = structured_llm.invoke(evaluate_prompt)
        return result.verdict, result.feedback

    这里模拟评估逻辑：
    - 内容长度 > 100 字符 → 批准
    - 否则 → 需要改进
    """
    if len(content) > 100:
        return "approved", "内容质量良好，可以发布。"
    else:
        return "needs_improvement", "内容太短，请扩展并添加更多细节。"


# ============================================================================
# 生成器节点
# ============================================================================


def generate(state: EvaluatorState) -> dict:
    """生成器节点：生成或优化内容。

    生成器的行为取决于当前状态：
    - 首次迭代：根据任务描述生成初始内容
    - 后续迭代：根据评估反馈优化内容

    这展示了评估-优化循环的核心机制：
    - 生成器可以读取评估器的反馈
    - 根据反馈改进内容
    - 每次迭代都在前一次基础上改进
    """
    task = state["task"]
    iteration = state.get("iteration", 0)
    feedback = state.get("feedback")
    current_content = state.get("content")

    # 根据是否是首次迭代构造不同的提示词
    if iteration == 0:
        # 首次生成
        prompt = f"""
        请完成以下任务：

        任务：{task}

        要求：
        - 内容详细完整
        - 至少 200 字
        - 结构清晰
        """
    else:
        # 根据反馈优化
        prompt = f"""
        请优化以下内容：

        原始任务：{task}
        当前内容：{current_content}
        评估反馈：{feedback}

        要求：
        - 根据反馈改进
        - 保持原有优点
        - 增加更多细节
        """

    # 调用 LLM 生成/优化
    new_content = _simulate_llm(prompt)

    # 更新迭代次数
    new_iteration = iteration + 1

    return {
        "content": new_content,
        "iteration": new_iteration,
        "messages": [
            f"[generate] 第 {new_iteration} 次生成完成，内容长度 {len(new_content)} 字符"
        ],
    }


# ============================================================================
# 评估器节点
# ============================================================================


def evaluate(state: EvaluatorState) -> dict:
    """评估器节点：评估生成内容的质量。

    评估器的职责：
    1. 读取生成器的输出
    2. 根据质量标准评估
    3. 提供改进建议（如果需要）

    评估结果：
    - approved：质量达标，可以结束
    - needs_improvement：需要改进，继续循环

    设计要点：
    - 评估器应该是独立的（不依赖生成器的内部状态）
    - 评估标准应该明确（可以是规则或 LLM 判断）
    - 反馈应该具体可操作
    """
    content = state.get("content", "")
    task = state.get("task", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # 检查是否达到最大迭代次数
    if iteration >= max_iterations:
        return {
            "evaluation": "approved",
            "feedback": f"已达到最大迭代次数 {max_iterations}，强制结束。",
            "messages": [f"[evaluate] 达到最大迭代次数，强制结束"],
        }

    # 调用评估器（模拟）
    evaluation, feedback = _simulate_evaluation(content, task)

    return {
        "evaluation": evaluation,
        "feedback": feedback,
        "messages": [f"[evaluate] 评估结果：{evaluation}，反馈：{feedback}"],
    }


# ============================================================================
# 路由函数
# ============================================================================


def should_continue(state: EvaluatorState) -> Literal["generate", "__end__"]:
    """路由函数：根据评估结果决定是否继续优化。

    这是评估-优化循环的核心控制逻辑：
    - 如果评估结果是 "approved" → 结束
    - 如果评估结果是 "needs_improvement" → 继续优化

    关于最大迭代次数：
    - 在 evaluate 节点中检查
    - 达到最大次数时强制返回 "approved"
    - 防止无限循环
    """
    evaluation = state.get("evaluation", "needs_improvement")

    if evaluation == "approved":
        return "__end__"
    else:
        return "generate"


# ============================================================================
# 图构建
# ============================================================================


def build_evaluator_optimizer_graph() -> StateGraph:
    """构建并返回评估-优化循环图。

    关键点：
    1. 使用条件边实现循环
    2. 评估器决定是否继续
    3. 生成器根据反馈优化

    关于循环的实现：
    - add_conditional_edges() 定义条件路由
    - 路由函数返回 "generate" 或 "__end__"
    - LangGraph 根据返回值决定下一步
    """
    # 创建 StateGraph
    workflow = StateGraph(EvaluatorState)

    # 添加节点
    workflow.add_node("generate", generate)
    workflow.add_node("evaluate", evaluate)

    # 添加边：START → generate
    workflow.add_edge(START, "generate")

    # 添加边：generate → evaluate
    workflow.add_edge("generate", "evaluate")

    # 添加条件边：evaluate → 根据评估结果路由
    workflow.add_conditional_edges(
        source="evaluate",
        path=should_continue,
        path_map={
            "generate": "generate",
            "__end__": END,
        },
    )

    return workflow


# ============================================================================
# 导出编译后的图
# ============================================================================

# 构建并编译图
graph = build_evaluator_optimizer_graph().compile()
