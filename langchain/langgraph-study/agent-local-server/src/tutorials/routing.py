"""模式 3：条件路由工作流（Routing）

来自 LangGraph 官方文档：
> Routing classifies an input and directs it to a specialized followup task.
> LLMs can be used to classify the input for subsequent directed handling.

核心思想：
- LLM 分析输入类型
- 根据类型选择不同的处理路径
- 每条路径专门处理一类输入

适用场景：
- 客服系统（根据问题类型路由到不同部门）
- 内容分发（根据内容类型选择处理方式）
- 多语言处理（根据语言选择翻译器）

图结构：
    START
      ↓
  classify_input（LLM 分类）
      ↓
  ┌─────────────────────────┐
  ↓         ↓         ↓     │
code_    math_    general_   │
handler  handler  handler    │
  ↓         ↓         ↓     │
  └─────────────────────────┘
      ↓
    END

实现方式：
使用 add_conditional_edges() 定义条件路由，
路由函数根据状态返回目标节点名称。
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from tutorials.models import RouterState

# ============================================================================
# 辅助函数（模拟 LLM 调用）
# ============================================================================


def _simulate_llm(prompt: str) -> str:
    """模拟 LLM 调用。"""
    return "[模拟 LLM 响应] 基于提示词生成的内容"


def _simulate_classification(text: str) -> Literal["code", "math", "general"]:
    """模拟 LLM 分类。

    实际项目中应使用结构化输出：
        from pydantic import BaseModel
        class Classification(BaseModel):
            category: Literal["code", "math", "general"]
        structured_llm = llm.with_structured_output(Classification)
        result = structured_llm.invoke(classify_prompt)
        return result.category
    """
    text_lower = text.lower()

    # 简单的关键词匹配模拟 LLM 分类
    code_keywords = [
        "code",
        "function",
        "variable",
        "bug",
        "error",
        "python",
        "javascript",
    ]
    math_keywords = ["calculate", "equation", "math", "number", "sum", "multiply"]

    if any(kw in text_lower for kw in code_keywords):
        return "code"
    if any(kw in text_lower for kw in math_keywords):
        return "math"
    return "general"


# ============================================================================
# 分类节点
# ============================================================================


def classify_input(state: RouterState) -> dict:
    """分类节点：使用 LLM 分析输入类型。

    路由工作流的第一步：
    1. 接收用户输入
    2. 使用 LLM 分析输入类型
    3. 将分类结果存入状态

    注意：这个节点不决定路由，只负责分类。
    路由逻辑在路由函数中实现（见 route_decision）。
    """
    # 调用 LLM 分类（模拟）
    category = _simulate_classification(state["user_input"])

    return {
        "category": category,
        "messages": [f"[classify_input] 分类结果：{category}"],
    }


# ============================================================================
# 专业化处理节点
# ============================================================================


def code_handler(state: RouterState) -> dict:
    """代码问题处理器。

    专门处理代码相关的问题：
    - 代码审查
    - Bug 修复建议
    - 代码优化

    这是路由工作流的"专业化"特性：
    - 每条路径专门处理一类输入
    - 可以使用不同的提示词和工具
    - 提高处理质量和效率
    """
    prompt = f"""
    你是一个专业的程序员。请帮助解决以下代码问题：

    用户问题：{state["user_input"]}

    请提供：
    1. 问题分析
    2. 解决方案
    3. 代码示例（如果需要）
    """

    response = _simulate_llm(prompt)

    return {
        "response": response,
        "messages": ["[code_handler] 代码问题处理完成"],
    }


def math_handler(state: RouterState) -> dict:
    """数学问题处理器。

    专门处理数学相关的问题：
    - 计算求解
    - 公式推导
    - 数学概念解释

    展示路由的优势：
    - 数学问题可以用专门的数学提示词
    - 可以集成计算工具
    - 提供更准确的数学解答
    """
    prompt = f"""
    你是一个数学专家。请帮助解决以下数学问题：

    用户问题：{state["user_input"]}

    请提供：
    1. 问题理解
    2. 解题步骤
    3. 最终答案
    """

    response = _simulate_llm(prompt)

    return {
        "response": response,
        "messages": ["[math_handler] 数学问题处理完成"],
    }


def general_handler(state: RouterState) -> dict:
    """通用问题处理器。

    处理不属于特定类别的问题：
    - 一般咨询
    - 闲聊
    - 其他问题

    这是路由的"兜底"路径：
    - 当输入不属于任何专业类别时使用
    - 提供通用的回答
    """
    prompt = f"""
    你是一个友好的助手。请回答以下问题：

    用户问题：{state["user_input"]}

    请提供有帮助的回答。
    """

    response = _simulate_llm(prompt)

    return {
        "response": response,
        "messages": ["[general_handler] 通用问题处理完成"],
    }


# ============================================================================
# 路由函数
# ============================================================================


def route_decision(
    state: RouterState,
) -> Literal["code_handler", "math_handler", "general_handler"]:
    """路由函数：根据分类结果选择处理节点。

    这是条件路由的核心：
    - 读取状态中的分类结果
    - 返回目标节点名称
    - LangGraph 根据返回值决定执行哪个节点

    路由函数的要求：
    - 接收状态作为参数
    - 返回节点名称字符串
    - 必须是纯函数（无副作用）
    """
    category = state.get("category", "general")

    # 根据分类结果路由到相应处理器
    if category == "code":
        return "code_handler"
    elif category == "math":
        return "math_handler"
    else:
        return "general_handler"


# ============================================================================
# 图构建
# ============================================================================


def build_routing_graph() -> StateGraph:
    """构建并返回条件路由工作流图。

    关键点：
    1. 使用 add_conditional_edges() 定义条件路由
    2. 路由函数根据状态返回目标节点
    3. 可以有多个分支，每个分支处理不同类型

    关于 add_conditional_edges：
    - source: 源节点名称
    - path: 路由函数（接收状态，返回节点名称）
    - path_map: 可选，映射路由函数返回值到节点名称
    """
    # 创建 StateGraph
    workflow = StateGraph(RouterState)

    # 添加节点
    workflow.add_node("classify_input", classify_input)
    workflow.add_node("code_handler", code_handler)
    workflow.add_node("math_handler", math_handler)
    workflow.add_node("general_handler", general_handler)

    # 添加边：START → classify_input
    workflow.add_edge(START, "classify_input")

    # 添加条件边：classify_input → 根据分类结果路由
    # route_decision 函数返回目标节点名称
    workflow.add_conditional_edges(
        source="classify_input",
        path=route_decision,
        path_map={
            "code_handler": "code_handler",
            "math_handler": "math_handler",
            "general_handler": "general_handler",
        },
    )

    # 添加边：所有处理器 → END
    workflow.add_edge("code_handler", END)
    workflow.add_edge("math_handler", END)
    workflow.add_edge("general_handler", END)

    return workflow


# ============================================================================
# 导出编译后的图
# ============================================================================

# 构建并编译图
graph = build_routing_graph().compile()
