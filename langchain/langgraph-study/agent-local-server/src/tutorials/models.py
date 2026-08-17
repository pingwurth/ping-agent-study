"""共享数据模型定义 — 教程示例的状态和类型。

本模块定义了所有教程示例共享的数据结构：
- StoryState: 顺序工作流状态
- AnalysisState: 并行化工作流状态
- RouterState: 路由工作流状态
- EvaluatorState: 评估-优化工作流状态
- AgentState: 代理状态

设计原则：
1. 每个工作流模式有独立的状态类型
2. 状态只存储原始数据，不存储格式化文本
3. 使用 TypedDict 让 LangGraph 能自动推断 schema
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

# ============================================================================
# 模式 1：顺序工作流（Prompt Chaining）状态
# ============================================================================


class StoryState(TypedDict):
    """故事生成工作流的状态。

    顺序工作流的核心思想：
    - 每个节点处理一个问题
    - 节点的输出是下一个节点的输入
    - 最终汇聚成完整结果

    Attributes:
        topic:          故事主题
        outline:        故事大纲（由 generate_outline 生成）
        characters:     角色设定（由 generate_characters 生成）
        story:          完整故事（由 write_story 生成）
        title:          故事标题（由 generate_title 生成）
        messages:       处理日志
    """

    topic: str
    outline: str | None
    characters: str | None
    story: str | None
    title: str | None
    messages: Annotated[list[str], operator.add]


# ============================================================================
# 模式 2：并行化工作流状态
# ============================================================================


class AnalysisState(TypedDict):
    """多维度分析工作流的状态。

    并行化的核心思想：
    - 多个分析器同时运行
    - 每个分析器负责一个维度
    - 最后聚合所有结果

    Attributes:
        text:               待分析的文本
        sentiment:          情感分析结果（由 analyze_sentiment 生成）
        keywords:           关键词提取结果（由 extract_keywords 生成）
        summary:            摘要生成结果（由 generate_summary 生成）
        entities:           实体识别结果（由 extract_entities 生成）
        final_report:       最终聚合报告（由 aggregate_results 生成）
        messages:           处理日志
    """

    text: str
    sentiment: str | None
    keywords: str | None
    summary: str | None
    entities: str | None
    final_report: str | None
    messages: Annotated[list[str], operator.add]


# ============================================================================
# 模式 3：条件路由工作流状态
# ============================================================================


class RouterState(TypedDict):
    """路由工作流的状态。

    条件路由的核心思想：
    - LLM 分析输入类型
    - 根据类型选择不同的处理路径
    - 每条路径专门处理一类输入

    Attributes:
        user_input:         用户输入内容
        category:           分类结果（由 classify_input 生成）
        response:           最终回复（由相应处理器生成）
        messages:           处理日志
    """

    user_input: str
    category: Literal["code", "math", "general"] | None
    response: str | None
    messages: Annotated[list[str], operator.add]


# ============================================================================
# 模式 4：评估-优化循环状态
# ============================================================================


class EvaluatorState(TypedDict):
    """评估-优化循环的状态。

    评估-优化的核心思想：
    - 生成器生成内容
    - 评估器评估质量
    - 如果质量不达标，生成器根据反馈优化
    - 循环直到质量达标或达到最大迭代次数

    Attributes:
        task:               要完成的任务描述
        content:            当前生成的内容
        evaluation:         评估结果（approved/needs_improvement）
        feedback:           评估器的反馈意见
        iteration:          当前迭代次数
        max_iterations:     最大迭代次数
        messages:           处理日志
    """

    task: str
    content: str | None
    evaluation: Literal["approved", "needs_improvement"] | None
    feedback: str | None
    iteration: int
    max_iterations: int
    messages: Annotated[list[str], operator.add]


# ============================================================================
# 模式 5：代理状态
# ============================================================================


class AgentState(TypedDict):
    """带工具的代理的状态。

    代理的核心思想：
    - LLM 决定是否需要使用工具
    - 如果需要，选择合适的工具执行
    - 将工具结果返回给 LLM
    - 循环直到 LLM 认为任务完成

    Attributes:
        messages:           对话历史（包含用户输入、AI 回复、工具调用结果）
        tool_results:       工具执行结果（用于记录和调试）
        final_answer:       最终答案
        iterations:         代理循环次数（防止无限循环）
    """

    messages: list[dict]
    tool_results: list[dict]
    final_answer: str | None
    iterations: int
