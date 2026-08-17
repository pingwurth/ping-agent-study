"""模式 2：并行化工作流（Parallelization）

来自 LangGraph 官方文档：
> Parallelization allows multiple LLM calls to run simultaneously,
> with their outputs aggregated downstream.

核心思想：
- 多个独立任务同时执行
- 利用并行性提高效率
- 最后聚合所有结果

适用场景：
- 多维度分析（情感 + 关键词 + 摘要 + 实体）
- 批量处理（多个文档同时处理）
- 对比评估（多个方案同时评估）

图结构：
              START
                ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
analyze    extract    generate
sentiment  keywords   summary
    ↓          ↓          ↓
    └──────────┼──────────┘
               ↓
         aggregate_results
               ↓
             END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from tutorials.models import AnalysisState

# ============================================================================
# 辅助函数（模拟 LLM 调用）
# ============================================================================


def _simulate_llm(prompt: str) -> str:
    """模拟 LLM 调用。"""
    return "[模拟 LLM 响应] 基于提示词生成的内容"


# ============================================================================
# 并行分析节点
# ============================================================================


def analyze_sentiment(state: AnalysisState) -> dict:
    """情感分析节点。

    并行节点 1：独立分析文本的情感倾向。

    并行化的关键特性：
    - 这个节点与其他分析节点同时运行
    - 不依赖其他节点的输出
    - 只读取 state['text']（输入数据）
    - 输出写入 state['sentiment']（独立字段）
    """
    prompt = f"""
    分析以下文本的情感倾向（积极/消极/中性）：

    文本：{state["text"][:500]}

    请给出：
    1. 情感分类（积极/消极/中性）
    2. 置信度（0-100）
    3. 简要理由
    """

    sentiment = _simulate_llm(prompt)

    return {
        "sentiment": sentiment,
        "messages": ["[analyze_sentiment] 情感分析完成"],
    }


def extract_keywords(state: AnalysisState) -> dict:
    """关键词提取节点。

    并行节点 2：独立提取文本的关键词。

    注意：这个节点与 analyze_sentiment 同时运行，
    它们之间没有依赖关系。
    """
    prompt = f"""
    从以下文本中提取 5-10 个关键词：

    文本：{state["text"][:500]}

    要求：
    - 按重要性排序
    - 包含实体（人名、地名、组织）
    - 包含主题词
    """

    keywords = _simulate_llm(prompt)

    return {
        "keywords": keywords,
        "messages": ["[extract_keywords] 关键词提取完成"],
    }


def generate_summary(state: AnalysisState) -> dict:
    """摘要生成节点。

    并行节点 3：独立生成文本摘要。

    这三个节点（sentiment, keywords, summary）会同时运行，
    LangGraph 自动处理并行执行和结果收集。
    """
    prompt = f"""
    为以下文本生成简短摘要（50-100 字）：

    文本：{state["text"][:500]}

    要求：
    - 保留核心信息
    - 语言简洁
    - 突出要点
    """

    summary = _simulate_llm(prompt)

    return {
        "summary": summary,
        "messages": ["[generate_summary] 摘要生成完成"],
    }


def extract_entities(state: AnalysisState) -> dict:
    """实体识别节点。

    并行节点 4：独立识别文本中的实体。

    展示并行化的扩展性：
    - 可以轻松添加更多并行分析维度
    - 每个维度独立运行，互不影响
    """
    prompt = f"""
    识别以下文本中的命名实体：

    文本：{state["text"][:500]}

    实体类型：
    - 人名
    - 地名
    - 组织名
    - 时间
    - 数字
    """

    entities = _simulate_llm(prompt)

    return {
        "entities": entities,
        "messages": ["[extract_entities] 实体识别完成"],
    }


# ============================================================================
# 聚合节点
# ============================================================================


def aggregate_results(state: AnalysisState) -> dict:
    """聚合节点：汇总所有并行分析的结果。

    这是并行化工作流的关键节点：
    - 等待所有并行节点完成
    - 读取所有节点的输出
    - 生成最终的综合报告

    LangGraph 保证：
    - 所有并行节点完成后才执行聚合节点
    - 聚合节点可以访问所有并行节点的输出
    """
    # 读取所有并行节点的输出
    sentiment = state.get("sentiment", "未完成")
    keywords = state.get("keywords", "未完成")
    summary = state.get("summary", "未完成")
    entities = state.get("entities", "未完成")

    # 构造综合报告
    prompt = f"""
    基于以下分析结果，生成综合报告：

    情感分析：{sentiment}
    关键词：{keywords}
    摘要：{summary}
    实体识别：{entities}

    请生成结构化的分析报告。
    """

    final_report = _simulate_llm(prompt)

    return {
        "final_report": final_report,
        "messages": ["[aggregate_results] 综合报告生成完成"],
    }


# ============================================================================
# 图构建
# ============================================================================


def build_parallelization_graph() -> StateGraph:
    """构建并返回并行化工作流图。

    关键点：
    1. 多个节点从同一源节点出发（START → 多个分析器）
    2. 这些节点会并行执行
    3. 所有并行节点完成后，执行聚合节点

    LangGraph 的并行执行机制：
    - 当一个节点有多个出边时，这些目标节点会并行执行
    - 使用 fan-out/fan-in 模式
    - 聚合节点等待所有并行节点完成
    """
    # 创建 StateGraph
    workflow = StateGraph(AnalysisState)

    # 添加并行分析节点
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("extract_keywords", extract_keywords)
    workflow.add_node("generate_summary", generate_summary)
    workflow.add_node("extract_entities", extract_entities)

    # 添加聚合节点
    workflow.add_node("aggregate_results", aggregate_results)

    # 添加边：START → 所有并行节点（fan-out）
    # LangGraph 会自动并行执行这些节点
    workflow.add_edge(START, "analyze_sentiment")
    workflow.add_edge(START, "extract_keywords")
    workflow.add_edge(START, "generate_summary")
    workflow.add_edge(START, "extract_entities")

    # 添加边：所有并行节点 → 聚合节点（fan-in）
    # LangGraph 会等待所有并行节点完成后再执行聚合节点
    workflow.add_edge("analyze_sentiment", "aggregate_results")
    workflow.add_edge("extract_keywords", "aggregate_results")
    workflow.add_edge("generate_summary", "aggregate_results")
    workflow.add_edge("extract_entities", "aggregate_results")

    # 添加边：聚合节点 → END
    workflow.add_edge("aggregate_results", END)

    return workflow


# ============================================================================
# 导出编译后的图
# ============================================================================

# 构建并编译图
graph = build_parallelization_graph().compile()
