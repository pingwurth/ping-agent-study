"""单元测试 — 测试图的结构和节点函数。

单元测试不启动 LangGraph Server，直接测试 Python 对象。
"""

from langgraph.pregel import Pregel

from agent.graph import graph
from agent.models import EmailAgentState
from agent.nodes import _simulate_doc_search, _simulate_llm_classify


def test_graph_is_pregel_instance() -> None:
    """验证 graph 是一个 Pregel 实例（编译后的图）。"""
    assert isinstance(graph, Pregel)


def test_graph_has_expected_nodes() -> None:
    """验证图包含所有预期的节点。"""
    # 编译后的图暴露 nodes 属性
    node_names = set(graph.nodes.keys())
    expected = {
        "__start__",
        "read_email",
        "classify_intent",
        "search_documentation",
        "bug_tracking",
        "draft_response",
        "human_review",
        "send_reply",
    }
    assert expected.issubset(node_names), f"缺少节点: {expected - node_names}"


def test_simulate_llm_classify_billing() -> None:
    """验证 LLM 分类模拟能正确识别账单相关邮件。"""
    result = _simulate_llm_classify("I was charged twice for my subscription!")
    assert result["intent"] == "billing"
    assert result["urgency"] == "high"


def test_simulate_llm_classify_bug() -> None:
    """验证 LLM 分类模拟能正确识别 Bug 报告。"""
    result = _simulate_llm_classify("The app crashed when I clicked the button")
    assert result["intent"] == "bug"


def test_simulate_llm_classify_feature() -> None:
    """验证 LLM 分类模拟能正确识别功能请求。"""
    result = _simulate_llm_classify("It would be nice to have dark mode")
    assert result["intent"] == "feature"


def test_simulate_llm_classify_question() -> None:
    """验证 LLM 分类模拟能正确识别一般咨询。"""
    result = _simulate_llm_classify("How do I reset my password?")
    assert result["intent"] == "question"


def test_simulate_llm_classify_complex() -> None:
    """验证 LLM 分类模拟对无法识别的邮件返回 complex。"""
    result = _simulate_llm_classify("Hello, I need some assistance with something.")
    assert result["intent"] == "complex"


def test_simulate_doc_search_returns_results() -> None:
    """验证知识库搜索模拟返回结果。"""
    results = _simulate_doc_search("password reset")
    assert isinstance(results, list)
    assert len(results) >= 2
    assert all(isinstance(r, str) for r in results)


def test_email_agent_state_structure() -> None:
    """验证 EmailAgentState 的类型注解包含所有必要字段。"""
    # TypedDict 的 __annotations__ 包含所有字段
    annotations = EmailAgentState.__annotations__
    required_fields = [
        "email_content",
        "sender_email",
        "email_id",
        "classification",
        "search_results",
        "customer_history",
        "draft_response",
        "messages",
    ]
    for field in required_fields:
        assert field in annotations, f"缺少字段: {field}"
