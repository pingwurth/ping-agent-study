"""集成测试 — 测试图的完整运行流程。

集成测试直接调用编译后的图，验证端到-end行为。
包括：
1. 正常流程（question 类型，经过 search_documentation）
2. Bug 流程（经过 bug_tracking）
3. Billing 流程（需要人工审核，使用 interrupt/resume）
4. 人工审核拒绝场景

关于 ainvoke 和 interrupt 的行为：
- 当图遇到 interrupt() 时，ainvoke 不会阻塞
- 而是返回当前状态，其中包含 __interrupt__ 字段
- __interrupt__ 列表中的每个元素包含 value（审核问题）和 id（用于 resume）
- 通过 Command(resume={...}) 恢复执行，id 会自动匹配
"""

import pytest
from langgraph.types import Command

from agent.graph import graph

pytestmark = pytest.mark.anyio


# ============================================================================
# 辅助函数
# ============================================================================

def _make_input(
    email_content: str = "How do I reset my password?",
    sender_email: str = "customer@example.com",
    email_id: str = "EMAIL-001",
) -> dict:
    """构造图的输入数据。"""
    return {
        "email_content": email_content,
        "sender_email": sender_email,
        "email_id": email_id,
        "classification": None,
        "search_results": None,
        "customer_history": None,
        "draft_response": None,
        "messages": None,
    }


# ============================================================================
# 测试：Question 类型（咨询）— 经过 search_documentation
# ============================================================================

@pytest.mark.langsmith
async def test_question_flow() -> None:
    """测试咨询类邮件的处理流程。

    预期路径：read_email → classify_intent → search_documentation
    → draft_response → human_review（interrupt 暂停）

    所有路径最终都会到达 human_review 并暂停（因为 interrupt）。
    """
    inputs = _make_input(
        email_content="How do I reset my password?",
        sender_email="user@test.com",
        email_id="TEST-Q-001",
    )
    config = {"configurable": {"thread_id": "test-question-001"}}

    result = await graph.ainvoke(inputs, config=config)

    # 验证分类结果
    assert result["classification"] is not None
    assert result["classification"]["intent"] == "question"

    # 验证搜索结果（question 类型会经过 search_documentation）
    assert result["search_results"] is not None
    assert len(result["search_results"]) > 0

    # 验证有回复草稿（所有路径都经过 draft_response）
    assert result["draft_response"] is not None
    assert len(result["draft_response"]) > 0

    # 验证图在 human_review 处暂停（__interrupt__ 字段存在）
    assert "__interrupt__" in result
    assert len(result["__interrupt__"]) > 0

    # 验证消息日志记录了各步骤
    messages = result.get("messages") or []
    assert any("classify_intent" in msg for msg in messages)
    assert any("search_documentation" in msg for msg in messages)
    assert any("draft_response" in msg for msg in messages)


# ============================================================================
# 测试：Bug 类型 — 经过 bug_tracking
# ============================================================================

@pytest.mark.langsmith
async def test_bug_flow_creates_ticket() -> None:
    """测试 Bug 报告类邮件的处理流程。

    预期路径：read_email → classify_intent → bug_tracking
    → draft_response → human_review（interrupt 暂停）
    """
    inputs = _make_input(
        email_content="The app crashed when I tried to save. It's broken!",
        sender_email="user@test.com",
        email_id="TEST-B-001",
    )
    config = {"configurable": {"thread_id": "test-bug-001"}}

    result = await graph.ainvoke(inputs, config=config)

    # 验证分类结果
    assert result["classification"] is not None
    assert result["classification"]["intent"] == "bug"

    # 验证有回复草稿
    assert result["draft_response"] is not None

    # 验证消息日志包含 bug_tracking
    messages = result.get("messages") or []
    assert any("bug_tracking" in msg for msg in messages)


# ============================================================================
# 测试：Billing 类型 — interrupt/resume 完整流程
# ============================================================================

@pytest.mark.langsmith
async def test_billing_flow_with_resume() -> None:
    """测试账单类邮件的完整处理流程 — 验证 interrupt/resume 机制。

    预期路径：read_email → classify_intent → draft_response
    → human_review（interrupt 暂停）→ resume → send_reply → END

    这是 LangGraph 最核心的功能之一：
    1. 图在 human_review 处暂停，等待人工输入
    2. 通过 Command(resume=...) 恢复执行
    3. 暂停期间所有状态被保存在 checkpointer 中
    """
    inputs = _make_input(
        email_content="I was charged twice for my subscription!",
        sender_email="billing@test.com",
        email_id="TEST-BILL-001",
    )
    config = {"configurable": {"thread_id": "test-billing-001"}}

    # --- 第一次执行：图在 human_review 处暂停 ---
    result = await graph.ainvoke(inputs, config=config)

    # 验证分类结果是 billing
    assert result["classification"] is not None
    assert result["classification"]["intent"] == "billing"

    # 验证有回复草稿（billing 也经过 draft_response）
    assert result["draft_response"] is not None
    assert len(result["draft_response"]) > 0

    # 验证图在 human_review 处暂停
    assert "__interrupt__" in result
    interrupt_info = result["__interrupt__"][0]
    assert interrupt_info.value["intent"] == "billing"

    # --- 第二次执行：用 Command(resume=...) 恢复 ---
    human_decision = Command(resume={
        "approved": True,
        "edited_response": "我们已收到您的反馈，将尽快处理退款。",
        "feedback": "回复得不错",
    })

    final_result = await graph.ainvoke(human_decision, config=config)

    # 验证最终结果
    assert final_result["draft_response"] == "我们已收到您的反馈，将尽快处理退款。"

    # 验证消息日志包含 human_review 和 send_reply
    messages = final_result.get("messages") or []
    assert any("human_review" in msg for msg in messages)
    assert any("send_reply" in msg for msg in messages)


# ============================================================================
# 测试：人工审核拒绝
# ============================================================================

@pytest.mark.langsmith
async def test_human_review_rejection() -> None:
    """测试人工审核者拒绝回复的场景。

    当审核者拒绝时，图应该直接结束，不发送回复。
    """
    inputs = _make_input(
        email_content="I was charged twice!",
        sender_email="angry@test.com",
        email_id="TEST-REJ-001",
    )
    config = {"configurable": {"thread_id": "test-reject-001"}}

    # 第一次执行：暂停在 human_review
    await graph.ainvoke(inputs, config=config)

    # 审核者拒绝
    human_decision = Command(resume={
        "approved": False,
        "feedback": "回复语气不够诚恳，需要重写",
    })

    # 恢复执行 — 应该直接结束，不发送
    final_result = await graph.ainvoke(human_decision, config=config)

    # 验证没有执行 send_reply
    messages = final_result.get("messages") or []
    assert not any("send_reply" in msg for msg in messages)


# ============================================================================
# 测试：Complex 类型（复杂问题）— 直接到 draft_response
# ============================================================================

@pytest.mark.langsmith
async def test_complex_flow_direct_to_draft() -> None:
    """测试复杂问题类邮件 — 直接从 classify_intent 到 draft_response。

    预期路径：read_email → classify_intent → draft_response
    → human_review（interrupt 暂停）
    """
    inputs = _make_input(
        email_content="I need some assistance with something complicated.",
        sender_email="user@test.com",
        email_id="TEST-C-001",
    )
    config = {"configurable": {"thread_id": "test-complex-001"}}

    result = await graph.ainvoke(inputs, config=config)

    # 验证分类结果
    assert result["classification"] is not None
    assert result["classification"]["intent"] == "complex"

    # 验证没有搜索结果（complex 不经过 search_documentation）
    assert result["search_results"] is None

    # 验证有回复草稿
    assert result["draft_response"] is not None
