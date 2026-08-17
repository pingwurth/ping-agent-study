"""节点实现 — 邮件客服 Agent 的所有处理步骤。

每个节点都是一个纯函数：
- 输入：当前状态 (EmailAgentState) + 运行时上下文
- 输出：状态更新字典，或者 Command 对象（同时包含更新和路由目标）

节点分类（来自 Thinking in LangGraph）：
- LLM 步骤：   classify_intent, draft_response — 调用大语言模型
- 数据步骤：   search_documentation, read_email — 从外部获取数据
- 动作步骤：   send_reply, bug_tracking — 执行外部操作
- 人工步骤：   human_review — 需要人工介入

注意：本文件中的 LLM 调用和外部 API 都是模拟实现，
实际项目中应替换为真实的 LangChain LLM 调用和 API 请求。
"""

from __future__ import annotations

import random
from typing import Literal

from langgraph.types import Command, interrupt

from agent.models import EmailAgentState

# ============================================================================
# 辅助函数（模拟 LLM 调用和外部 API）
# ============================================================================

def _simulate_llm_classify(email_content: str) -> dict:
    """模拟 LLM 的邮件分类调用。

    实际项目中应替换为：
        structured_llm = llm.with_structured_output(EmailClassification)
        classification = structured_llm.invoke(classification_prompt)

    这里用关键词匹配模拟 LLM 的分类行为。
    """
    content_lower = email_content.lower()

    # 根据关键词模拟分类逻辑
    if any(kw in content_lower for kw in ["charged", "billing", "invoice", "payment", "refund"]):
        return {
            "intent": "billing",
            "urgency": "high",
            "topic": "账单问题",
            "summary": "客户遇到了账单/支付相关问题",
        }
    if any(kw in content_lower for kw in ["bug", "error", "crash", "broken", "not working"]):
        return {
            "intent": "bug",
            "urgency": "medium",
            "topic": "产品故障",
            "summary": "客户报告了产品故障或错误",
        }
    if any(kw in content_lower for kw in ["feature", "request", "wish", "would be nice", "suggestion"]):
        return {
            "intent": "feature",
            "urgency": "low",
            "topic": "功能请求",
            "summary": "客户提出了新功能建议",
        }
    if any(kw in content_lower for kw in ["how", "what", "where", "help", "question"]):
        return {
            "intent": "question",
            "urgency": "low",
            "topic": "一般咨询",
            "summary": "客户有一般性问题需要解答",
        }
    # 默认为复杂问题
    return {
        "intent": "complex",
        "urgency": "medium",
        "topic": "复杂问题",
        "summary": "客户的问题需要多步骤处理",
    }


def _simulate_doc_search(query: str) -> list[str]:
    """模拟知识库搜索。

    实际项目中应替换为向量数据库搜索：
        results = vector_store.similarity_search(query, k=3)
        return [doc.page_content for doc in results]
    """
    # 模拟搜索结果
    mock_docs = [
        "重置密码：进入 设置 > 安全 > 修改密码。密码需至少12位，包含大小写字母和数字。",
        "退款政策：购买后30天内可申请全额退款。请联系客服处理。",
        "常见错误 E1001：清除浏览器缓存后重试。如问题持续，请提交工单。",
        "功能路线图：Q3 将推出移动端新版本，包含离线模式和推送通知。",
        "订阅管理：在账户设置中可以升级、降级或取消订阅。",
    ]
    # 随机返回 2-3 条结果，模拟搜索行为
    count = random.randint(2, 3)
    return random.sample(mock_docs, min(count, len(mock_docs)))


def _simulate_customer_history(email: str) -> dict:
    """模拟查询客户历史记录。

    实际项目中应替换为 CRM 系统查询。
    """
    return {
        "email": email,
        "plan": "professional",
        "tenure_months": random.randint(1, 36),
        "previous_tickets": random.randint(0, 5),
        "satisfaction_score": round(random.uniform(3.0, 5.0), 1),
    }


def _simulate_llm_draft(
    email_content: str,
    classification: dict,
    search_results: list[str] | None,
    customer_history: dict | None,
) -> str:
    """模拟 LLM 生成回复草稿。

    实际项目中应替换为：
        response = llm.invoke(draft_prompt)
        return response.content
    """
    intent = classification.get("intent", "unknown")
    topic = classification.get("topic", "未知")

    # 根据意图生成不同风格的回复
    if intent == "billing":
        return (
            f"尊敬的客户，\n\n"
            f"感谢您联系我们。关于您提到的账单问题（{topic}），"
            f"我们已经收到并正在处理中。\n\n"
            f"我们的财务团队将在 24 小时内核实相关信息，"
            f"如有任何疑问会第一时间与您联系。\n\n"
            f"如有其他问题，请随时回复此邮件。\n\n"
            f"此致\n客服团队"
        )
    if intent == "bug":
        docs_hint = ""
        if search_results:
            docs_hint = f"\n\n根据我们的知识库，您可以尝试：\n- {search_results[0]}"
        return (
            f"尊敬的客户，\n\n"
            f"感谢您报告此问题。我们已将您的反馈转交给技术团队。\n"
            f"工单编号将在 2 小时内通过邮件发送给您。{docs_hint}\n\n"
            f"此致\n技术支持团队"
        )
    if intent == "feature":
        return (
            "尊敬的客户，\n\n"
            "感谢您的功能建议！我们非常重视用户的反馈。\n"
            "您的建议已记录到我们的产品路线图中，"
            "产品团队会在规划会议中评估。\n\n"
            "此致\n产品团队"
        )
    # question / complex / default
    return (
        f"尊敬的客户，\n\n"
        f"感谢您的来信。关于您咨询的问题，以下是我们的回复：\n\n"
        f"{search_results[0] if search_results else '我们正在为您查询相关信息。'}\n\n"
        f"如有更多疑问，请随时联系我们。\n\n"
        f"此致\n客服团队"
    )


# ============================================================================
# 节点实现
# ============================================================================


def read_email(state: EmailAgentState) -> dict:
    """节点 1：读取并预处理邮件。

    这是图的入口节点，负责：
    1. 提取邮件关键信息（发件人、内容、ID）
    2. 记录处理开始日志

    输入：EmailAgentState（包含 email_content, sender_email, email_id）
    输出：状态更新字典 — 追加一条消息到 messages 列表

    设计要点：
    - 节点返回的是"更新"，LangGraph 会自动与现有状态合并
    - 对于列表字段，返回新列表会替换旧值（而非追加）
      所以这里需要手动合并：旧 messages + 新消息
    """
    # 获取当前消息列表（可能为 None）
    current_messages = state.get("messages") or []

    # 返回状态更新 — 注意：这里是替换整个 messages 列表
    return {
        "messages": current_messages + [
            f"[read_email] 开始处理邮件 {state['email_id']}，"
            f"发件人: {state['sender_email']}"
        ],
    }


def classify_intent(
    state: EmailAgentState,
) -> Command[
    Literal["search_documentation", "draft_response", "bug_tracking"]
]:
    """节点 2：使用 LLM 分类邮件意图并动态路由。

    这是 LangGraph 最强大的模式之一 —— 节点内部决定下一步去哪。
    通过返回 Command 对象，节点同时提供：
    - update: 状态更新（分类结果）
    - goto:   下一个要执行的节点名称

    路由逻辑：
    - question 或 feature → search_documentation（查知识库）→ draft_response
    - bug                 → bug_tracking（创建工单）→ draft_response
    - 其他（billing/complex/critical）→ draft_response → human_review

    所有路径最终都经过 draft_response → human_review：
    - draft_response 生成回复草稿
    - human_review 审核草稿（通过 interrupt 暂停等待人工决策）
    - 审核通过后 send_reply 发送

    这比在图层面定义条件边更清晰 —— 路由逻辑和产生路由的决策在同一个函数里。
    """
    # --- 模拟 LLM 调用 ---
    # 实际项目：
    #   structured_llm = llm.with_structured_output(EmailClassification)
    #   classification = structured_llm.invoke(classification_prompt)
    classification = _simulate_llm_classify(state["email_content"])

    # --- 根据分类结果决定路由 ---
    intent = classification["intent"]
    urgency = classification["urgency"]

    # 咨询或功能请求 → 先搜索知识库，再起草回复
    if intent in ("question", "feature"):
        goto = "search_documentation"
    # 故障报告 → 先创建工单，再起草回复
    elif intent == "bug":
        goto = "bug_tracking"
    # 账单/紧急/复杂 → 直接起草回复（然后人工审核）
    else:
        goto = "draft_response"

    # --- 记录日志 ---
    current_messages = state.get("messages") or []
    log_msg = (
        f"[classify_intent] 分类结果: intent={intent}, urgency={urgency}, "
        f"topic={classification['topic']} → 路由到 [{goto}]"
    )

    # 返回 Command：同时更新状态和指定下一个节点
    return Command(
        update={
            "classification": classification,
            "messages": current_messages + [log_msg],
        },
        goto=goto,
    )


def search_documentation(
    state: EmailAgentState,
) -> Command[Literal["draft_response"]]:
    """节点 3：搜索知识库获取相关信息。

    这是一个"数据步骤"节点 —— 从外部数据源获取信息。
    包含错误处理模式：如果搜索失败，用错误信息替代结果，
    让后续节点能优雅降级。

    路由：搜索完成后总是去 draft_response（起草回复）。

    错误处理策略（来自 Thinking in LangGraph）：
    - 瞬态错误（网络、限流）→ 重试（通过 RetryPolicy 配置，见 graph.py）
    - 可恢复错误 → 存入状态，让 LLM 在后续步骤中处理
    """
    classification = state.get("classification") or {}
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    # --- 模拟知识库搜索 ---
    # 实际项目：
    #   try:
    #       results = vector_store.similarity_search(query, k=3)
    #       search_results = [doc.page_content for doc in results]
    #   except SearchAPIError as e:
    #       search_results = [f"搜索暂时不可用: {e}"]
    try:
        search_results = _simulate_doc_search(query)
    except Exception as e:
        # 错误降级：记录错误，让后续节点能继续工作
        search_results = [f"搜索暂时不可用: {e}"]

    # --- 模拟客户历史查询 ---
    customer_history = _simulate_customer_history(state["sender_email"])

    # --- 记录日志 ---
    current_messages = state.get("messages") or []
    log_msg = (
        f"[search_documentation] 搜索 '{query}'，"
        f"找到 {len(search_results)} 条结果"
    )

    return Command(
        update={
            "search_results": search_results,
            "customer_history": customer_history,
            "messages": current_messages + [log_msg],
        },
        goto="draft_response",
    )


def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """节点 4：创建/更新 Bug 追踪工单。

    这是一个"动作步骤"节点 —— 执行外部操作（创建工单）。
    在实际项目中，这会调用 Jira、Linear 等工单系统的 API。

    包含 Saga 补偿模式的基础：
    - 工单创建成功 → 记录工单号，继续到 draft_response
    - 工单创建失败 → 记录错误，仍然继续（降级处理）
    """
    classification = state.get("classification") or {}

    # --- 模拟工单创建 ---
    # 实际项目：
    #   try:
    #       ticket = jira_client.create_issue(...)
    #       ticket_id = ticket.key
    #   except JiraError as e:
    #       ticket_id = f"FAILED: {e}"
    ticket_id = f"BUG-{random.randint(1000, 9999)}"

    # --- 记录日志 ---
    current_messages = state.get("messages") or []
    log_msg = (
        f"[bug_tracking] 创建工单 {ticket_id}，"
        f"主题: {classification.get('topic', '未知')}"
    )

    return Command(
        update={
            "messages": current_messages + [log_msg],
        },
        goto="draft_response",
    )


def draft_response(state: EmailAgentState) -> dict:
    """节点 5：使用 LLM 生成回复草稿。

    这是一个"LLM 步骤"节点 —— 调用大语言模型生成文本。

    设计要点：
    - 提示词模板在节点内部组装（不在状态中）
    - 从状态中获取原始数据，按需格式化
    - 不同节点可以用不同方式格式化同一份数据
    """
    classification = state.get("classification") or {
        "intent": "unknown",
        "topic": "未知",
        "summary": "无法分类",
    }
    search_results = state.get("search_results")
    customer_history = state.get("customer_history")

    # --- 模拟 LLM 调用生成回复 ---
    # 实际项目：
    #   prompt = f"""
    #   你是一个专业的客服代表。根据以下信息生成回复：
    #   邮件内容: {state['email_content']}
    #   分类: {classification}
    #   搜索结果: {search_results}
    #   客户信息: {customer_history}
    #   """
    #   response = llm.invoke(prompt)
    #   draft = response.content
    draft = _simulate_llm_draft(
        state["email_content"],
        classification,
        search_results,
        customer_history,
    )

    # --- 记录日志 ---
    current_messages = state.get("messages") or []
    log_msg = f"[draft_response] 生成回复草稿，长度 {len(draft)} 字符"

    return {
        "draft_response": draft,
        "messages": current_messages + [log_msg],
    }


def human_review(
    state: EmailAgentState,
) -> Command[Literal["send_reply", "__end__"]]:
    """节点 6：人工审核 — 使用 interrupt() 暂停执行。

    这是 LangGraph 最独特的功能之一 —— interrupt()。
    当执行到 interrupt() 时：
    1. 图的执行立即暂停
    2. 当前状态被保存到 checkpointer（内存或数据库）
    3. interrupt() 的参数被返回给调用者（作为"问题"）
    4. 调用者可以随时用 Command(resume=...) 恢复执行
    5. resume 的值成为 interrupt() 的返回值

    这意味着：
    - 图可以暂停数天甚至数周
    - 换一台机器也能恢复（如果 checkpointer 持久化）
    - 人工审核完全异步，不阻塞系统

    路由逻辑：
    - approved=True  → send_reply（发送回复）
    - approved=False → __end__（结束，不发送）
    """
    # --- 构造审核请求 ---
    # interrupt() 的参数会返回给调用者，告诉他们需要做什么
    classification = state.get("classification") or {}
    human_input = interrupt({
        "email_id": state.get("email_id", ""),
        "original_email": state.get("email_content", ""),
        "draft_response": state.get("draft_response", ""),
        "urgency": classification.get("urgency", "unknown"),
        "intent": classification.get("intent", "unknown"),
        "action": "请审核并批准/修改此回复",
    })

    # --- 处理人工决策 ---
    # human_input 是调用者通过 Command(resume=...) 传入的值
    approved = human_input.get("approved", False)

    # 如果审核者修改了回复，使用修改后的版本
    final_response = human_input.get(
        "edited_response",
        state.get("draft_response", ""),
    )

    # --- 记录日志 ---
    current_messages = state.get("messages") or []
    feedback = human_input.get("feedback", "")
    log_msg = (
        f"[human_review] 审核结果: {'批准' if approved else '拒绝'}"
        + (f"，反馈: {feedback}" if feedback else "")
    )

    # 根据审核结果路由
    goto = "send_reply" if approved else "__end__"

    return Command(
        update={
            "draft_response": final_response,
            "messages": current_messages + [log_msg],
        },
        goto=goto,
    )


def send_reply(state: EmailAgentState) -> dict:
    """节点 7：发送回复邮件。

    这是图的最终处理节点（对于需要发送的流程）。
    执行实际的邮件发送操作。

    设计要点：
    - 这是"动作步骤"节点
    - 发送失败应该有重试策略（通过 graph.py 中的 RetryPolicy 配置）
    """
    # --- 模拟邮件发送 ---
    # 实际项目：
    #   email_client.send(
    #       to=state["sender_email"],
    #       subject=f"Re: {state['email_id']}",
    #       body=state["draft_response"],
    #   )

    # --- 记录日志 ---
    current_messages = state.get("messages") or []
    log_msg = (
        f"[send_reply] 回复已发送至 {state['sender_email']}，"
        f"内容长度 {len(state.get('draft_response', ''))} 字符"
    )

    return {
        "messages": current_messages + [log_msg],
    }
