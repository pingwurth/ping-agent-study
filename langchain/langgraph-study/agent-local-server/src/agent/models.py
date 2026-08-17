"""数据模型定义 — 邮件客服 Agent 的状态和类型。

本模块定义了 LangGraph 图的核心数据结构：
- EmailClassification: LLM 对邮件的分类结果
- HumanDecision: 人工审核的决策结果
- EmailAgentState: 图的状态（所有节点共享的"内存"）

设计原则（来自 Thinking in LangGraph）：
1. 状态只存储原始数据，不存储格式化文本
2. 提示词模板在节点内部组装，不放在状态中
3. 使用 TypedDict 让 LangGraph 能自动推断 schema
"""

from __future__ import annotations

from typing import Literal, TypedDict

# ============================================================================
# 邮件分类结果
# ============================================================================

class EmailClassification(TypedDict):
    """LLM 对邮件的结构化分类输出。

    使用 TypedDict（而非 dataclass）以便 LLM 的 structured output 能直接匹配。
    LangGraph 的状态更新是字典合并，TypedDict 与此天然契合。

    Attributes:
        intent:     邮件意图分类 — question(咨询) / bug(故障) / billing(账单)
                    / feature(功能请求) / complex(复杂问题)
        urgency:    紧急程度 — low / medium / high / critical
        topic:      邮件主题摘要（简短名词短语）
        summary:    一句话总结邮件内容
    """

    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


# ============================================================================
# 人工审核决策
# ============================================================================

class HumanDecision(TypedDict, total=False):
    """人工审核者在 interrupt() 暂停后返回的决策。

    total=False 表示所有字段都是可选的 — 审核者可能只填写部分字段。

    Attributes:
        approved:           是否批准发送回复（True=发送, False=丢弃）
        edited_response:    审核者修改后的回复内容（可选，不填则用原草稿）
        feedback:           审核者的反馈意见（可选，用于改进后续回复质量）
    """

    approved: bool
    edited_response: str
    feedback: str


# ============================================================================
# 图的状态定义
# ============================================================================

class EmailAgentState(TypedDict):
    """邮件客服 Agent 的完整状态。

    这是 LangGraph 中所有节点共享的"内存"。每个节点读取状态、
    执行操作、返回状态更新（字典），LangGraph 自动合并更新。

    设计要点：
    - 只存原始数据，不存格式化字符串（prompt 模板在节点内部组装）
    - 可选字段用 `| None`，节点需要检查后再使用
    - messages 列表用于记录处理过程，方便调试和审计

    字段说明：
        # --- 输入 ---
        email_content:      原始邮件内容
        sender_email:       发件人邮箱地址
        email_id:           邮件唯一标识符

        # --- 分类结果（由 classify_intent 节点填充）---
        classification:     LLM 的结构化分类结果

        # --- 搜索结果（由 search_documentation 节点填充）---
        search_results:     知识库搜索结果列表

        # --- 客户历史（由 search_documentation 节点填充）---
        customer_history:   客户的历史交互记录

        # --- 回复草稿（由 draft_response 节点填充）---
        draft_response:     生成的回复草稿

        # --- 处理日志 ---
        messages:           处理过程中的消息记录，用于调试和审计
    """

    # 输入字段
    email_content: str
    sender_email: str
    email_id: str

    # 分类结果（初始为 None，由 classify_intent 填充）
    classification: EmailClassification | None

    # 搜索结果（初始为 None，由 search_documentation 填充）
    search_results: list[str] | None

    # 客户历史（初始为 None，由 search_documentation 填充）
    customer_history: dict | None

    # 回复草稿（初始为 None，由 draft_response 填充）
    draft_response: str | None

    # 处理日志（记录每一步的操作）
    messages: list[str] | None
