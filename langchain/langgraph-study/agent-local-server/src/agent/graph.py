"""LangGraph 邮件客服 Agent — 完整的图定义。

本文件展示了如何用 LangGraph 构建一个多步骤、支持人工审核的邮件处理 Agent。

核心概念（来自 Thinking in LangGraph）：
1. State（状态）：所有节点共享的"内存"，定义在 models.py 中
2. Nodes（节点）：每个节点做一件事，定义在 nodes.py 中
3. Edges（边）：连接节点，分为静态边和动态路由
4. Graph（图）：用 StateGraph 组装节点和边，编译为可执行的图

图的结构（所有路径都经过 draft_response → human_review）：

    START
      ↓
  read_email（读取邮件）
      ↓
  classify_intent（分类意图）──┐ 动态路由（Command.goto）
      ├─→ search_documentation ─┤
      ├─→ bug_tracking ─────────┤
      └─→ draft_response ←─────┘
           ↓
      human_review（人工审核，interrupt 暂停）
           ├─→ send_reply（审核通过）→ END
           └─→ END（审核拒绝）
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agent.models import EmailAgentState
from agent.nodes import (
    bug_tracking,
    classify_intent,
    draft_response,
    human_review,
    read_email,
    search_documentation,
    send_reply,
)


def build_graph() -> StateGraph:
    """构建并返回邮件客服 Agent 的图。

    构建步骤：
    1. 创建 StateGraph，指定状态类型
    2. 添加所有节点（可选：配置重试策略和错误处理）
    3. 添加静态边（总是成立的连接）
    4. 编译图（可选：配置 checkpointer）

    关于静态边 vs 动态路由：
    - 静态边：用 add_edge() 定义，表示"总是从 A 到 B"
    - 动态路由：在节点内部用 Command(goto=...) 决定，表示"根据条件从 A 到 B 或 C"
    - 本图中，classify_intent 和 human_review 使用动态路由
      其他节点使用静态边

    关于 RetryPolicy：
    - 为可能遇到瞬态错误的节点配置自动重试
    - 搜索和邮件发送都可能因网络问题失败
    - max_attempts: 最大重试次数
    - retry_on: 指定哪些异常触发重试

    关于 Checkpointer：
    - langgraph dev 环境会自动提供 checkpointer（无需手动配置）
    - 生产环境通过 LangSmith Deployment 自动管理持久化
    - checkpointer 是 interrupt()（人工审核）工作的前提条件
    """
    # --- 步骤 1：创建 StateGraph ---
    # EmailAgentState 定义了图的输入/输出 schema
    workflow = StateGraph(EmailAgentState)

    # --- 步骤 2：添加节点 ---
    # 每个节点是一个函数，接收状态并返回更新

    # read_email：读取邮件，无特殊配置
    workflow.add_node("read_email", read_email)

    # classify_intent：LLM 分类，无特殊配置
    # （注意：这个节点返回 Command，内部处理路由）
    workflow.add_node("classify_intent", classify_intent)

    # search_documentation：搜索知识库，配置重试策略
    # 搜索可能因网络问题失败，自动重试最多 3 次
    workflow.add_node(
        "search_documentation",
        search_documentation,
        retry_policy=RetryPolicy(max_attempts=3),
    )

    # bug_tracking：创建工单，配置重试策略
    workflow.add_node(
        "bug_tracking",
        bug_tracking,
        retry_policy=RetryPolicy(max_attempts=2),
    )

    # draft_response：生成回复草稿
    workflow.add_node("draft_response", draft_response)

    # human_review：人工审核（使用 interrupt()）
    workflow.add_node("human_review", human_review)

    # send_reply：发送回复，配置重试策略
    workflow.add_node(
        "send_reply",
        send_reply,
        retry_policy=RetryPolicy(max_attempts=3),
    )

    # --- 步骤 3：添加静态边 ---
    # 静态边表示"总是"从一个节点到另一个节点

    # START → read_email：图总是从读取邮件开始
    workflow.add_edge(START, "read_email")

    # read_email → classify_intent：读取后总是分类
    workflow.add_edge("read_email", "classify_intent")

    # draft_response → human_review：起草后总是送审
    workflow.add_edge("draft_response", "human_review")

    # send_reply → END：发送后总是结束
    workflow.add_edge("send_reply", END)

    # 注意：以下路由是动态的，在节点内部通过 Command(goto=...) 实现：
    # - classify_intent → search_documentation / bug_tracking / human_review / draft_response
    # - human_review → send_reply / END
    # 所以这里不需要为它们调用 add_conditional_edges()

    return workflow


# ============================================================================
# 编译图并导出
# ============================================================================

# 构建图
workflow = build_graph()

# 编译图
# - langgraph dev 会自动提供 checkpointer（持久化由平台管理）
# - interrupt() 依赖 checkpointer，但 langgraph dev 环境会自动注入
# - 编译后的图是一个 Pregel 对象，可以调用 invoke/stream/stream_events
graph = workflow.compile()
