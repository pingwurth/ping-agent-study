"""
U10 - Context Compaction（上下文压缩）
======================================
本文件演示 **上下文压缩** 机制：如何管理 Agent 的上下文窗口。
使用 LangChain 的消息裁剪和摘要功能实现。

核心概念：
  1. 上下文窗口是有限的（Claude 约 200K tokens）
  2. 长对话会填满上下文窗口，导致 API 调用失败
  3. 上下文压缩通过裁剪、摘要等方式保留关键信息

LangChain 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  LangChain 提供多种消息裁剪策略：                         │
  │                                                          │
  │  1. trim_messages: 按 token 数裁剪                       │
  │  2. 消息摘要：用模型总结早期消息                          │
  │  3. 滑动窗口：只保留最近 N 条消息                        │
  │                                                          │
  │  在 LangGraph 中，压缩可以作为一个节点插入图中：           │
  │                                                          │
  │  agent → tools → compact_context → agent                 │
  └──────────────────────────────────────────────────────────┘

压缩策略：
  ① 摘要压缩：将早期对话总结为摘要
  ② 截断工具结果：截断过长的工具输出
  ③ 丢弃中间消息：保留首尾，丢弃中间的工具调用细节
  ④ 手动触发：用户可以手动请求压缩（/compact）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
    trim_messages,
)
from langgraph.graph import StateGraph, MessagesState, START, END

model = get_model()

# ── 上下文窗口配置 ────────────────────────────────────────
MAX_CONTEXT_TOKENS = 200000   # Claude 的上下文窗口大小
SYSTEM_PROMPT_TOKENS = 8000   # 系统提示词的预估 token 开销
TOOLS_TOKENS = 2000           # 工具定义的预估 token 开销
RESERVED_TOKENS = 20000       # 为模型思考和输出预留的空间

AVAILABLE_HISTORY_TOKENS = (
    MAX_CONTEXT_TOKENS - SYSTEM_PROMPT_TOKENS
    - TOOLS_TOKENS - RESERVED_TOKENS
)

COMPACTION_THRESHOLD = int(AVAILABLE_HISTORY_TOKENS * 0.8)


# ── Token 估算 ────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """
    粗略估算文本的 token 数量。

    简化实现：按字符数估算（中文约 1.5 字符/token，英文约 4 字符/token）。
    实际项目中应使用 tiktoken 或模型专用的 tokenizer。
    """
    return len(text) // 3


def count_message_tokens(message) -> int:
    """估算单条消息的 token 数。"""
    content = ""
    if isinstance(message, str):
        content = message
    elif hasattr(message, "content"):
        if isinstance(message.content, str):
            content = message.content
        else:
            content = str(message.content)
    return estimate_tokens(content) + 10  # 每条消息有约 10 token 的开销


# ── 上下文压缩器 ──────────────────────────────────────────
class ContextCompactor:
    """
    上下文压缩管理器。

    Claude Code 的压缩触发方式：
      - 自动触发：当对话历史接近上下文窗口限制时
      - 手动触发：用户输入 /compact 命令
      - 策略触发：在特定操作前（如大规模重构）主动压缩

    压缩算法：
      ① 保留最近的 N 条消息（最近的上下文最重要）
      ② 将早期消息压缩为摘要
      ③ 截断过长的工具输出
      ④ 保留关键的系统信息（如文件路径、错误信息）
    """

    def __init__(self, keep_recent: int = 10):
        """
        Args:
            keep_recent: 保留最近的多少条消息不压缩
        """
        self.keep_recent = keep_recent
        self.compaction_count = 0

    def should_compact(self, messages: list) -> bool:
        """
        判断是否需要压缩。

        检查条件：
          - 对话历史的估算 token 数超过阈值
          - 消息数量足够多（至少 20 条才值得压缩）
        """
        if len(messages) < 20:
            return False

        tokens = sum(count_message_tokens(msg) for msg in messages)
        return tokens > COMPACTION_THRESHOLD

    def compact(self, messages: list) -> list:
        """
        执行上下文压缩。

        压缩策略：
          ① 分离早期消息和最近消息
          ② 将早期消息总结为一条摘要消息
          ③ 保留最近消息的完整内容
          ④ 截断过长的工具输出

        Args:
            messages: 原始消息列表（LangChain Message 对象）

        Returns:
            list: 压缩后的消息列表
        """
        if len(messages) <= self.keep_recent:
            return messages

        # 分离早期消息和最近消息
        early_messages = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]

        # 生成早期消息的摘要
        summary = self._summarize_messages(early_messages)

        # 组装压缩后的消息列表
        compacted = [
            HumanMessage(content=f"[上下文压缩摘要 - 第 {self.compaction_count + 1} 次压缩]\n\n{summary}")
        ]
        compacted.extend(recent_messages)

        self.compaction_count += 1
        return compacted

    def _summarize_messages(self, messages: list) -> str:
        """
        将消息列表总结为摘要。

        简化实现：提取关键信息。
        Claude Code 实际使用 LLM 来生成高质量摘要。
        """
        key_points = []

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if isinstance(msg, HumanMessage):
                if len(content) > 200:
                    key_points.append(f"用户请求: {content[:200]}...")
                else:
                    key_points.append(f"用户请求: {content}")

            elif isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools = [tc["name"] for tc in msg.tool_calls]
                    key_points.append(f"Agent: 调用了工具 {', '.join(tools)}")
                elif content:
                    text = content[:100] if len(content) > 100 else content
                    key_points.append(f"Agent 回复: {text}")

            elif isinstance(msg, ToolMessage):
                key_points.append(f"工具结果: [{len(content)} chars]")

        return "\n".join(key_points[-10:])  # 只保留最近 10 个要点

    def truncate_tool_result(self, content: str, max_length: int = 5000) -> str:
        """
        截断过长的工具输出。

        截断时保留头尾，中间用省略号代替。
        """
        if len(content) <= max_length:
            return content

        half = max_length // 2
        return (
            content[:half]
            + f"\n\n... [截断 {len(content) - max_length} 字符] ...\n\n"
            + content[-half:]
        )


# ── 使用 LangChain 的 trim_messages ──────────────────────
def compact_with_langchain(messages: list, max_tokens: int = 160000) -> list:
    """
    使用 LangChain 的 trim_messages 进行消息裁剪。

    trim_messages 支持多种策略：
      - "last": 保留最后 N 条消息
      - "first": 保留前 N 条消息

    还支持：
      - token_counter: 自定义 token 计算函数
      - start_on: 从哪种消息类型开始
      - include_system: 是否保留 system message
    """
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=count_message_tokens,
        strategy="last",
        start_on="human",  # 从 HumanMessage 开始保留
        include_system=True,  # 保留 SystemMessage
    )


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("上下文压缩演示\n")

    compactor = ContextCompactor(keep_recent=5)

    # 模拟一个长对话（使用 LangChain Message 对象）
    messages = [SystemMessage(content="You are a helpful assistant.")]
    for i in range(30):
        messages.append(HumanMessage(content=f"问题 {i+1}: 请帮我处理这个任务"))
        messages.append(AIMessage(content=f"回答 {i+1}: 已完成"))

    total_tokens = sum(count_message_tokens(msg) for msg in messages)
    print(f"原始消息数: {len(messages)}")
    print(f"估算 token 数: {total_tokens}")
    print(f"压缩阈值: {COMPACTION_THRESHOLD} tokens")

    # 检查是否需要压缩
    print(f"是否需要压缩: {compactor.should_compact(messages)}")

    # 执行压缩
    if compactor.should_compact(messages):
        compacted = compactor.compact(messages)
        compacted_tokens = sum(count_message_tokens(msg) for msg in compacted)
        print(f"\n压缩后消息数: {len(compacted)}")
        print(f"压缩后 token 数: {compacted_tokens}")
        print(f"压缩次数: {compactor.compaction_count}")
        print(f"\n压缩后的第一条消息:")
        print(compacted[0].content[:500])

    # 演示 LangChain trim_messages
    print("\n\n── LangChain trim_messages ──")
    trimmed = compact_with_langchain(messages, max_tokens=500)
    print(f"裁剪后消息数: {len(trimmed)}")
    for msg in trimmed:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"  [{msg.__class__.__name__}] {content[:60]}...")

    # 演示工具输出截断
    print("\n\n── 工具输出截断演示 ──")
    long_output = "A" * 10000
    truncated = compactor.truncate_tool_result(long_output, max_length=2000)
    print(f"原始长度: {len(long_output)}")
    print(f"截断后长度: {len(truncated)}")
