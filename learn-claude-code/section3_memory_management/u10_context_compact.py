"""
U10 - Context Compaction（上下文压缩）
======================================
本文件演示 **上下文压缩** 机制：如何管理 Agent 的上下文窗口。
使用 Anthropic SDK 直接调用 Claude API。

核心概念：
  1. 上下文窗口是有限的（Claude 约 200K tokens）
  2. 长对话会填满上下文窗口，导致 API 调用失败
  3. 上下文压缩通过裁剪、摘要等方式保留关键信息

压缩策略：
  ┌──────────────────────────────────────────────────────────┐
  │  ① 摘要压缩：将早期对话总结为摘要                        │
  │  ② 截断工具结果：截断过长的工具输出                      │
  │  ③ 丢弃中间消息：保留首尾，丢弃中间的工具调用细节        │
  │  ④ 手动触发：用户可以手动请求压缩（/compact）            │
  └──────────────────────────────────────────────────────────┘

Anthropic SDK 消息格式（dict）：
  {
    "role": "user" | "assistant",
    "content": "文本内容" | [{"type": "text", "text": "..."}]
  }
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ── 初始化 Anthropic 客户端 ────────────────────────────────
client, MODEL = create_client()

# ── 上下文窗口配置 ──────────────────────────────────────────
MAX_CONTEXT_TOKENS = 200000   # Claude 的上下文窗口大小（200K tokens）
SYSTEM_PROMPT_TOKENS = 8000   # 系统提示词的预估 token 开销
TOOLS_TOKENS = 2000           # 工具定义的预估 token 开销（JSON Schema）
RESERVED_TOKENS = 20000       # 为模型思考和输出预留的空间

# 实际可用于对话历史的 token 数 = 总窗口 - 系统提示 - 工具定义 - 预留
AVAILABLE_HISTORY_TOKENS = (
    MAX_CONTEXT_TOKENS - SYSTEM_PROMPT_TOKENS
    - TOOLS_TOKENS - RESERVED_TOKENS
)

# 压缩阈值：当历史 token 超过可用空间的 80% 时触发压缩
COMPACTION_THRESHOLD = int(AVAILABLE_HISTORY_TOKENS * 0.8)


# ── Token 估算 ─────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """
    粗略估算文本的 token 数量。

    简化实现：按字符数 / 3 估算。
    - 英文约 4 字符/token
    - 中文约 1.5 字符/token
    - 取折中值 len(text) // 3

    实际项目中应使用 Anthropic 的 tokenizer 或 tiktoken。
    """
    return len(text) // 3


def estimate_messages_tokens(messages: list[dict]) -> int:
    """
    估算消息列表的总 token 数。

    Anthropic SDK 消息格式：
      - content 可以是 str（纯文本）
      - 也可以是 list（多模态内容：text, image, tool_use 等）

    Args:
        messages: Anthropic 格式的消息列表 [{"role": ..., "content": ...}]

    Returns:
        int: 估算的总 token 数
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            # 纯文本内容
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # 多模态内容：遍历每个 block
            for block in content:
                if isinstance(block, dict):
                    # text block: {"type": "text", "text": "..."}
                    # tool_use block: {"type": "tool_use", "input": {...}}
                    text = block.get("text", "") or str(block.get("input", ""))
                    total += estimate_tokens(text)
        # 每条消息有约 10 token 的结构开销（role, metadata 等）
        total += 10
    return total


# ── 上下文压缩器 ───────────────────────────────────────────
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
        初始化压缩器。

        Args:
            keep_recent: 保留最近的多少条消息不压缩（默认 10）
        """
        self.keep_recent = keep_recent
        self.compaction_count = 0  # 已执行的压缩次数

    def should_compact(self, messages: list[dict]) -> bool:
        """
        判断是否需要压缩。

        检查两个条件：
          - 消息数量 >= 20 条（太少不值得压缩）
          - 估算 token 数 > 压缩阈值

        Args:
            messages: Anthropic 格式的消息列表

        Returns:
            bool: True 表示需要压缩
        """
        # 条件1：消息太少，不值得压缩
        if len(messages) < 20:
            return False

        # 条件2：token 数超过阈值
        tokens = estimate_messages_tokens(messages)
        return tokens > COMPACTION_THRESHOLD

    def compact(self, messages: list[dict]) -> list[dict]:
        """
        执行上下文压缩。

        压缩策略：
          ① 分离早期消息和最近消息
          ② 将早期消息总结为一条摘要消息
          ③ 保留最近消息的完整内容

        Args:
            messages: 原始消息列表（Anthropic dict 格式）

        Returns:
            list[dict]: 压缩后的消息列表
        """
        if len(messages) <= self.keep_recent:
            return messages

        # 分离：早期消息（待压缩）和最近消息（保留）
        early_messages = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]

        # 生成早期消息的摘要
        summary = self._summarize_messages(early_messages)

        # 组装压缩后的消息列表
        # 摘要作为一条 user 消息插入开头
        compacted = [
            {
                "role": "user",
                "content": f"[上下文压缩摘要 - 第 {self.compaction_count + 1} 次压缩]\n\n{summary}"
            }
        ]
        compacted.extend(recent_messages)

        self.compaction_count += 1
        return compacted

    def _summarize_messages(self, messages: list[dict]) -> str:
        """
        将消息列表总结为摘要。

        简化实现：提取关键信息（用户请求、助手回复、工具调用）。
        Claude Code 实际使用 LLM 来生成高质量摘要。

        Args:
            messages: 待总结的消息列表

        Returns:
            str: 摘要文本
        """
        key_points = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # 处理 content 为 list 的情况
            if isinstance(content, list):
                # 提取所有 text block 的文本
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = " ".join(text_parts)

            if role == "user":
                # 用户请求：截取前 200 字符
                if len(content) > 200:
                    key_points.append(f"用户请求: {content[:200]}...")
                else:
                    key_points.append(f"用户请求: {content}")

            elif role == "assistant":
                # 助手回复：检查是否有工具调用
                if isinstance(msg.get("content"), list):
                    # 多模态内容：检查 tool_use block
                    tool_names = []
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_names.append(block.get("name", "unknown"))
                    if tool_names:
                        key_points.append(f"Agent: 调用了工具 {', '.join(tool_names)}")
                    else:
                        text = content[:100] if len(content) > 100 else content
                        key_points.append(f"Agent 回复: {text}")
                elif content:
                    text = content[:100] if len(content) > 100 else content
                    key_points.append(f"Agent 回复: {text}")

        # 只保留最近 10 个要点，避免摘要本身过长
        return "\n".join(key_points[-10:])

    def truncate_tool_result(self, content: str, max_length: int = 5000) -> str:
        """
        截断过长的工具输出。

        截断策略：保留头尾，中间用省略号代替。
        这样既保留了输出的开头（通常是重要信息），
        也保留了结尾（可能是总结或错误信息）。

        Args:
            content: 原始工具输出
            max_length: 最大允许长度（默认 5000 字符）

        Returns:
            str: 截断后的文本
        """
        if len(content) <= max_length:
            return content

        half = max_length // 2
        return (
            content[:half]
            + f"\n\n... [截断 {len(content) - max_length} 字符] ...\n\n"
            + content[-half:]
        )


# ── 程序入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  上下文压缩演示（Anthropic SDK 版）")
    print("=" * 60)
    print()

    # 创建压缩器实例
    compactor = ContextCompactor(keep_recent=5)

    # 模拟一个长对话（使用 Anthropic dict 格式）
    # Anthropic 消息格式：{"role": "user"|"assistant", "content": "..."}
    messages = []
    for i in range(30):
        messages.append({"role": "user", "content": f"问题 {i+1}: 请帮我处理这个任务"})
        messages.append({"role": "assistant", "content": f"回答 {i+1}: 已完成"})

    total_tokens = estimate_messages_tokens(messages)
    print(f"原始消息数: {len(messages)}")
    print(f"估算 token 数: {total_tokens}")
    print(f"压缩阈值: {COMPACTION_THRESHOLD} tokens")
    print(f"是否需要压缩: {compactor.should_compact(messages)}")
    print()

    # 执行压缩
    if compactor.should_compact(messages):
        compacted = compactor.compact(messages)
        compacted_tokens = estimate_messages_tokens(compacted)
        print(f"压缩后消息数: {len(compacted)}")
        print(f"压缩后 token 数: {compacted_tokens}")
        print(f"压缩率: {(1 - compacted_tokens / total_tokens) * 100:.1f}%")
        print(f"压缩次数: {compactor.compaction_count}")
        print()
        print(f"压缩后的第一条消息（摘要）:")
        print(compacted[0]["content"][:500])

    # 演示工具输出截断
    print()
    print("-" * 60)
    print("工具输出截断演示")
    print("-" * 60)
    long_output = "A" * 10000  # 模拟 10000 字符的工具输出
    truncated = compactor.truncate_tool_result(long_output, max_length=2000)
    print(f"原始长度: {len(long_output)}")
    print(f"截断后长度: {len(truncated)}")
    print(f"截断后预览: {truncated[:100]}...")
