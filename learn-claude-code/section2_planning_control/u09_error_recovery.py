"""
U09 - Error Recovery（错误恢复）
=================================
本文件演示 **错误恢复** 机制：Agent 如何处理和从错误中恢复。
使用原生 Anthropic SDK 实现。

核心概念：
  1. Agent 在执行过程中会遇到各种错误
  2. 错误恢复能力是 Agent 智能的重要体现
  3. 不同类型的错误需要不同的恢复策略
  4. 最自然的恢复方式是将错误信息反馈给模型

错误恢复的工作原理：
  ┌──────────────────────────────────────────────────────────┐
  │  当工具执行失败时：                                       │
  │                                                          │
  │  1. 不要崩溃或抛出异常                                    │
  │  2. 将错误信息作为工具结果返回给模型                       │
  │  3. 模型看到错误后，会自动调整策略                         │
  │  4. 模型可能尝试不同的方法，或向用户求助                   │
  │                                                          │
  │  这种"将错误反馈给模型"的方式是最自然的错误恢复机制：      │
  │  模型本身就是错误处理的决策者。                            │
  └──────────────────────────────────────────────────────────┘

Claude Code 的错误恢复策略：
  - 工具错误 → 将错误信息反馈给模型，让它调整命令
  - 权限错误 → 提示用户授予权限
  - 文件不存在 → 搜索相似文件名
  - 速率限制 → 等待后重试
  - 上下文溢出 → 压缩上下文或重启会话
  - 多次失败 → 向用户求助
"""

import os
import sys
import json
import time
import subprocess
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ============================================================
# 初始化客户端
# ============================================================
client, MODEL = create_client()


# ════════════════════════════════════════════════════════════
# 第一部分：错误类型定义
# ════════════════════════════════════════════════════════════

class ErrorType:
    """
    错误类型常量。

    将错误分类的目的是为每种错误类型制定不同的恢复策略：
      - TOOL_ERROR:        工具执行失败（通常是参数错误或命令错误）
      - PERMISSION_ERROR:  权限不足（需要用户授权）
      - NOT_FOUND:         资源不存在（文件、目录、命令等）
      - RATE_LIMIT:        API 速率限制（需要等待后重试）
      - CONTEXT_OVERFLOW:  上下文窗口溢出（需要压缩或重启）
      - NETWORK_ERROR:     网络连接问题（可以重试）
      - UNKNOWN:           未知错误（保守处理）
    """
    TOOL_ERROR = "tool_error"
    PERMISSION_ERROR = "permission"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


def classify_error(error_msg: str) -> str:
    """
    根据错误信息文本分类错误类型。

    分类策略：通过关键词匹配来判断错误类型。
    这是一种简单但有效的方式。
    在实际应用中，可以使用更复杂的规则或 ML 模型。

    Args:
        error_msg: 错误信息文本

    Returns:
        str: 错误类型（ErrorType 的常量之一）
    """
    msg = error_msg.lower()

    # 权限错误：文件系统权限不足
    if "permission denied" in msg or "access denied" in msg:
        return ErrorType.PERMISSION_ERROR

    # 资源不存在：文件、目录、命令未找到
    if "not found" in msg or "no such file" in msg or "command not found" in msg:
        return ErrorType.NOT_FOUND

    # 速率限制：API 调用过于频繁
    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return ErrorType.RATE_LIMIT

    # 上下文溢出：消息太长，超出模型的上下文窗口
    if "context" in msg and ("overflow" in msg or "exceed" in msg or "too long" in msg):
        return ErrorType.CONTEXT_OVERFLOW

    # 网络错误：连接超时、DNS 解析失败等
    if "timeout" in msg or "connection" in msg or "network" in msg:
        return ErrorType.NETWORK_ERROR

    # 工具执行错误：命令返回非零退出码
    if "error" in msg or "failed" in msg:
        return ErrorType.TOOL_ERROR

    # 无法分类的错误
    return ErrorType.UNKNOWN


# ════════════════════════════════════════════════════════════
# 第二部分：ErrorRecoveryStrategy - 错误恢复策略
# ════════════════════════════════════════════════════════════

class ErrorRecoveryStrategy:
    """
    错误恢复策略管理器。

    为每种错误类型定义恢复策略：
      - action: 恢复动作（retry / ask_user / fallback / abort）
      - delay:  重试前的等待时间（秒）
      - message: 给用户的提示信息

    恢复动作说明：
      - retry:     自动重试（适用于临时性错误）
      - ask_user:  请求用户介入（适用于需要授权的错误）
      - fallback:  使用备选方案（适用于资源不存在）
      - abort:     终止操作（适用于不可恢复的错误）

    关键设计：
      - max_retries: 最大重试次数，防止无限重试
      - retry_counts: 记录每种错误的重试次数
    """

    def __init__(self, max_retries: int = 3):
        """
        初始化恢复策略。

        Args:
            max_retries: 最大重试次数
        """
        self.max_retries = max_retries
        # 记录重试次数：key = "工具名:错误类型"
        self.retry_counts: dict[str, int] = {}

    def should_retry(self, error_type: str, tool_name: str) -> bool:
        """
        判断是否应该重试。

        不可重试的错误：
          - PERMISSION_ERROR: 权限问题需要用户介入
          - CONTEXT_OVERFLOW: 上下文溢出需要压缩或重启

        Args:
            error_type: 错误类型
            tool_name:  工具名称

        Returns:
            bool: 是否应该重试
        """
        key = f"{tool_name}:{error_type}"
        count = self.retry_counts.get(key, 0)

        # 这些错误类型不应该重试
        if error_type in (ErrorType.PERMISSION_ERROR, ErrorType.CONTEXT_OVERFLOW):
            return False

        # 检查是否超过最大重试次数
        return count < self.max_retries

    def get_recovery_action(self, error_type: str, error_msg: str) -> dict:
        """
        获取恢复动作。

        根据错误类型返回相应的恢复策略。

        Args:
            error_type: 错误类型
            error_msg:  错误信息

        Returns:
            dict: {"action": str, "delay": int, "message": str}
        """
        if error_type == ErrorType.RATE_LIMIT:
            return {
                "action": "retry",
                "delay": 5,
                "message": "触发速率限制，等待 5 秒后重试..."
            }

        if error_type == ErrorType.NETWORK_ERROR:
            return {
                "action": "retry",
                "delay": 2,
                "message": "网络错误，2 秒后重试..."
            }

        if error_type == ErrorType.PERMISSION_ERROR:
            return {
                "action": "ask_user",
                "delay": 0,
                "message": "权限不足，请授予权限后重试。"
            }

        if error_type == ErrorType.NOT_FOUND:
            return {
                "action": "fallback",
                "delay": 0,
                "message": "资源不存在，请检查路径或名称。"
            }

        if error_type == ErrorType.CONTEXT_OVERFLOW:
            return {
                "action": "abort",
                "delay": 0,
                "message": "上下文窗口溢出，请缩短输入或开启新会话。"
            }

        # 默认：重试
        return {
            "action": "retry",
            "delay": 0,
            "message": f"错误: {error_msg}"
        }

    def record_attempt(self, error_type: str, tool_name: str):
        """
        记录一次重试尝试。

        Args:
            error_type: 错误类型
            tool_name:  工具名称
        """
        key = f"{tool_name}:{error_type}"
        self.retry_counts[key] = self.retry_counts.get(key, 0) + 1


# ════════════════════════════════════════════════════════════
# 第三部分：工具定义（带错误恢复）
# ════════════════════════════════════════════════════════════

# 全局恢复策略实例
recovery = ErrorRecoveryStrategy(max_retries=3)

# Bash 工具的 JSON Schema 定义
BASH_TOOL = {
    "name": "bash",
    "description": "执行 shell 命令。如果命令失败，会自动分析错误并提供恢复建议。",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令"
            }
        },
        "required": ["command"]
    }
}


def execute_bash(command: str) -> str:
    """
    执行 shell 命令，带错误恢复逻辑。

    错误恢复流程：
      1. 执行命令
      2. 如果成功，返回输出
      3. 如果失败，分类错误类型
      4. 根据错误类型获取恢复策略
      5. 记录重试次数
      6. 返回错误信息和恢复建议

    关键设计原则：
      - 不要崩溃：即使命令失败，也要返回有意义的信息
      - 信息丰富：告诉模型发生了什么错误，以及建议怎么做
      - 模型决策：让模型决定是重试、换方法、还是求助用户

    Args:
        command: 要执行的 shell 命令

    Returns:
        str: 命令输出或错误信息（包含恢复建议）
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # 命令成功
            return result.stdout

        # 命令失败：进入恢复流程
        error_msg = result.stderr or result.stdout or "Command failed"
        error_type = classify_error(error_msg)

        # 记录重试次数
        recovery.record_attempt(error_type, "bash")

        # 获取恢复建议
        action = recovery.get_recovery_action(error_type, error_msg)

        # 返回错误信息和建议（这会被发送给模型）
        return (
            f"命令执行失败: {error_msg}\n"
            f"错误类型: {error_type}\n"
            f"建议: {action['message']}"
        )

    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时（超过 30 秒）。请尝试更简单的命令。"
    except Exception as e:
        return f"错误: {e}"


# ════════════════════════════════════════════════════════════
# 第四部分：Agent 循环（带错误恢复）
# ════════════════════════════════════════════════════════════

def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    处理工具调用。

    Args:
        tool_name:  工具名称
        tool_input: 工具输入参数

    Returns:
        str: 工具执行结果
    """
    if tool_name == "bash":
        command = tool_input.get("command", "")
        return execute_bash(command)
    return f"未知工具: {tool_name}"


def run_agent(query: str) -> str:
    """
    运行带错误恢复的 Agent。

    Agent 循环中的错误恢复：
      1. 模型决定调用工具
      2. 执行工具，如果失败则返回错误信息
      3. 将错误信息作为 tool_result 发回给模型
      4. 模型看到错误后，可能：
         a. 尝试不同的命令
         b. 使用不同的工具
         c. 向用户求助
         d. 给出解释并停止

    这种方式的关键洞察：
      模型本身就是最好的错误处理器。
      它能理解错误信息，并根据上下文做出最佳决策。

    Args:
        query: 用户输入

    Returns:
        str: Agent 的最终回答
    """
    messages = [{"role": "user", "content": query}]

    # Agent 循环，最多 10 轮
    for turn in range(10):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=f"""你是一个编程助手，工作在 {os.getcwd()} 目录。
使用 bash 工具执行任务。
如果命令失败，分析错误信息，尝试不同的方法。
用中文回复。""",
            tools=[BASH_TOOL],
            messages=messages,
        )

        # 检查是否需要调用工具
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # 执行工具（可能成功也可能失败）
                    result = handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        # 模型给出了最终回答
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text
        return final_text

    return "达到最大轮次，任务未完成。"


# ════════════════════════════════════════════════════════════
# 第五部分：程序入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  U09 - Error Recovery（错误恢复）演示")
    print("=" * 60)

    # ── 演示 1：错误分类 ──
    print("\n── 演示 1：错误分类 ──\n")
    print("将各种错误信息分类为不同的错误类型：\n")

    test_errors = [
        ("Permission denied: /etc/passwd", "文件权限不足"),
        ("No such file or directory: missing.py", "文件不存在"),
        ("Rate limit exceeded. Please wait.", "API 速率限制"),
        ("Context window overflow: message too long", "上下文溢出"),
        ("Connection timeout after 30s", "网络超时"),
        ("bash: command not found: pythn", "命令拼写错误"),
        ("Something weird happened", "未知错误"),
    ]

    for err_msg, description in test_errors:
        err_type = classify_error(err_msg)
        print(f"  '{err_msg[:45]}...'")
        print(f"    描述: {description}")
        print(f"    类型: {err_type}")
        action = recovery.get_recovery_action(err_type, err_msg)
        print(f"    策略: {action['action']} - {action['message']}")
        print()

    # ── 演示 2：恢复策略 ──
    print("\n── 演示 2：各错误类型的恢复策略 ──\n")

    strategies = [
        (ErrorType.RATE_LIMIT, "速率限制"),
        (ErrorType.NETWORK_ERROR, "网络错误"),
        (ErrorType.PERMISSION_ERROR, "权限不足"),
        (ErrorType.NOT_FOUND, "资源不存在"),
        (ErrorType.CONTEXT_OVERFLOW, "上下文溢出"),
        (ErrorType.TOOL_ERROR, "工具错误"),
        (ErrorType.UNKNOWN, "未知错误"),
    ]

    for err_type, description in strategies:
        action = recovery.get_recovery_action(err_type, "示例错误")
        retry = recovery.should_retry(err_type, "bash")
        print(f"  {description:12} → 动作: {action['action']:12} 可重试: {retry}")

    # ── 演示 3：错误恢复流程图 ──
    print("\n\n── 演示 3：错误恢复流程 ──\n")
    print("""
    Agent 调用工具
         │
         ▼
    ┌─────────────┐
    │  执行工具    │
    └──────┬──────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  成功         失败
     │           │
     │           ▼
     │    ┌──────────────┐
     │    │  分类错误类型  │
     │    └──────┬───────┘
     │           │
     │    ┌──────┴──────────────────────────────┐
     │    │      │      │      │      │         │
     │    ▼      ▼      ▼      ▼      ▼         ▼
     │  RATE   NET    PERM   NOT    CTX       OTHER
     │  LIMIT  ERR    ERR    FOUND  OVER
     │    │      │      │      │      │         │
     │    ▼      ▼      ▼      ▼      ▼         ▼
     │  等待    重试   求助   搜索   终止       重试
     │  5秒    2秒    用户   相似   会话
     │                 ↓     文件
     │              授权      ↓
     │              后重试   备选方案
     │
     ▼
    返回结果给模型
         │
         ▼
    模型决定下一步（可能继续调用工具，或给出最终回答）
    """)

    # ── 交互模式 ──
    print("\n" + "=" * 60)
    print("  交互模式")
    print("  输入任务，Agent 会自动处理错误")
    print("  输入 q 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\033[36mu09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit"):
            print("bye!")
            break

        try:
            # 重置重试计数
            recovery = ErrorRecoveryStrategy(max_retries=3)
            response = run_agent(query)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print()
