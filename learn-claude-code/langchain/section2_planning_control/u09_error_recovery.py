"""
U09 - Error Recovery（错误恢复）
=================================
本文件演示 **错误恢复** 机制：Agent 如何处理和从错误中恢复。
使用 LangGraph 条件边和重试节点实现。

核心概念：
  1. Agent 在执行过程中会遇到各种错误
  2. 错误恢复能力是 Agent 智能的重要体现
  3. 使用 LangGraph 的条件边实现自动重试和回退

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  在 LangGraph 图中添加错误处理分支：                       │
  │                                                          │
  │  agent → tools → check_error                             │
  │                    ↓ (成功)     ↓ (可重试)                │
  │                   END         agent (重试)                │
  │                    ↓ (不可重试)                           │
  │                   END (告知用户)                          │
  └──────────────────────────────────────────────────────────┘
"""

import os
import time
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END

model = get_model()


# ── 错误分类 ──────────────────────────────────────────────
class ErrorType:
    """错误类型分类"""
    TOOL_ERROR = "tool_error"
    PERMISSION_ERROR = "permission"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


def classify_error(error_msg: str) -> str:
    """
    根据错误信息分类错误类型。

    错误分类的目的：
      - 可恢复的错误 → 重试或调整策略
      - 不可恢复的错误 → 停止并通知用户
      - 速率限制 → 等待后重试
    """
    msg = error_msg.lower()

    if "permission denied" in msg or "access denied" in msg:
        return ErrorType.PERMISSION_ERROR
    if "not found" in msg or "no such file" in msg:
        return ErrorType.NOT_FOUND
    if "rate limit" in msg or "too many requests" in msg:
        return ErrorType.RATE_LIMIT
    if "context" in msg and ("overflow" in msg or "exceed" in msg):
        return ErrorType.CONTEXT_OVERFLOW
    if "timeout" in msg or "connection" in msg:
        return ErrorType.NETWORK_ERROR

    return ErrorType.UNKNOWN


# ── 错误恢复策略 ──────────────────────────────────────────
class ErrorRecoveryStrategy:
    """
    错误恢复策略管理器。

    Claude Code 的恢复策略：
      1. 工具错误 → 将错误信息反馈给模型，让它调整命令
      2. 权限错误 → 提示用户授予权限
      3. 文件不存在 → 搜索相似文件名
      4. 速率限制 → 等待后重试
      5. 上下文溢出 → 压缩上下文
      6. 多次失败 → 向用户求助
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_counts: dict[str, int] = {}

    def should_retry(self, error_type: str, tool_name: str) -> bool:
        """判断是否应该重试。"""
        key = f"{tool_name}:{error_type}"
        count = self.retry_counts.get(key, 0)

        if error_type in (ErrorType.PERMISSION_ERROR, ErrorType.CONTEXT_OVERFLOW):
            return False

        return count < self.max_retries

    def get_recovery_action(self, error_type: str, error_msg: str) -> dict:
        """获取恢复动作。"""
        if error_type == ErrorType.RATE_LIMIT:
            return {"action": "retry", "delay": 5, "message": "Rate limited. Waiting 5 seconds..."}
        if error_type == ErrorType.NETWORK_ERROR:
            return {"action": "retry", "delay": 2, "message": "Network error. Retrying..."}
        if error_type == ErrorType.PERMISSION_ERROR:
            return {"action": "ask_user", "delay": 0, "message": "Permission denied."}
        if error_type == ErrorType.NOT_FOUND:
            return {"action": "fallback", "delay": 0, "message": "Resource not found."}
        if error_type == ErrorType.CONTEXT_OVERFLOW:
            return {"action": "abort", "delay": 0, "message": "Context overflow."}

        return {"action": "retry", "delay": 0, "message": f"Error: {error_msg}"}

    def record_attempt(self, error_type: str, tool_name: str):
        """记录重试次数。"""
        key = f"{tool_name}:{error_type}"
        self.retry_counts[key] = self.retry_counts.get(key, 0) + 1


# ── 工具定义（带错误恢复）─────────────────────────────────
recovery = ErrorRecoveryStrategy(max_retries=3)


@tool
def bash(command: str) -> str:
    """Run a shell command with automatic error recovery.

    Args:
        command: 要执行的 shell 命令
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            return result.stdout

        # 命令执行失败，进入恢复流程
        error_msg = result.stderr or result.stdout or "Command failed"
        error_type = classify_error(error_msg)
        recovery.record_attempt(error_type, "bash")

        action = recovery.get_recovery_action(error_type, error_msg)
        return f"Error: {error_msg}\nSuggestion: {action['message']}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error: {e}"


# ── 带错误恢复的 Agent Graph ──────────────────────────────
SYSTEM = f"""You are a coding agent at {os.getcwd()}.
Use bash to solve tasks. If a command fails, try a different approach.
Act, don't explain."""


def call_model(state: MessagesState):
    """节点：调用模型。"""
    messages = [SystemMessage(content=SYSTEM)] + state["messages"]
    response = model.bind_tools([bash]).invoke(messages)
    return {"messages": [response]}


def tool_node(state: MessagesState):
    """节点：执行工具。"""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        output = bash.invoke(tool_call["args"])
        results.append(ToolMessage(
            content=output,
            tool_call_id=tool_call["id"],
        ))

    return {"messages": results}


def should_continue(state: MessagesState) -> str:
    """
    条件边：判断是否继续。

    LangGraph 的条件边是实现错误恢复的关键：
      - 模型有 tool_calls → 继续执行工具
      - 模型没有 tool_calls → 结束
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


def build_recovery_agent():
    """
    构建带错误恢复的 Agent 图。

    错误恢复的工作原理：
      1. 工具执行失败时，返回错误信息（而不是崩溃）
      2. 错误信息作为 ToolMessage 追加到消息历史
      3. 模型看到错误信息后，会自动调整策略
      4. 模型可能尝试不同的命令或向用户求助

    这种"将错误反馈给模型"的方式是 LangGraph Agent
    最自然的错误恢复机制——模型本身就是错误处理的决策者。
    """
    graph = StateGraph(MessagesState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END,
    })
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("错误恢复 Agent\n")

    # 演示错误分类
    print("── 错误分类示例 ──")
    test_errors = [
        "Permission denied: /etc/passwd",
        "No such file or directory: missing.py",
        "Rate limit exceeded. Please wait.",
        "Context window overflow: message too long",
        "Connection timeout",
        "Something weird happened",
    ]
    for err in test_errors:
        err_type = classify_error(err)
        print(f"  '{err[:40]}...' → {err_type}")

    print("\n── 交互模式 ──")
    print("输入问题，Agent 会自动处理错误。输入 q 退出。\n")

    agent = build_recovery_agent()

    while True:
        try:
            query = input("\033[036mu09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit"):
            print("bye!")
            break

        try:
            result = agent.invoke({"messages": [HumanMessage(content=query)]})
            last_message = result["messages"][-1]
            print(last_message.content)
        except Exception as e:
            print(f"Error: {e}")
        print()
