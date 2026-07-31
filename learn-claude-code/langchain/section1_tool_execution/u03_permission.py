"""
U03 - Permission System（权限系统）
===================================
本文件演示 **权限控制** 机制：如何在工具执行前进行安全审查。
使用 LangGraph 的条件节点实现权限拦截。

核心概念：
  1. 不是所有工具调用都应该自动执行
  2. 危险操作（如 rm -rf、写入系统文件）需要用户确认
  3. 权限系统在工具执行前拦截，询问用户是否允许
  4. Claude Code 支持三种权限模式：自动允许、需确认、禁止

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  在 LangGraph 图中插入权限检查节点：                       │
  │                                                          │
  │  agent → permission_check → tools → agent                │
  │                 ↓ (拒绝)                                  │
  │               agent（被告知权限被拒绝）                    │
  └──────────────────────────────────────────────────────────┘
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END

model = get_model()


# ── 危险命令检测 ──────────────────────────────────────────
DANGEROUS_PATTERNS = [
    "rm -rf",
    "rm -f /",
    "sudo rm",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    "chmod 777",
    "curl | bash",
    "wget | bash",
    ":(){:|:&};:",
]


def is_dangerous(command: str) -> bool:
    """检查命令是否为危险命令。"""
    cmd_lower = command.lower().strip()
    return any(pattern in cmd_lower for pattern in DANGEROUS_PATTERNS)


def request_permission(command: str) -> bool:
    """请求用户确认是否允许执行命令。"""
    print(f"\n\033[033m⚠ 权限检查：检测到可能的危险命令\033[0m")
    print(f"\033[033m  命令: {command}\033[0m")

    while True:
        response = input("\033[033m  允许执行？[y/n/a] (y=允许, n=拒绝, a=本会话自动允许): \033[0m").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        elif response == "a":
            print("\033[033m  已设置本会话自动允许此工具\033[0m")
            return True
        print("  请输入 y/n/a")


# ── 工具定义 ──────────────────────────────────────────────
@tool
def bash(command: str) -> str:
    """Run a shell command.

    Args:
        command: 要执行的 shell 命令
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


# ── 带权限检查的 Agent Graph ──────────────────────────────
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


def call_model(state: MessagesState):
    """节点：调用模型。"""
    messages = [SystemMessage(content=SYSTEM)] + state["messages"]
    response = model.bind_tools([bash]).invoke(messages)
    return {"messages": [response]}


def permission_check(state: MessagesState, config: dict):
    """
    节点：权限检查。

    在工具执行前拦截，检查每个 tool_call 是否安全：
      - 安全的命令 → 放行，进入 tool_node
      - 危险的命令 → 请求用户确认
        - 用户允许 → 放行
        - 用户拒绝 → 生成拒绝的 ToolMessage，跳过执行

    这是 Claude Code 的关键安全机制：
      - AI 模型可能生成有害的命令
      - 权限系统给用户最终的控制权
    """
    from langchain_core.messages import AIMessage

    last_message = state["messages"][-1]
    auto_approve = config.get("configurable", {}).get("auto_approve", False)

    # 检查所有 tool_calls
    approved_calls = []
    denied_messages = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "bash":
            command = tool_call["args"].get("command", "")

            if auto_approve or not is_dangerous(command):
                # 自动批准或非危险命令
                approved_calls.append(tool_call)
            else:
                # 危险命令，请求用户确认
                if request_permission(command):
                    approved_calls.append(tool_call)
                else:
                    # 用户拒绝，生成拒绝的 ToolMessage
                    denied_messages.append(ToolMessage(
                        content="Error: Permission denied by user. Try a safer approach.",
                        tool_call_id=tool_call["id"],
                    ))
        else:
            # 非 bash 工具默认放行
            approved_calls.append(tool_call)

    # 如果有被拒绝的调用，直接返回拒绝消息（不执行工具）
    if denied_messages:
        return {"messages": denied_messages}

    return state


def tool_node(state: MessagesState):
    """节点：执行工具调用。"""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "bash":
            output = bash.invoke(tool_call["args"])
        else:
            output = f"Unknown tool: {tool_call['name']}"

        results.append(ToolMessage(
            content=output,
            tool_call_id=tool_call["id"],
        ))

    return {"messages": results}


def should_continue(state: MessagesState) -> str:
    """条件边：判断是否继续。"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


def build_permission_agent():
    """
    构建带权限检查的 Agent 图。

    图结构：
      ┌────────┐              ┌──────────────┐              ┌────────┐
      │ agent  │ ──────────→ │  permission   │ ──────────→ │ tools  │
      │(call   │              │  check        │              │(execute│
      │ model) │              │(安全审查)      │              │ tools) │
      └────────┘              └──────────────┘              └────────┘
           ↑                       │ (拒绝)                      │
           │                       ↓                             │
           │                    agent                            │
           │                       ↑                             │
           └───────────────────────┘←────────────────────────────┘
    """
    graph = StateGraph(MessagesState)

    graph.add_node("agent", call_model)
    graph.add_node("permission", permission_check)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_edge("agent", "permission")

    # 权限检查后：有 tool_calls → tools, 否则 → END
    def after_permission(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "end"

    graph.add_conditional_edges("permission", after_permission, {
        "tools": "tools",
        "end": END,
    })

    graph.add_edge("tools", "agent")

    return graph.compile()


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("权限控制 Agent - 输入问题，回车发送。输入 q 退出。\n")
    print("提示：输入危险命令（如 'rm -rf /'）会触发权限检查\n")

    agent = build_permission_agent()

    while True:
        try:
            query = input("\033[036mu03 >> \033[0m")
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
