"""
U04 - Hooks（钩子系统）
========================
本文件演示 **Hooks（钩子）** 机制：在工具执行的特定时机自动运行代码。
使用 LangChain Callbacks 和 LangGraph 节点实现。

核心概念：
  1. Hooks 是在工具执行生命周期中自动触发的脚本
  2. 三种类型：PreToolUse（执行前）、PostToolUse（执行后）、Stop（会话结束）
  3. 可用于自动格式化、安全检查、日志记录等

LangChain 的 Callback 机制：
  ┌──────────────────────────────────────────────────────────┐
  │  LangChain 提供 BaseCallbackHandler 接口：                │
  │    on_tool_start()  → PreToolUse                         │
  │    on_tool_end()    → PostToolUse                        │
  │    on_chain_end()   → Stop                               │
  │                                                          │
  │  Callbacks 在工具执行的生命周期中自动触发                   │
  │  无需修改工具代码，通过配置即可注入                         │
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
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import StateGraph, MessagesState, START, END

model = get_model()


# ── Hook 类型定义 ─────────────────────────────────────────
class HookType:
    """Hook 类型常量"""
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"


# ── Hook 管理器 ───────────────────────────────────────────
class HookManager:
    """
    管理和执行 Hooks 的核心类。

    Hook 配置格式：
      hooks.register("PreToolUse", "bash", "echo 'about to run bash'")
      hooks.register("PostToolUse", "bash", "echo 'bash completed'")

    matcher: 匹配工具名称的模式（支持 | 分隔多个工具名）
    command: 要执行的 shell 命令
    """

    def __init__(self):
        self.hooks: dict[str, list[dict]] = {
            HookType.PRE_TOOL_USE: [],
            HookType.POST_TOOL_USE: [],
            HookType.STOP: [],
        }
        self.log: list[dict] = []

    def register(self, hook_type: str, matcher: str, command: str):
        """注册一个 Hook。"""
        self.hooks[hook_type].append({
            "matcher": matcher,
            "command": command,
        })

    def _matches(self, matcher: str, tool_name: str) -> bool:
        """检查工具名称是否匹配 matcher 模式。"""
        patterns = matcher.split("|")
        return tool_name in patterns

    def run_hooks(self, hook_type: str, tool_name: str = "",
                  file_path: str = "", **kwargs) -> dict:
        """
        执行指定类型的所有匹配 Hooks。

        环境变量（供 Hook 脚本使用）：
          - HOOK_TYPE: Hook 类型
          - TOOL_NAME: 工具名称
          - FILE_PATH: 操作的文件路径

        Returns:
            dict: {"allowed": bool, "output": str}
        """
        results = {"allowed": True, "output": ""}

        for hook in self.hooks.get(hook_type, []):
            if not self._matches(hook["matcher"], tool_name):
                continue

            env = os.environ.copy()
            env["HOOK_TYPE"] = hook_type
            env["TOOL_NAME"] = tool_name
            env["FILE_PATH"] = file_path

            try:
                result = subprocess.run(
                    hook["command"],
                    shell=True, capture_output=True, text=True,
                    env=env, timeout=30,
                )

                log_entry = {
                    "hook_type": hook_type,
                    "tool_name": tool_name,
                    "command": hook["command"],
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timestamp": time.time(),
                }
                self.log.append(log_entry)

                # PreToolUse: 退出码非 0 表示阻止工具执行
                if hook_type == HookType.PRE_TOOL_USE and result.returncode != 0:
                    results["allowed"] = False
                    results["output"] = f"Blocked by hook: {result.stderr}"
                    return results

                results["output"] += result.stdout

            except subprocess.TimeoutExpired:
                results["output"] += f"[Hook timeout: {hook['command']}]\n"
            except Exception as e:
                results["output"] += f"[Hook error: {e}]\n"

        return results


# ── LangChain Callback 实现 Hooks ─────────────────────────
class HookCallbackHandler(BaseCallbackHandler):
    """
    使用 LangChain Callback 机制实现 Hooks。

    LangChain 的 Callback 系统：
      - on_tool_start: 工具执行前触发（对应 PreToolUse）
      - on_tool_end: 工具执行后触发（对应 PostToolUse）
      - on_chain_end: Chain 结束时触发（对应 Stop）

    优点：
      - 无需修改工具代码
      - 通过配置即可注入
      - 支持多个 Callback 叠加
    """

    def __init__(self, hook_manager: HookManager):
        self.hook_manager = hook_manager

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        """PreToolUse Hook：工具执行前触发。"""
        tool_name = serialized.get("name", "unknown")
        file_path = input_str if isinstance(input_str, str) else str(input_str)

        result = self.hook_manager.run_hooks(
            HookType.PRE_TOOL_USE,
            tool_name=tool_name,
            file_path=file_path,
        )

        if not result["allowed"]:
            print(f"\033[033m⚠ Hook 阻止了工具执行: {result['output']}\033[0m")

    def on_tool_end(self, output, *, run_id, **kwargs):
        """PostToolUse Hook：工具执行后触发。"""
        self.hook_manager.run_hooks(HookType.POST_TOOL_USE)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        """Stop Hook：Chain 结束时触发。"""
        self.hook_manager.run_hooks(HookType.STOP)


# ── 工具定义 ──────────────────────────────────────────────
@tool
def bash(command: str) -> str:
    """Run a shell command.

    Args:
        command: 要执行的 shell 命令
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


# ── 带 Hook 的 Agent Graph ────────────────────────────────
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


def call_model(state: MessagesState):
    """节点：调用模型。"""
    messages = [SystemMessage(content=SYSTEM)] + state["messages"]
    response = model.bind_tools([bash]).invoke(messages)
    return {"messages": [response]}


def tool_node_with_hooks(state: MessagesState):
    """
    节点：执行工具（带 Hook 支持）。

    在 LangGraph 中，Hook 可以通过两种方式实现：
      1. Callback（推荐）：通过 config 传入 callback handler
      2. 节点内嵌：在 tool_node 中直接调用 hook_manager

    这里演示节点内嵌方式，更直观。
    """
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        command = tool_call["args"].get("command", "")

        # ① PreToolUse Hook
        pre_result = hook_manager.run_hooks(
            HookType.PRE_TOOL_USE,
            tool_name="bash",
            file_path=command,
        )

        if not pre_result["allowed"]:
            results.append(ToolMessage(
                content=pre_result["output"],
                tool_call_id=tool_call["id"],
            ))
            continue

        # ② 执行工具
        output = bash.invoke(tool_call["args"])

        # ③ PostToolUse Hook
        hook_manager.run_hooks(HookType.POST_TOOL_USE, tool_name="bash")

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


def build_hook_agent():
    """构建带 Hook 支持的 Agent 图。"""
    graph = StateGraph(MessagesState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node_with_hooks)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END,
    })
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Hook 系统 Agent - 输入问题，回车发送。输入 q 退出。\n")

    # 初始化 Hook 管理器并注册示例 Hooks
    hook_manager = HookManager()

    # PreToolUse Hook：阻止删除根目录
    hook_manager.register(
        HookType.PRE_TOOL_USE,
        matcher="bash",
        command='echo "$TOOL_NAME" | grep -q "rm -rf /" && exit 1 || exit 0',
    )

    # PostToolUse Hook：记录所有工具调用
    hook_manager.register(
        HookType.POST_TOOL_USE,
        matcher="bash",
        command='echo "[$(date)] Tool used: $TOOL_NAME" >> /tmp/agent_hook.log',
    )

    agent = build_hook_agent()

    while True:
        try:
            query = input("\033[036mu04 >> \033[0m")
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
