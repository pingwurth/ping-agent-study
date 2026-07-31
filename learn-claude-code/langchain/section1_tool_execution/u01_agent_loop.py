"""
U01 - Agent Loop（智能体循环）
=============================
本文件演示 Agent 最核心的运行机制：**思考—行动—观察** 循环。
使用 LangGraph 实现，展示两种方式：
  1. 高层 API：`create_react_agent` 一行代码创建 Agent
  2. 底层 API：手动用 `StateGraph` 构建循环，理解内部机制

核心概念：
  1. Agent 不是一次性问答，而是自主循环直到任务完成
  2. 每轮循环：模型"思考"→ 输出工具调用 → 执行工具 → 将结果反馈给模型
  3. 当模型不再请求工具调用时，循环结束，输出最终回答

依赖：
  - langchain-anthropic: LangChain 的 Anthropic 模型封装
  - langgraph: 图结构的 Agent 编排框架
  - python-dotenv: 加载 .env 文件中的环境变量

.env 配置项：
  - ANTHROPIC_API_KEY: API 密钥（必需）
  - MODEL_ID: 模型 ID（必需）
  - ANTHROPIC_BASE_URL: 自定义代理地址（可选）
"""

import os
import subprocess

# ── readline 中文修复 ──────────────────────────────────────
# macOS 默认使用 libedit 而非 GNU readline，在处理中文输入时
# 退格键（Backspace）会出错。以下四行配置可修复该问题。
try:
    import readline

    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    # Windows 或精简环境可能没有 readline，忽略即可
    pass

# ── 初始化 LangChain 模型 ──────────────────────────────────
# ChatAnthropic 是 LangChain 对 Anthropic API 的封装
# 它自动处理消息格式转换、工具绑定、流式输出等
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# 获取模型实例
model = get_model()

# ── System Prompt（系统提示词）─────────────────────────────
# 告诉模型它的角色和行为规范
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


# ── 工具定义（Tool Definition）────────────────────────────
# LangChain 使用 @tool 装饰器定义工具
# 装饰器会自动从函数签名和 docstring 生成 JSON Schema
# 无需手动编写 input_schema
@tool
def bash(command: str) -> str:
    """Run a shell command.

    Args:
        command: 要执行的 shell 命令
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "拒绝执行 potentially dangerous command"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    return (result.stdout + result.stderr).strip() or "(no output)"


# ── 方式一：高层 API（create_react_agent）─────────────────
# create_react_agent 是 LangGraph 提供的开箱即用的 ReAct Agent
# 它自动实现了 "思考-行动-观察" 循环：
#
#   ┌─────────────────────────────────────────────────┐
#   │  create_react_agent 内部流程：                    │
#   │  ① 将用户消息 + 工具定义发送给模型               │
#   │  ② 如果模型返回 tool_calls → 执行工具            │
#   │  ③ 将工具结果作为 ToolMessage 追加到消息          │
#   │  ④ 重复直到模型不再调用工具 → 返回最终回答        │
#   └─────────────────────────────────────────────────┘
from langgraph.prebuilt import create_react_agent


def run_with_high_level_api(query: str) -> str:
    """
    使用高层 API 运行 Agent。

    create_react_agent 内部构建了一个 LangGraph StateGraph：
      - 节点 "agent": 调用模型
      - 节点 "tools": 执行工具
      - 条件边: 模型有 tool_calls → "tools", 否则 → END
    """
    agent = create_react_agent(
        model,
        [bash],
        prompt=SYSTEM,  # 注入 system prompt
    )

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    # 提取最后一条 AI 消息的文本
    last_message = result["messages"][-1]
    return last_message.content


# ── 方式二：底层 API（StateGraph 手动构建）────────────────
# 这里手动用 StateGraph 构建与 create_react_agent 相同的逻辑
# 帮助理解 Agent 循环的内部机制
from langgraph.graph import StateGraph, MessagesState, START, END


def call_model(state: MessagesState):
    """
    节点：调用模型。

    模型接收当前所有消息（含 system prompt、用户消息、
    AI 回复、工具结果），决定下一步行动。
    """
    # 将 system prompt 注入为 SystemMessage
    from langchain_core.messages import SystemMessage
    messages = [SystemMessage(content=SYSTEM)] + state["messages"]

    # 调用模型，绑定工具
    # bind_tools 让模型知道有哪些工具可用
    response = model.bind_tools([bash]).invoke(messages)
    return {"messages": [response]}


def tool_node(state: MessagesState):
    """
    节点：执行工具调用。

    遍历模型返回的 tool_calls，逐个执行并收集结果。
    每个结果封装为 ToolMessage 返回。
    """
    from langchain_core.messages import ToolMessage

    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        # 执行工具
        if tool_call["name"] == "bash":
            output = bash.invoke(tool_call["args"])
        else:
            output = f"Unknown tool: {tool_call['name']}"

        # 封装为 ToolMessage（包含 tool_call_id 用于匹配）
        results.append(ToolMessage(
            content=output,
            tool_call_id=tool_call["id"],
        ))

    return {"messages": results}


def should_continue(state: MessagesState) -> str:
    """
    条件边：判断是否继续循环。

    - 如果模型最后一条消息包含 tool_calls → 需要执行工具 → "tools"
    - 否则 → 模型已给出最终回答 → END
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


def build_agent_graph():
    """
    手动构建 Agent 的 LangGraph 状态图。

    图结构：
      ┌────────┐     tool_calls      ┌────────┐
      │ agent  │ ──────────────────→ │ tools  │
      │(call   │ ←────────────────── │(execute│
      │ model) │     tool results    │ tools) │
      └────────┘                     └────────┘
           │                              │
           │ end_turn                     │
           ↓                              │
          END ←───────────────────────────┘
    """
    graph = StateGraph(MessagesState)

    # 添加节点
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    # 入口：从 agent 开始
    graph.add_edge(START, "agent")

    # 条件边：agent → tools 或 END
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END,
    })

    # tools → agent（执行完工具后回到模型）
    graph.add_edge("tools", "agent")

    # 编译为可执行的 Runnable
    return graph.compile()


def run_with_low_level_api(query: str) -> str:
    """
    使用底层 API 运行 Agent。

    与高层 API 效果完全相同，但你能看到每一步的细节。
    """
    agent = build_agent_graph()

    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    last_message = result["messages"][-1]
    return last_message.content


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"PID: {os.getpid()}")

    # 选择模式
    print("选择 Agent 实现方式：")
    print("  1. 高层 API（create_react_agent）")
    print("  2. 底层 API（StateGraph 手动构建）")
    mode = input("请选择 [1/2]（默认 1）: ").strip() or "1"

    run_fn = run_with_high_level_api if mode == "1" else run_with_low_level_api
    mode_name = "高层 API" if mode == "1" else "底层 API"
    print(f"\n使用 {mode_name} 模式。输入问题，回车发送。输入 q 退出。\n")

    while True:
        try:
            query = input("\033[036ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit"):
            print("bye!")
            break

        try:
            response = run_fn(query)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print()
