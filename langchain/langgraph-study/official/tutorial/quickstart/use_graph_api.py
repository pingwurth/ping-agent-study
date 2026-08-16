# ============================================================
# Step 1: 定义工具 (Tools) 和模型 (Model)
# ============================================================
# 本示例展示如何使用 LangGraph 的 Graph API 构建一个简单的 Agent
# Agent 可以调用工具执行算术运算（加法、乘法、除法）

import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# 加载 .env 文件中的环境变量（API Key、模型名称等）
load_dotenv()

# 初始化语言模型（LLM）
# - model: 模型名称，默认使用 qwen-turbo
# - model_provider: 模型提供商，默认使用 openai 兼容接口
# - base_url: API 基础地址（此处使用阿里云 DashScope）
# - temperature: 温度参数，越低越确定性
model = init_chat_model(
    model=os.getenv("MODEL_NAME", "qwen-turbo"),
    model_provider=os.getenv("MODEL_PROVIDER", "openai"),
    base_url=os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    temperature=0.3,
)


# ----------------------------------------------------------
# 定义工具函数
# 使用 @tool 装饰器将普通函数转换为 LangChain 工具
# 工具的 docstring 非常重要——LLM 会根据它来决定何时调用哪个工具
# ----------------------------------------------------------

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.（计算 a 乘以 b）

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.（计算 a 加 b）

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.（计算 a 除以 b）

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a / b


# ----------------------------------------------------------
# 将工具绑定到模型
# - tools: 工具列表
# - tools_by_name: 按名称索引的工具字典，方便后续根据名称查找工具
# - model_with_tools: 绑定工具后的模型，LLM 现在知道可以调用这些工具
# ----------------------------------------------------------
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# ============================================================
# Step 2: 定义状态 (State)
# ============================================================
# State 是 LangGraph 的核心概念之一
# 它是一个 TypedDict，定义了图中流转的数据结构
# - messages: 消息列表，使用 Annotated + operator.add 实现消息追加（而非替换）
# - llm_calls: 记录 LLM 被调用的次数

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    # Annotated[list[AnyMessage], operator.add] 表示：
    # 当多个节点返回 messages 时，使用 operator.add（即列表拼接）来合并
    messages: Annotated[list[AnyMessage], operator.add]
    # LLM 调用次数计数器
    llm_calls: int


# ============================================================
# Step 3: 定义模型节点 (Model Node)
# ============================================================
# 这是图中的一个节点，负责调用 LLM
# LLM 会根据对话历史决定：直接回答，还是调用工具

from langchain.messages import SystemMessage


def llm_call(state: MessagesState):
    """LLM 决定是否调用工具"""
    # 构建消息列表：系统提示 + 历史消息
    # SystemMessage 告诉 LLM 它的角色是执行算术运算的助手
    messages = [
        SystemMessage(
            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        )
    ] + state["messages"]

    # 调用绑定工具的模型
    # 如果 LLM 认为需要计算，会在响应中包含 tool_calls
    response = model_with_tools.invoke(messages)

    # 返回更新状态：
    # - messages: 将 LLM 的响应追加到消息列表
    # - llm_calls: 调用次数 +1
    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# ============================================================
# Step 4: 定义工具节点 (Tool Node)
# ============================================================
# 当 LLM 决定调用工具时，这个节点负责执行实际的工具调用
# 它从最新的消息中提取 tool_calls，执行对应的工具，并返回结果

from langchain.messages import ToolMessage


def tool_node(state: MessagesState):
    """执行工具调用"""
    result = []
    # 遍历最新消息中的所有工具调用请求
    for tool_call in state["messages"][-1].tool_calls:
        # 根据工具名称查找对应的工具函数
        tool = tools_by_name[tool_call["name"]]
        # 执行工具，传入参数
        observation = tool.invoke(tool_call["args"])
        # 将工具执行结果封装为 ToolMessage
        # tool_call_id 用于将结果与对应的工具调用关联
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# ============================================================
# Step 5: 定义条件边 (Conditional Edge) 逻辑
# ============================================================
# 条件边用于决定图的下一步走向
# 这里根据 LLM 的响应决定：调用工具，还是结束对话

from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据 LLM 是否请求调用工具来决定下一步"""
    messages = state["messages"]
    last_message = messages[-1]

    # 如果 LLM 的响应包含 tool_calls，说明它想调用工具
    # 路由到 tool_node 执行工具
    if last_message.tool_calls:
        return "tool_node"

    # 否则，LLM 直接回复用户，流程结束
    return END


# ============================================================
# Step 6: 构建 Agent（编译图）
# ============================================================
# 将前面定义的节点和边组装成一个可执行的图

# 创建状态图，指定状态类型
agent_builder = StateGraph(MessagesState)

# 添加节点
# - "llm_call": LLM 调用节点
# - "tool_node": 工具执行节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 添加边，定义节点之间的连接关系
# 流程：START → llm_call → (条件判断) → tool_node 或 END
#                                   ↑              ↓
#                                   └──────────────┘ (工具执行后回到 LLM)
agent_builder.add_edge(START, "llm_call")

# 条件边：从 llm_call 出发，根据 should_continue 的返回值路由
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)

# 工具执行后，回到 LLM 节点继续推理
agent_builder.add_edge("tool_node", "llm_call")

# 编译图，生成可执行的 Agent
agent = agent_builder.compile()

# ============================================================
# Step 7: 可视化并运行 Agent
# ============================================================

# 可视化图结构（需要在 Jupyter Notebook 环境中运行）
# xray=True 显示内部细节
from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# ----------------------------------------------------------
# 运行 Agent
# ----------------------------------------------------------
from langchain.messages import HumanMessage

# 创建用户消息
messages = [HumanMessage(content="Add 3 and 4.")]

# 调用 Agent，传入初始状态
# Agent 会自动：接收消息 → LLM 推理 → 调用工具 → 返回结果
messages = agent.invoke({"messages": messages})

# 打印完整的消息历史
# 预期流程：用户问 "Add 3 and 4." → LLM 调用 add(3, 4) → 工具返回 7 → LLM 回复 "3 + 4 = 7"
for m in messages["messages"]:
    m.pretty_print()
