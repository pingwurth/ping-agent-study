# ============================================================
# LangGraph Functional API 教程
# ============================================================
# 本示例展示如何使用 LangGraph 的 Functional API 构建 Agent
# 与 Graph API 不同，Functional API 使用 @task 和 @entrypoint 装饰器
# 更接近传统的函数式编程风格，代码更简洁直观

# Step 1: 定义工具 (Tools) 和模型 (Model)
import os
import warnings

# 忽略 LangGraph 的 Beta 警告（v3 流式协议实验性警告）
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

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

# ----------------------------------------------------------
# 导入 LangGraph Functional API 相关模块
# - add_messages: 用于合并消息列表
# - @task: 标记可执行的任务函数
# - @entrypoint: 标记图的入口点函数
# ----------------------------------------------------------
from langgraph.graph import add_messages
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
)
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task


# ============================================================
# Step 2: 定义模型任务 (Model Task)
# ============================================================
# @task 装饰器将函数标记为可执行的任务
# 任务可以被并行执行，返回的是一个 Future 对象

@task
def call_llm(messages: list[BaseMessage]):
    """调用 LLM，决定是否调用工具"""
    # 构建消息列表：系统提示 + 历史消息
    # SystemMessage 告诉 LLM 它的角色是执行算术运算的助手
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + messages
    )


# ============================================================
# Step 3: 定义工具任务 (Tool Task)
# ============================================================
# 每个工具调用都会创建一个独立的任务
# 多个工具调用可以并行执行

@task
def call_tool(tool_call: ToolCall):
    """执行工具调用"""
    # 根据工具名称查找对应的工具函数
    tool = tools_by_name[tool_call["name"]]
    # 执行工具，传入整个 tool_call 对象
    return tool.invoke(tool_call)


# ============================================================
# Step 4: 定义 Agent（入口点）
# ============================================================
# @entrypoint() 装饰器标记这是图的入口函数
# 函数的执行流程：
# 1. 调用 LLM 获取响应
# 2. 如果 LLM 请求调用工具，并行执行所有工具
# 3. 将工具结果添加到消息历史
# 4. 重复直到 LLM 不再请求工具调用

@entrypoint()
def agent(messages: list[BaseMessage]):
    # 首次调用 LLM
    # .result() 阻塞等待 Future 完成并获取结果
    model_response = call_llm(messages).result()

    # 循环处理工具调用
    while True:
        # 如果 LLM 没有请求调用工具，退出循环
        if not model_response.tool_calls:
            break

        # 并行执行所有工具调用
        # 每个 tool_call 创建一个独立的任务（Future）
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]

        # 等待所有工具执行完成
        tool_results = [fut.result() for fut in tool_result_futures]

        # 将 LLM 响应和工具结果添加到消息历史
        # add_messages 会正确处理消息追加
        messages = add_messages(messages, [model_response, *tool_results])

        # 再次调用 LLM，让它基于工具结果继续推理
        model_response = call_llm(messages).result()

    # 将最终的 LLM 响应添加到消息历史
    messages = add_messages(messages, model_response)
    return messages


# ============================================================
# Step 5: 运行 Agent
# ============================================================
# 使用 stream_events 方法获取执行过程的事件流
# version="v3" 使用最新的事件格式

# 创建用户消息
messages = [HumanMessage(content="Add 3 and 4.")]

# 流式获取执行事件
# stream_events 返回一个事件流，可以实时观察 Agent 的执行过程
stream = agent.stream_events(messages, version="v3")

# 打印每个事件快照
# 预期流程：
# 1. LLM 调用 → 请求调用 add(3, 4)
# 2. 工具执行 → 返回 7
# 3. LLM 响应 → "3 + 4 = 7"
for snapshot in stream.values:
    print(snapshot)
    print("\n")
