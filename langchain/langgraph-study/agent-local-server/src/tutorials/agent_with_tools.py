"""模式 5：带工具的代理（Agent with Tools）

来自 LangGraph 官方文档：
> Agents make decisions dynamically, deciding what actions to take and in what order.
> The LLM decides when and how to use tools, then processes the results.

核心思想：
- LLM 自主决定是否需要使用工具
- 如果需要，选择合适的工具执行
- 将工具结果返回给 LLM
- 循环直到 LLM 认为任务完成

适用场景：
- 复杂问题求解（需要多步骤推理）
- 数据查询（需要调用外部 API）
- 代码执行（需要运行代码）

图结构：
    START
      ↓
   agent（LLM 决策）
      ↓
   ┌────────────────┐
   ↓                │
has_tool_calls     │
   ↓                │
   tools ←──────────┘
   ↓
   agent
   ↓
no_tool_calls
   ↓
    END

实现方式：
使用条件边实现代理循环，LLM 决定是否调用工具。
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from tutorials.models import AgentState

# ============================================================================
# 工具定义
# ============================================================================


def search_tool(query: str) -> str:
    """搜索工具：模拟搜索 API。

    实际项目中应使用真实的搜索 API：
        from langchain_community.tools import TavilySearchResults
        search = TavilySearchResults(max_results=3)
        return search.invoke(query)
    """
    # 模拟搜索结果
    mock_results = {
        "天气": "今天北京晴，气温 25°C，适合出行。",
        "新闻": "最新科技新闻：AI 技术取得重大突破。",
        "python": "Python 是一种解释型、面向对象的高级编程语言。",
    }

    # 根据关键词返回结果
    for key, value in mock_results.items():
        if key in query:
            return value

    return f"搜索 '{query}' 的结果：找到相关信息。"


def calculator_tool(expression: str) -> str:
    """计算器工具：执行数学计算。

    实际项目中应使用安全的计算库：
        from langchain_community.tools import Calculator
        calculator = Calculator()
        return calculator.invoke(expression)
    """
    try:
        # 安全的数学表达式求值
        # 注意：生产环境应使用更安全的方法
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def wiki_tool(topic: str) -> str:
    """维基百科工具：获取百科信息。

    实际项目中应使用维基百科 API：
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.invoke(topic)
    """
    # 模拟维基百科结果
    mock_wiki = {
        "python": "Python 由 Guido van Rossum 于 1991 年创建，是一种通用编程语言。",
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架。",
        "langgraph": "LangGraph 是 LangChain 的图编排框架，用于构建有状态的代理。",
    }

    for key, value in mock_wiki.items():
        if key in topic.lower():
            return value

    return f"关于 '{topic}' 的维基百科信息：暂无相关条目。"


# 工具映射表
TOOLS = {
    "search": search_tool,
    "calculator": calculator_tool,
    "wiki": wiki_tool,
}


# ============================================================================
# 辅助函数（模拟 LLM 调用）
# ============================================================================


def _simulate_llm_with_tools(messages: list[dict]) -> dict:
    """模拟带工具调用的 LLM。

    实际项目中应使用：
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4")
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
        return response

    这里模拟 LLM 的工具调用决策：
    - 如果消息中包含"搜索"、"查"等关键词 → 调用 search
    - 如果消息中包含数字和运算符 → 调用 calculator
    - 如果消息中包含"什么是"、"介绍"等 → 调用 wiki
    - 否则 → 直接回答（无工具调用）
    """
    # 获取最后一条用户消息
    last_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_message = msg.get("content", "")
            break

    last_message_lower = last_message.lower()

    # 根据关键词决定工具调用
    if any(kw in last_message_lower for kw in ["搜索", "查", "找", "search"]):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": f'{{"query": "{last_message}"}}',
                    },
                }
            ],
        }
    elif any(kw in last_message_lower for kw in ["计算", "算", "多少", "calculate"]):
        # 提取数学表达式（简化处理）
        import re

        numbers = re.findall(r"\d+[\+\-\*\/]\d+", last_message)
        expression = numbers[0] if numbers else "1+1"
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": f'{{"expression": "{expression}"}}',
                    },
                }
            ],
        }
    elif any(kw in last_message_lower for kw in ["什么是", "介绍", "what is", "定义"]):
        # 提取主题
        topic = last_message.replace("什么是", "").replace("介绍", "").strip()
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "wiki",
                        "arguments": f'{{"topic": "{topic}"}}',
                    },
                }
            ],
        }
    else:
        # 无工具调用，直接回答
        return {
            "role": "assistant",
            "content": f"我理解您的问题：{last_message}。这是一个有趣的话题，让我来回答...",
            "tool_calls": None,
        }


# ============================================================================
# 代理节点
# ============================================================================


def agent(state: AgentState) -> dict:
    """代理节点：LLM 决策中心。

    代理的核心逻辑：
    1. 读取对话历史
    2. 调用 LLM（可能返回工具调用）
    3. 将 LLM 响应添加到对话历史

    关于工具调用：
    - LLM 返回 tool_calls 字段表示需要调用工具
    - tool_calls 是一个列表，可以同时调用多个工具
    - 每个工具调用有 id、type、function 等字段
    """
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)

    # 调用 LLM（模拟）
    llm_response = _simulate_llm_with_tools(messages)

    # 更新对话历史
    new_messages = messages + [llm_response]

    # 更新迭代次数
    new_iterations = iterations + 1

    # 记录日志
    tool_calls = llm_response.get("tool_calls")
    if tool_calls:
        log_msg = f"[agent] LLM 决定调用工具：{tool_calls[0]['function']['name']}"
    else:
        log_msg = "[agent] LLM 直接回答，无工具调用"

    return {
        "messages": new_messages,
        "iterations": new_iterations,
        "final_answer": llm_response.get("content"),
    }


# ============================================================================
# 工具执行节点
# ============================================================================


def tools(state: AgentState) -> dict:
    """工具执行节点：执行 LLM 请求的工具。

    工具执行流程：
    1. 从 LLM 响应中提取工具调用
    2. 执行对应的工具函数
    3. 将工具结果添加到对话历史

    关于错误处理：
    - 工具执行可能失败（网络错误、API 限流等）
    - 应该捕获异常并返回友好的错误信息
    - 让 LLM 决定如何处理错误
    """
    messages = state.get("messages", [])
    tool_results = state.get("tool_results", [])

    # 获取最后一条消息（LLM 的工具调用）
    last_message = messages[-1] if messages else {}
    tool_calls = last_message.get("tool_calls", [])

    if not tool_calls:
        return {"messages": messages}

    # 执行每个工具调用
    new_messages = messages.copy()
    new_tool_results = tool_results.copy()

    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]

        # 解析参数
        import json

        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = {}

        # 执行工具
        tool_func = TOOLS.get(function_name)
        if tool_func:
            try:
                # 调用工具函数
                if function_name == "search":
                    result = tool_func(args.get("query", ""))
                elif function_name == "calculator":
                    result = tool_func(args.get("expression", ""))
                elif function_name == "wiki":
                    result = tool_func(args.get("topic", ""))
                else:
                    result = f"未知工具：{function_name}"
            except Exception as e:
                result = f"工具执行错误：{e}"
        else:
            result = f"未找到工具：{function_name}"

        # 添加工具结果到对话历史
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": result,
        }
        new_messages.append(tool_message)

        # 记录工具结果
        new_tool_results.append(
            {
                "tool": function_name,
                "args": args,
                "result": result,
            }
        )

    return {
        "messages": new_messages,
        "tool_results": new_tool_results,
    }


# ============================================================================
# 路由函数
# ============================================================================


def should_continue_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """路由函数：根据 LLM 响应决定是否继续。

    代理循环的控制逻辑：
    - 如果 LLM 返回工具调用 → 执行工具
    - 如果 LLM 直接回答 → 结束

    关于安全限制：
    - 检查迭代次数，防止无限循环
    - 可以设置最大迭代次数
    """
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    max_iterations = 10  # 最大迭代次数

    # 检查是否达到最大迭代次数
    if iterations >= max_iterations:
        return "__end__"

    # 获取最后一条消息
    last_message = messages[-1] if messages else {}

    # 检查是否有工具调用
    tool_calls = last_message.get("tool_calls")
    if tool_calls:
        return "tools"
    else:
        return "__end__"


# ============================================================================
# 图构建
# ============================================================================


def build_agent_graph() -> StateGraph:
    """构建并返回代理图。

    关键点：
    1. 代理节点是决策中心
    2. 工具节点执行实际操作
    3. 条件边实现循环

    与工作流的区别：
    - 工作流：预定义的执行路径
    - 代理：动态决策，LLM 决定下一步
    """
    # 创建 StateGraph
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tools)

    # 添加边：START → agent
    workflow.add_edge(START, "agent")

    # 添加条件边：agent → 根据 LLM 响应路由
    workflow.add_conditional_edges(
        source="agent",
        path=should_continue_agent,
        path_map={
            "tools": "tools",
            "__end__": END,
        },
    )

    # 添加边：tools → agent（工具执行后返回代理）
    workflow.add_edge("tools", "agent")

    return workflow


# ============================================================================
# 导出编译后的图
# ============================================================================

# 构建并编译图
graph = build_agent_graph().compile()
