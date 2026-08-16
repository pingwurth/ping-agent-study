# LangGraph Hello World 示例
# 演示如何使用 StateGraph 构建一个简单的对话图

from langgraph.graph import StateGraph, MessagesState, START, END


def mock_llm(state: MessagesState):
    """模拟 LLM 节点：接收消息状态，返回固定回复"""
    return {"messages": [{"role": "ai", "content": "你好，世界"}]}


# 构建状态图：节点为 mock_llm，边为 START -> mock_llm -> END
graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

# 运行图，传入用户消息并打印结果
result = graph.invoke({"messages": [{"role": "user", "content": "你好！"}]})
print(result)
