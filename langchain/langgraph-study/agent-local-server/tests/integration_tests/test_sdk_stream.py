"""SDK 流式输出测试 — 通过 LangGraph SDK 客户端测试图的流式输出。

此测试需要 LangGraph Server 运行在 localhost:2024。
启动方式：langgraph dev

运行此测试：
    pytest tests/integration_tests/test_sdk_stream.py -v
"""


import pytest
from langgraph_sdk import get_client


@pytest.mark.langsmith
@pytest.mark.skipif(
    True,  # 默认跳过，因为需要运行 langgraph dev
    reason="需要 LangGraph Server 运行在 localhost:2024，请先执行 langgraph dev",
)
async def test_sdk_stream() -> None:
    """通过 SDK 客户端测试流式输出。"""
    client = get_client(url="http://localhost:2024")

    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent",  # Name of assistant. Defined in langgraph.json.
        input={
            "email_content": "How do I reset my password?",
            "sender_email": "user@test.com",
            "email_id": "SDK-TEST-001",
            "classification": None,
            "search_results": None,
            "customer_history": None,
            "draft_response": None,
            "messages": None,
        },
    ):
        print(f"Event: {chunk.event}")
        print(chunk.data)
        print()
