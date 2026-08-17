"""运行所有教程示例。

本脚本演示如何运行 LangGraph 的五种核心模式：
1. 顺序工作流（Prompt Chaining）
2. 并行化工作流（Parallelization）
3. 条件路由工作流（Routing）
4. 评估-优化循环（Evaluator-Optimizer）
5. 带工具的代理（Agent with Tools）

使用方法：
    cd src
    python -m tutorials.run_examples

每个示例都会：
1. 构建图
2. 准备输入数据
3. 运行图
4. 显示结果
"""

from __future__ import annotations

import asyncio
import sys

# 添加当前目录到 Python 路径
sys.path.insert(
    0,
    "/home/ping/gitProjects/my/ai/ping-agent-study/langchain/langgraph-study/agent-local-server/src",
)


# ============================================================================
# 示例 1：顺序工作流
# ============================================================================


async def run_prompt_chaining() -> None:
    """运行顺序工作流示例。

    演示：
    - 任务分解为多个步骤
    - 每个步骤的输出是下一步的输入
    - 最终生成完整的故事
    """
    print("\n" + "=" * 60)
    print("示例 1：顺序工作流（Prompt Chaining）")
    print("=" * 60)

    from tutorials.prompt_chaining import graph as prompt_chaining_graph

    # 准备输入
    input_data = {
        "topic": "一只会编程的猫",
        "messages": [],
    }

    # 运行图
    print("\n输入：", input_data["topic"])
    print("\n执行流程：")

    result = await prompt_chaining_graph.ainvoke(input_data)

    # 显示结果
    print("\n结果：")
    print(f"  大纲：{result.get('outline', '无')[:50]}...")
    print(f"  角色：{result.get('characters', '无')[:50]}...")
    print(f"  故事：{result.get('story', '无')[:50]}...")
    print(f"  标题：{result.get('title', '无')}")

    print("\n处理日志：")
    for msg in result.get("messages", []):
        print(f"  - {msg}")


# ============================================================================
# 示例 2：并行化工作流
# ============================================================================


async def run_parallelization() -> None:
    """运行并行化工作流示例。

    演示：
    - 多个分析器同时运行
    - 利用并行性提高效率
    - 最后聚合所有结果
    """
    print("\n" + "=" * 60)
    print("示例 2：并行化工作流（Parallelization）")
    print("=" * 60)

    from tutorials.parallelization import graph as parallelization_graph

    # 准备输入
    input_data = {
        "text": """
        LangGraph 是 LangChain 的图编排框架，用于构建有状态的代理。
        它支持多种工作流模式，包括顺序执行、并行化、条件路由等。
        LangGraph 的核心思想是将复杂任务分解为多个节点，通过边连接形成图。
        """,
        "messages": [],
    }

    # 运行图
    print("\n输入文本：", input_data["text"][:50], "...")
    print("\n执行流程：")

    result = await parallelization_graph.ainvoke(input_data)

    # 显示结果
    print("\n分析结果：")
    print(f"  情感分析：{result.get('sentiment', '无')[:50]}...")
    print(f"  关键词：{result.get('keywords', '无')[:50]}...")
    print(f"  摘要：{result.get('summary', '无')[:50]}...")
    print(f"  实体识别：{result.get('entities', '无')[:50]}...")
    print(f"  综合报告：{result.get('final_report', '无')[:50]}...")

    print("\n处理日志：")
    for msg in result.get("messages", []):
        print(f"  - {msg}")


# ============================================================================
# 示例 3：条件路由工作流
# ============================================================================


async def run_routing() -> None:
    """运行条件路由工作流示例。

    演示：
    - LLM 分析输入类型
    - 根据类型选择不同处理路径
    - 每条路径专门处理一类输入
    """
    print("\n" + "=" * 60)
    print("示例 3：条件路由工作流（Routing）")
    print("=" * 60)

    from tutorials.routing import graph as routing_graph

    # 测试不同类型的输入
    test_cases = [
        {"user_input": "这段 Python 代码有 bug，帮我看看", "messages": []},
        {"user_input": "计算 123 + 456 的结果", "messages": []},
        {"user_input": "今天天气怎么样？", "messages": []},
    ]

    for i, input_data in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}：{input_data['user_input']}")

        # 运行图
        result = await routing_graph.ainvoke(input_data)

        # 显示结果
        print(f"  分类结果：{result.get('category', '无')}")
        print(f"  回复：{result.get('response', '无')[:50]}...")

        print("  处理日志：")
        for msg in result.get("messages", []):
            print(f"    - {msg}")


# ============================================================================
# 示例 4：评估-优化循环
# ============================================================================


async def run_evaluator_optimizer() -> None:
    """运行评估-优化循环示例。

    演示：
    - 生成器生成内容
    - 评估器评估质量
    - 循环直到质量达标
    """
    print("\n" + "=" * 60)
    print("示例 4：评估-优化循环（Evaluator-Optimizer）")
    print("=" * 60)

    from tutorials.evaluator_optimizer import graph as evaluator_optimizer_graph

    # 准备输入
    input_data = {
        "task": "写一段关于 LangGraph 的介绍",
        "content": None,
        "evaluation": None,
        "feedback": None,
        "iteration": 0,
        "max_iterations": 3,
        "messages": [],
    }

    # 运行图
    print("\n任务：", input_data["task"])
    print("最大迭代次数：", input_data["max_iterations"])
    print("\n执行流程：")

    result = await evaluator_optimizer_graph.ainvoke(input_data)

    # 显示结果
    print("\n最终结果：")
    print(f"  迭代次数：{result.get('iteration', 0)}")
    print(f"  评估结果：{result.get('evaluation', '无')}")
    print(f"  最终内容：{result.get('content', '无')[:50]}...")

    print("\n处理日志：")
    for msg in result.get("messages", []):
        print(f"  - {msg}")


# ============================================================================
# 示例 5：带工具的代理
# ============================================================================


async def run_agent_with_tools() -> None:
    """运行带工具的代理示例。

    演示：
    - LLM 自主决定使用哪些工具
    - 动态执行路径
    - 循环直到任务完成
    """
    print("\n" + "=" * 60)
    print("示例 5：带工具的代理（Agent with Tools）")
    print("=" * 60)

    from tutorials.agent_with_tools import graph as agent_with_tools_graph

    # 测试不同类型的查询
    test_cases = [
        {
            "messages": [{"role": "user", "content": "搜索 LangGraph 的最新信息"}],
            "tool_results": [],
            "final_answer": None,
            "iterations": 0,
        },
        {
            "messages": [{"role": "user", "content": "计算 123 * 456"}],
            "tool_results": [],
            "final_answer": None,
            "iterations": 0,
        },
        {
            "messages": [{"role": "user", "content": "什么是 Python？"}],
            "tool_results": [],
            "final_answer": None,
            "iterations": 0,
        },
    ]

    for i, input_data in enumerate(test_cases, 1):
        user_message = input_data["messages"][0]["content"]
        print(f"\n测试案例 {i}：{user_message}")

        # 运行图
        result = await agent_with_tools_graph.ainvoke(input_data)

        # 显示结果
        print(f"  迭代次数：{result.get('iterations', 0)}")
        print(f"  工具调用次数：{len(result.get('tool_results', []))}")

        if result.get("tool_results"):
            print("  工具调用详情：")
            for tool_result in result["tool_results"]:
                print(f"    - {tool_result['tool']}: {tool_result['result'][:30]}...")

        final_answer = result.get('final_answer') or '无'
        print(f"  最终答案：{final_answer[:50]}...")

        print("  处理日志：")
        for msg in result.get("messages", []):
            if isinstance(msg, str):
                print(f"    - {msg}")


# ============================================================================
# 主函数
# ============================================================================


async def main() -> None:
    """运行所有教程示例。"""
    print("LangGraph 工作流与代理教程")
    print(
        "基于官方文档：https://docs.langchain.com/oss/python/langgraph/workflows-agents"
    )

    # 运行所有示例
    await run_prompt_chaining()
    await run_parallelization()
    await run_routing()
    await run_evaluator_optimizer()
    await run_agent_with_tools()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
