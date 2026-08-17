"""调试单个工作流示例。

本脚本用于调试单个工作流，提供详细的执行信息。

使用方法：
    python src/tutorials/debug_single.py

可选参数：
    --workflow: 要调试的工作流名称（默认: prompt_chaining）
    --verbose: 显示详细信息（默认: True）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

# 添加当前目录到 Python 路径
sys.path.insert(0, "/home/ping/gitProjects/my/ai/ping-agent-study/langchain/langgraph-study/agent-local-server/src")


# ============================================================================
# 调试函数
# ============================================================================

async def debug_prompt_chaining(verbose: bool = True) -> dict[str, Any]:
    """调试顺序工作流。"""
    from tutorials.prompt_chaining import graph

    print("=" * 60)
    print("调试顺序工作流（Prompt Chaining）")
    print("=" * 60)

    input_data = {
        "topic": "一只会编程的猫",
        "messages": [],
    }

    print(f"\n输入: {input_data}")

    result = await graph.ainvoke(input_data)

    if verbose:
        print("\n输出:")
        for key, value in result.items():
            if key == "messages":
                print(f"  {key}:")
                for msg in value:
                    print(f"    - {msg}")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"  {key}: {value_str}")

    return result


async def debug_parallelization(verbose: bool = True) -> dict[str, Any]:
    """调试并行化工作流。"""
    from tutorials.parallelization import graph

    print("=" * 60)
    print("调试并行化工作流（Parallelization）")
    print("=" * 60)

    input_data = {
        "text": """
        LangGraph 是 LangChain 的图编排框架，用于构建有状态的代理。
        它支持多种工作流模式，包括顺序执行、并行化、条件路由等。
        LangGraph 的核心思想是将复杂任务分解为多个节点，通过边连接形成图。
        """,
        "messages": [],
    }

    print(f"\n输入文本: {input_data['text'][:50]}...")

    result = await graph.ainvoke(input_data)

    if verbose:
        print("\n输出:")
        for key, value in result.items():
            if key == "messages":
                print(f"  {key}:")
                for msg in value:
                    print(f"    - {msg}")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"  {key}: {value_str}")

    return result


async def debug_routing(verbose: bool = True) -> dict[str, Any]:
    """调试条件路由工作流。"""
    from tutorials.routing import graph

    print("=" * 60)
    print("调试条件路由工作流（Routing）")
    print("=" * 60)

    test_cases = [
        {"user_input": "这段 Python 代码有 bug，帮我看看", "messages": []},
        {"user_input": "计算 123 + 456 的结果", "messages": []},
        {"user_input": "今天天气怎么样？", "messages": []},
    ]

    results = []
    for i, input_data in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {input_data['user_input']}")

        result = await graph.ainvoke(input_data)
        results.append(result)

        if verbose:
            print(f"  分类结果: {result.get('category', '无')}")
            print(f"  回复: {str(result.get('response', '无'))[:50]}...")

            print("  处理日志:")
            for msg in result.get("messages", []):
                print(f"    - {msg}")

    return results


async def debug_evaluator_optimizer(verbose: bool = True) -> dict[str, Any]:
    """调试评估-优化循环。"""
    from tutorials.evaluator_optimizer import graph

    print("=" * 60)
    print("调试评估-优化循环（Evaluator-Optimizer）")
    print("=" * 60)

    input_data = {
        "task": "写一段关于 LangGraph 的介绍",
        "content": None,
        "evaluation": None,
        "feedback": None,
        "iteration": 0,
        "max_iterations": 3,
        "messages": [],
    }

    print(f"\n任务: {input_data['task']}")
    print(f"最大迭代次数: {input_data['max_iterations']}")

    result = await graph.ainvoke(input_data)

    if verbose:
        print("\n输出:")
        print(f"  迭代次数: {result.get('iteration', 0)}")
        print(f"  评估结果: {result.get('evaluation', '无')}")

        print("  处理日志:")
        for msg in result.get("messages", []):
            print(f"    - {msg}")

    return result


async def debug_agent_with_tools(verbose: bool = True) -> dict[str, Any]:
    """调试带工具的代理。"""
    from tutorials.agent_with_tools import graph

    print("=" * 60)
    print("调试带工具的代理（Agent with Tools）")
    print("=" * 60)

    input_data = {
        "messages": [{"role": "user", "content": "搜索 LangGraph 的最新信息"}],
        "tool_results": [],
        "final_answer": None,
        "iterations": 0,
    }

    print(f"\n用户输入: {input_data['messages'][0]['content']}")

    result = await graph.ainvoke(input_data)

    if verbose:
        print("\n输出:")
        print(f"  迭代次数: {result.get('iterations', 0)}")
        print(f"  工具调用次数: {len(result.get('tool_results', []))}")

        if result.get("tool_results"):
            print("  工具调用详情:")
            for tool_result in result["tool_results"][:3]:  # 只显示前 3 个
                print(f"    - {tool_result['tool']}: {tool_result['result'][:30]}...")

        final_answer = result.get("final_answer") or "无"
        print(f"  最终答案: {final_answer[:50]}...")

    return result


# ============================================================================
# 主函数
# ============================================================================

async def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="调试 LangGraph 工作流示例")
    parser.add_argument(
        "--workflow",
        type=str,
        default="prompt_chaining",
        choices=["prompt_chaining", "parallelization", "routing", "evaluator_optimizer", "agent_with_tools", "all"],
        help="要调试的工作流名称",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="是否显示详细信息",
    )

    args = parser.parse_args()

    print("LangGraph 工作流调试工具")
    print("=" * 60)

    if args.workflow == "all":
        await debug_prompt_chaining(args.verbose)
        await debug_parallelization(args.verbose)
        await debug_routing(args.verbose)
        await debug_evaluator_optimizer(args.verbose)
        await debug_agent_with_tools(args.verbose)
    elif args.workflow == "prompt_chaining":
        await debug_prompt_chaining(args.verbose)
    elif args.workflow == "parallelization":
        await debug_parallelization(args.verbose)
    elif args.workflow == "routing":
        await debug_routing(args.verbose)
    elif args.workflow == "evaluator_optimizer":
        await debug_evaluator_optimizer(args.verbose)
    elif args.workflow == "agent_with_tools":
        await debug_agent_with_tools(args.verbose)

    print("\n" + "=" * 60)
    print("调试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
