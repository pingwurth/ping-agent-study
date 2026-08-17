"""LangGraph 工作流与代理教程。

本包包含 LangGraph 的核心概念教学示例，展示五种工作流和代理模式：
1. 顺序工作流（Prompt Chaining）
2. 并行化工作流（Parallelization）
3. 条件路由工作流（Routing）
4. 评估-优化循环（Evaluator-Optimizer）
5. 带工具的代理（Agent with Tools）

使用方法：
    from tutorials.prompt_chaining import graph as prompt_chaining_graph
    from tutorials.parallelization import graph as parallelization_graph
    from tutorials.routing import graph as routing_graph
    from tutorials.evaluator_optimizer import graph as evaluator_optimizer_graph
    from tutorials.agent_with_tools import graph as agent_with_tools_graph

    # 运行示例
    result = await prompt_chaining_graph.ainvoke({"topic": "测试", "messages": []})
"""
