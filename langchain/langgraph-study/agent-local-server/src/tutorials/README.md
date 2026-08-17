# LangGraph 工作流与代理教程

本目录包含 LangGraph 的核心概念教学示例，基于官方文档 https://docs.langchain.com/oss/python/langgraph/workflows-agents

## 目录结构

```
tutorials/
├── README.md                    # 本文件
├── DEBUG_GUIDE.md               # 详细的运行与调试指南
├── models.py                    # 共享数据模型定义
├── prompt_chaining.py           # 模式 1：顺序工作流（提示链）
├── parallelization.py           # 模式 2：并行化工作流
├── routing.py                   # 模式 3：条件路由工作流
├── evaluator_optimizer.py       # 模式 4：评估-优化循环
├── agent_with_tools.py          # 模式 5：带工具的代理
├── run_examples.py              # 运行所有示例的入口
└── debug_single.py              # 调试单个工作流的工具
```

## 快速开始

### 运行所有示例

```bash
cd /home/ping/gitProjects/my/ai/ping-agent-study/langchain/langgraph-study/agent-local-server
python src/tutorials/run_examples.py
```

### 调试单个工作流

```bash
# 调试顺序工作流
python src/tutorials/debug_single.py --workflow prompt_chaining

# 调试并行化工作流
python src/tutorials/debug_single.py --workflow parallelization

# 调试所有工作流
python src/tutorials/debug_single.py --workflow all
```

### 使用 Python 代码调试

```python
import asyncio
from tutorials.prompt_chaining import graph

async def debug():
    result = await graph.ainvoke({"topic": "测试", "messages": []})
    print(result)

asyncio.run(debug())
```

更多调试方法请参见 [DEBUG_GUIDE.md](DEBUG_GUIDE.md)。

## 核心概念

### 工作流 vs 代理

来自 LangGraph 官方文档：

> **Workflows** have predetermined code paths and are designed to operate in a certain order.
> **Agents** make decisions dynamically, deciding what actions to take and in what order.

- **工作流**：预先定义的代码路径，按特定顺序执行
- **代理**：动态决策，自主决定执行什么操作和顺序

### 五种核心模式

1. **Prompt Chaining（顺序工作流）**
   - 任务分解为一系列步骤
   - 每个步骤的输出是下一步的输入
   - 适用场景：内容生成管道、数据处理流水线

2. **Parallelization（并行化）**
   - 多个独立任务同时执行
   - 最后聚合结果
   - 适用场景：多维度分析、批量处理

3. **Routing（条件路由）**
   - 根据输入类型选择不同的处理路径
   - 使用 LLM 进行智能路由决策
   - 适用场景：客服分类、内容分发

4. **Evaluator-Optimizer（评估-优化循环）**
   - 生成 → 评估 → 优化 → 再评估的闭环
   - 持续改进直到满足质量标准
   - 适用场景：内容优化、代码生成

5. **Agent（代理）**
   - LLM 自主决定使用哪些工具
   - 动态执行路径
   - 适用场景：复杂问题求解、多步骤任务
