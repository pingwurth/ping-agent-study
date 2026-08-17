# LangGraph 教程运行与调试指南

## 快速开始

### 1. 运行所有示例

```bash
# 方式 1：直接运行脚本
cd src/tutorials
python run_examples.py

# 方式 2：从项目根目录运行
cd /home/ping/gitProjects/my/ai/ping-agent-study/langchain/langgraph-study/agent-local-server
python src/tutorials/run_examples.py
```

### 2. 运行单个示例

```bash
# 运行顺序工作流
python -c "
import asyncio
from tutorials.prompt_chaining import graph

async def main():
    result = await graph.ainvoke({'topic': '测试主题', 'messages': []})
    print(result)

asyncio.run(main())
"

# 运行并行化工作流
python -c "
import asyncio
from tutorials.parallelization import graph

async def main():
    result = await graph.ainvoke({'text': '测试文本', 'messages': []})
    print(result)

asyncio.run(main())
"
```

## 调试方法

### 方法 1：使用 print 调试（简单直接）

在节点函数中添加 print 语句：

```python
def generate_outline(state: StoryState) -> dict:
    print(f"[DEBUG] 输入状态: {state}")  # 添加调试输出

    prompt = f"..."
    outline = _simulate_llm(prompt)

    print(f"[DEBUG] 生成的大纲: {outline}")  # 添加调试输出

    return {
        "outline": outline,
        "messages": [...],
    }
```

### 方法 2：使用 Python debugger (pdb)

```python
def generate_outline(state: StoryState) -> dict:
    import pdb; pdb.set_trace()  # 设置断点

    prompt = f"..."
    outline = _simulate_llm(prompt)

    return {
        "outline": outline,
        "messages": [...],
    }
```

运行时会进入交互式调试器：
- `n` - 执行下一行
- `s` - 进入函数
- `c` - 继续执行
- `p variable` - 打印变量
- `q` - 退出调试器

### 方法 3：使用 IDE 调试器

#### VS Code 配置

创建 `.vscode/launch.json`：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "运行教程示例",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/src/tutorials/run_examples.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "调试单个工作流",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/src/tutorials/debug_single.py",
            "console": "integrated终端",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

然后在代码中设置断点（点击行号左侧），按 F5 启动调试。

### 方法 4：使用 LangGraph 的 streaming 功能

```python
import asyncio
from tutorials.prompt_chaining import graph

async def debug_with_streaming():
    """使用 streaming 观察每一步的执行。"""
    input_data = {
        "topic": "调试测试",
        "messages": [],
    }

    # 使用 astream 观察每一步
    async for event in graph.astream(input_data):
        print(f"[STREAM] 事件: {event}")

asyncio.run(debug_with_streaming())
```

### 方法 5：使用 LangSmith 进行可视化调试

LangSmith 提供了图形化的调试界面：

```python
import os
from langsmith import traceable

# 设置 LangSmith（需要 API key）
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# 然后正常运行图，LangSmith 会自动记录所有步骤
```

访问 https://smith.langchain.com 查看可视化追踪。

## 创建调试脚本

创建 `src/tutorials/debug_single.py`：

```python
"""调试单个工作流示例。"""

import asyncio
from tutorials.prompt_chaining import graph

async def debug_prompt_chaining():
    """调试顺序工作流。"""
    print("=" * 60)
    print("调试顺序工作流")
    print("=" * 60)

    input_data = {
        "topic": "调试测试主题",
        "messages": [],
    }

    print(f"\n输入: {input_data}")

    # 方式 1：使用 invoke（同步执行）
    result = await graph.ainvoke(input_data)

    print(f"\n输出:")
    for key, value in result.items():
        if key == "messages":
            print(f"  {key}:")
            for msg in value:
                print(f"    - {msg}")
        else:
            print(f"  {key}: {str(value)[:100]}...")

if __name__ == "__main__":
    asyncio.run(debug_prompt_chaining())
```

## 常见问题排查

### 1. 导入错误

```
ModuleNotFoundError: No module named 'tutorials'
```

**解决方案：**
```bash
# 确保在正确的目录运行
cd /home/ping/gitProjects/my/ai/ping-agent-study/langchain/langgraph-study/agent-local-server
python src/tutorials/run_examples.py

# 或者设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m tutorials.run_examples
```

### 2. 异步运行错误

```
RuntimeError: This event loop is already running
```

**解决方案：**
```python
# 使用 asyncio.run() 而不是直接调用
import asyncio
asyncio.run(main())

# 或者在 Jupyter notebook 中使用
import nest_asyncio
nest_asyncio.apply()
```

### 3. 状态更新不生效

**检查点：**
- 确保返回的是字典（不是 None）
- 确保字典的 key 与状态定义匹配
- 对于列表字段，确保手动合并（不是替换）

### 4. 条件路由不工作

**检查点：**
- 确保路由函数返回的是节点名称字符串
- 确保 `path_map` 中的 key 与路由函数返回值匹配
- 确保所有可能的返回值都有对应的边

## 性能分析

### 测量执行时间

```python
import asyncio
import time
from tutorials.parallelization import graph

async def measure_performance():
    input_data = {"text": "测试文本", "messages": []}

    start = time.time()
    result = await graph.ainvoke(input_data)
    end = time.time()

    print(f"执行时间: {end - start:.2f} 秒")
    return result

asyncio.run(measure_performance())
```

### 使用 cProfile 进行性能分析

```bash
python -m cProfile -s cumulative src/tutorials/run_examples.py
```

## 下一步

1. 将模拟的 LLM 调用替换为真实的 LangChain 调用
2. 添加更多测试用例
3. 集成 LangSmith 进行可视化调试
4. 添加错误处理和重试逻辑
