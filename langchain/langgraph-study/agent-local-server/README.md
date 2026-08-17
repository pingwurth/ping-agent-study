# LangGraph Agent — 邮件客服代理 & 工作流教程

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的多节点代理项目，包含两个部分：

1. **`src/agent/`** — 生产级邮件客服代理（7 节点，含人工审核中断）
2. **`src/tutorials/`** — 5 种 LangGraph 核心模式的教学示例

---

## 项目结构

```
agent-local-server/
├── .env                           # 环境变量（API 密钥、模型配置）
├── langgraph.json                 # LangGraph Server 配置入口
├── pyproject.toml                 # Python 项目元数据 & 依赖
├── Makefile                       # 快捷命令（test / lint / format）
├── uv.lock                        # uv 锁定文件
│
├── src/
│   ├── agent/                     # ===== 邮件客服代理 =====
│   │   ├── __init__.py            #   导出 graph 对象
│   │   ├── models.py              #   数据模型：EmailClassification, HumanDecision, EmailAgentState
│   │   ├── nodes.py               #   7 个节点实现（read_email → classify → search/bug → draft → review → send）
│   │   └── graph.py               #   StateGraph 组装、编译、导出
│   │
│   └── tutorials/                 # ===== LangGraph 教程 =====
│       ├── __init__.py
│       ├── models.py              #   5 种模式的共享状态类型
│       ├── prompt_chaining.py     #   模式 1：顺序工作流（提示链）
│       ├── parallelization.py     #   模式 2：并行化（Fan-out / Fan-in）
│       ├── routing.py             #   模式 3：条件路由
│       ├── evaluator_optimizer.py #   模式 4：评估-优化循环
│       ├── agent_with_tools.py    #   模式 5：带工具的代理
│       ├── run_examples.py        #   一键运行所有教程
│       ├── debug_single.py        #   调试单个工作流的 CLI 工具
│       ├── README.md              #   教程详细说明
│       └── DEBUG_GUIDE.md         #   调试指南
│
├── tests/
│   ├── conftest.py                #   pytest 共享 fixture
│   ├── unit_tests/
│   │   ├── __init__.py
│   │   └── test_configuration.py  #   8 个单元测试（图结构、分类模拟、状态验证）
│   └── integration_tests/
│       ├── __init__.py
│       ├── test_graph.py          #   5 个集成测试（完整流程、中断/恢复、拒绝路径）
│       └── test_sdk_stream.py     #   SDK 流式测试（需启动服务器，默认跳过）
│
└── static/
    └── studio_ui.png              # LangGraph Studio 截图
```

---

## 环境准备

### 1. 安装依赖

```bash
cd agent-local-server

# 使用 pip
pip install -e ".[dev]" "langgraph-cli[inmem]"

# 或使用 uv（更快）
uv sync
```

### 2. 配置环境变量

项目已包含 `.env` 文件，内容如下：

```env
# LangSmith 追踪（可选）
LANGSMITH_PROJECT=new-agent
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY=lsv2_...

# 模型配置
MODEL_PROVIDER=openai
OPENAI_API_KEY=your-api-key
BASE_URL=https://your-api-endpoint/v1
MODEL_NAME=your-model-name
```

> 所有 LLM 调用和外部 API 目前均为**模拟实现**（关键词分类、模板回复），无需真实 API 密钥即可运行。

---

## 运行指南

### 一、启动 LangGraph Server（生产代理）

```bash
langgraph dev
```

启动后：
- **API 地址**: `http://localhost:2024`
- **Studio UI**: 浏览器打开 LangGraph Studio，连接本地服务器
- **图入口**: `langgraph.json` 中定义的 `"agent"` → `src/agent/graph.py:graph`

SDK 客户端调用示例：

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")
# 创建 assistant、发送邮件、处理中断...
```

### 二、运行邮件客服代理（代码直接调用）

```python
import asyncio
from agent.graph import graph

async def run():
    result = await graph.ainvoke({
        "email_content": "我的账户被重复扣费了",
        "sender_email": "user@example.com",
        "email_id": "001",
        "messages": [],
    })
    print(result)

asyncio.run(run())
```

> 注意：`human_review` 节点会调用 `interrupt()` 暂停执行，需要通过 `Command(resume=...)` 恢复。直接调用 `ainvoke` 会在中断处阻塞。

### 三、运行教程示例

```bash
# 运行全部 5 个教程
python src/tutorials/run_examples.py

# 调试单个工作流
python src/tutorials/debug_single.py --workflow prompt_chaining
python src/tutorials/debug_single.py --workflow parallelization
python src/tutorials/debug_single.py --workflow routing
python src/tutorials/debug_single.py --workflow evaluator_optimizer
python src/tutorials/debug_single.py --workflow agent_with_tools
python src/tutorials/debug_single.py --workflow all
```

在 Python 代码中直接调用：

```python
import asyncio
from tutorials.prompt_chaining import graph as chaining_graph
from tutorials.parallelization import graph as parallel_graph
from tutorials.routing import graph as routing_graph
from tutorials.evaluator_optimizer import graph as eval_graph
from tutorials.agent_with_tools import graph as agent_graph

async def demo():
    # 顺序工作流
    r1 = await chaining_graph.ainvoke({"topic": "太空探索", "messages": []})

    # 并行化
    r2 = await parallel_graph.ainvoke({"text": "今天天气真好...", "messages": []})

    # 条件路由
    r3 = await routing_graph.ainvoke({"user_input": "帮我写个排序算法", "messages": []})

    # 评估-优化循环
    r4 = await eval_graph.ainvoke({"task": "写一首关于AI的诗", "messages": []})

    # 带工具的代理
    r5 = await agent_graph.ainvoke({"messages": ["搜索最新的AI新闻"]})

asyncio.run(demo())
```

### 四、运行测试

```bash
# 单元测试（无需服务器）
python -m pytest tests/unit_tests/ -v

# 集成测试（无需服务器，直接测试编译后的图）
python -m pytest tests/integration_tests/test_graph.py -v

# SDK 流式测试（需要先启动 langgraph dev）
python -m pytest tests/integration_tests/test_sdk_stream.py -v

# 运行所有测试
python -m pytest -v

# 使用 Makefile
make test                          # 单元测试
make integration_tests             # 集成测试
make test TEST_FILE=tests/unit_tests/test_configuration.py  # 指定文件
```

### 五、代码质量检查

```bash
# Lint
ruff check .

# 格式化
ruff format .

# Import 排序
ruff check --select I --fix .

# 类型检查
mypy --strict src/

# 使用 Makefile
make lint
make format
```

---

## 调试指南

### 1. LangGraph Studio 可视化调试

```bash
langgraph dev
```

Studio 提供：
- **图形可视化**：查看节点连接和数据流
- **状态检查**：点击节点查看输入/输出状态
- **编辑历史状态**：修改中间状态后从该点重新运行
- **热重载**：本地代码修改自动生效

### 2. Python 调试器（pdb / ipdb）

在任意节点函数中插入断点：

```python
# src/agent/nodes.py
async def classify_intent(state):
    import ipdb; ipdb.set_trace()  # 断点
    # ... 逻辑代码
```

或使用 `breakpoint()` (Python 3.7+)：

```python
async def classify_intent(state):
    breakpoint()
    # ... 逻辑代码
```

### 3. 日志调试

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在节点中
async def classify_intent(state):
    logging.debug(f"分类输入: {state}")
    # ...
```

### 4. 单步调试教程工作流

```bash
# 使用 debug_single.py 逐步查看每个节点的输出
python src/tutorials/debug_single.py --workflow prompt_chaining
```

### 5. pytest 调试

```bash
# 失败时进入 pdb
python -m pytest tests/ -v --pdb

# 显示 print 输出
python -m pytest tests/ -v -s

# 只运行失败的测试
python -m pytest tests/ -v --lf
```

---

## 各文件详细说明

### `src/agent/` — 邮件客服代理

| 文件 | 说明 |
|------|------|
| `models.py` | 定义 `EmailClassification`（分类结果）、`HumanDecision`（人工审核决策）、`EmailAgentState`（图状态）三个 TypedDict |
| `nodes.py` | 7 个异步节点函数：`read_email` → `classify_intent`（动态路由） → `search_documentation` / `bug_tracking` → `draft_response` → `human_review`（interrupt 中断） → `send_reply`。使用 `Command(goto=...)` 实现动态路由 |
| `graph.py` | 组装 `StateGraph`，添加节点和边，配置 `RetryPolicy`，使用 `MemorySaver` 作为 checkpointer，编译并导出 `graph` |
| `__init__.py` | 从 `graph.py` 导出 `graph` 对象 |

**代理流程**：

```
START → read_email → classify_intent
                         ├─ question/feature → search_documentation → draft_response
                         ├─ bug             → bug_tracking        → draft_response
                         └─ billing/complex → draft_response
                                                           ↓
                                                    human_review (interrupt)
                                                     ├─ approved → send_reply → END
                                                     └─ rejected → END
```

### `src/tutorials/` — 5 种 LangGraph 模式

| 文件 | 模式 | 说明 |
|------|------|------|
| `models.py` | — | 定义 5 种状态类型：`StoryState`, `AnalysisState`, `RouterState`, `EvaluatorState`, `AgentState` |
| `prompt_chaining.py` | 顺序工作流 | `generate_outline` → `generate_characters` → `write_story` → `generate_title` |
| `parallelization.py` | 并行化 | Fan-out 到 4 个分析节点，Fan-in 到 `aggregate_results` |
| `routing.py` | 条件路由 | `classify_input` 后路由到 `code_handler` / `math_handler` / `general_handler` |
| `evaluator_optimizer.py` | 评估-优化 | `generate` ↔ `evaluate` 循环，直到通过或达到最大迭代 |
| `agent_with_tools.py` | 工具代理 | LLM 决定调用 `search` / `calculator` / `wiki`，循环直到无需工具 |
| `run_examples.py` | — | 顺序运行全部 5 个教程的脚本 |
| `debug_single.py` | — | CLI 工具，`--workflow` 参数选择调试哪个工作流 |

### `tests/` — 测试

| 文件 | 说明 |
|------|------|
| `conftest.py` | 定义 `anyio_backend` fixture（asyncio） |
| `unit_tests/test_configuration.py` | 8 个测试：图结构验证、分类模拟函数测试、状态字段验证 |
| `integration_tests/test_graph.py` | 5 个测试：question/bug/billing 完整流程、人工审核拒绝路径、complex 直接草稿 |
| `integration_tests/test_sdk_stream.py` | SDK 流式测试（需 `langgraph dev` 运行，默认跳过） |

### 配置文件

| 文件 | 说明 |
|------|------|
| `pyproject.toml` | 项目元数据、依赖、ruff/mypy 配置 |
| `langgraph.json` | LangGraph Server 配置：图入口 `./src/agent/graph.py:graph`，依赖 `.`（当前目录），环境变量 `.env` |
| `.env` | API 密钥和模型端点配置 |
| `Makefile` | 快捷命令：`test`, `integration_tests`, `lint`, `format` |

---

## 依赖

| 类别 | 包 |
|------|-----|
| 运行时 | `langgraph>=1.0.0`, `python-dotenv>=1.0.1` |
| 开发 | `anyio`, `langgraph-cli[inmem]`, `mypy`, `pytest`, `ruff` |
| Python | `>=3.10` |
