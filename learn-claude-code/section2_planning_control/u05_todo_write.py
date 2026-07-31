"""
U05 - TodoWrite（任务管理工具）
================================
本文件演示 **TodoWrite** 工具：如何在 Agent 运行中管理任务列表。
使用原生 Anthropic SDK 实现。

核心概念：
  1. TodoWrite 让 Agent 能够将复杂任务分解为可跟踪的子任务
  2. 每个任务有三种状态：pending（待处理）、in_progress（进行中）、completed（完成）
  3. Agent 可以动态添加、更新、删除任务
  4. 任务列表帮助 Agent 保持方向感，避免在长对话中迷失

实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  TodoWrite 工具的 JSON Schema 定义：                      │
  │                                                          │
  │  {                                                       │
  │    "name": "TodoWrite",                                  │
  │    "input_schema": {                                     │
  │      "type": "object",                                   │
  │      "properties": {                                     │
  │        "todos": {                                        │
  │          "type": "array",                                │
  │          "items": { ... }                                │
  │        }                                                 │
  │      }                                                   │
  │    }                                                     │
  │  }                                                       │
  │                                                          │
  │  Agent 调用此工具 → 后端更新任务列表 → 返回进度摘要       │
  └──────────────────────────────────────────────────────────┘

为什么需要 TodoWrite？
  - 长对话中 Agent 容易"忘记"自己在做什么
  - 任务列表提供结构化的进度追踪
  - 用户可以看到 Agent 的工作进展
  - Agent 可以根据进度动态调整策略
"""

import os
import sys
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ============================================================
# 初始化客户端
# ============================================================
client, MODEL = create_client()


# ════════════════════════════════════════════════════════════
# 第一部分：数据结构定义
# ════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    """
    任务状态枚举。

    对应 Claude Code 中任务的三种状态：
      - PENDING:     等待执行，还没有开始
      - IN_PROGRESS: 正在执行中
      - COMPLETED:   已经完成

    继承 str 使得枚举值可以直接用于 JSON 序列化。
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Task:
    """
    单个任务的数据结构。

    字段说明：
      - content:     任务描述文本，说明要做什么
      - status:      当前状态（pending / in_progress / completed）
      - activeForm:  进行中时的动词形式，用于显示在状态行上
                     例如 content="分析需求文档" → activeForm="正在分析需求文档"

    在实际 Claude Code 中，activeForm 会显示在终端的状态栏上，
    让用户知道 Agent 当前在做什么。
    """
    content: str
    status: TaskStatus = TaskStatus.PENDING
    activeForm: str = ""


# ════════════════════════════════════════════════════════════
# 第二部分：TodoManager - 任务管理器
# ════════════════════════════════════════════════════════════

class TodoManager:
    """
    任务列表管理器。

    对应 Claude Code 的 TodoWrite 工具的核心功能：
      - update():          全量更新任务列表（替换旧列表）
      - get_progress():    获取格式化的进度摘要
      - get_active_task(): 获取当前正在进行的任务
      - get_pending_tasks(): 获取所有待处理的任务

    为什么是全量更新而不是增量更新？
      全量更新更简单可靠。Agent 每次调用 TodoWrite 时，
      传入完整的任务列表，避免了"在哪个位置插入/删除"的复杂性。
      Agent 本身就能看到当前的任务列表，所以它可以精确地
      构造更新后的完整列表。
    """

    def __init__(self):
        self.todos: list[Task] = []

    def update(self, todos: list[dict]) -> str:
        """
        全量更新任务列表。

        这是 TodoWrite 工具的核心方法。Agent 通过调用此方法
        来创建新任务、更新任务状态、或删除任务。

        Args:
            todos: 任务列表，每个任务是一个字典
                   {"content": str, "status": str, "activeForm": str}

        Returns:
            str: 更新后的任务进度摘要，会作为工具调用的返回值
                 传回给 Agent，让 Agent 知道当前进度
        """
        # 用新的任务列表替换旧的（全量替换）
        self.todos = []
        for t in todos:
            self.todos.append(Task(
                content=t.get("content", ""),
                status=TaskStatus(t.get("status", "pending")),
                activeForm=t.get("activeForm", ""),
            ))
        return self.get_progress()

    def get_progress(self) -> str:
        """
        获取当前任务进度摘要。

        格式示例：
          [✓] 分析需求文档        ← 已完成
          [→] 设计数据库 schema   ← 进行中
          [ ] 实现 API 接口       ← 待处理
          [ ] 编写前端页面        ← 待处理

          进度: 1/4 完成

        符号说明：
          [✓] = completed（已完成）
          [→] = in_progress（进行中）
          [ ] = pending（待处理）
        """
        if not self.todos:
            return "暂无任务。"

        lines = []
        for task in self.todos:
            if task.status == TaskStatus.COMPLETED:
                lines.append(f"  [✓] {task.content}")
            elif task.status == TaskStatus.IN_PROGRESS:
                lines.append(f"  [→] {task.content}")
            else:
                lines.append(f"  [ ] {task.content}")

        completed = sum(1 for t in self.todos if t.status == TaskStatus.COMPLETED)
        total = len(self.todos)
        lines.append(f"\n  进度: {completed}/{total} 完成")

        return "\n".join(lines)

    def get_active_task(self) -> Optional[Task]:
        """
        获取当前正在进行的任务。

        Returns:
            Task 或 None：如果没有任何任务处于 in_progress 状态，返回 None
        """
        for task in self.todos:
            if task.status == TaskStatus.IN_PROGRESS:
                return task
        return None

    def get_pending_tasks(self) -> list[Task]:
        """
        获取所有待处理的任务。

        Returns:
            list[Task]: 所有状态为 PENDING 的任务列表
        """
        return [t for t in self.todos if t.status == TaskStatus.PENDING]


# ════════════════════════════════════════════════════════════
# 第三部分：TodoWrite 工具定义（JSON Schema）
# ════════════════════════════════════════════════════════════

# Anthropic SDK 使用 JSON Schema 来定义工具的输入参数。
# 这与 OpenAI 的 function calling 格式类似，但有一些区别：
#   - 使用 "input_schema" 而不是 "parameters"
#   - Schema 格式更严格（必须是 object 类型）
#
# 工具定义的三个关键字段：
#   - name:         工具名称，Agent 通过此名称调用工具
#   - description:  工具描述，帮助模型理解何时以及如何使用此工具
#   - input_schema: 输入参数的 JSON Schema 定义

TODO_WRITE_TOOL = {
    "name": "TodoWrite",
    "description": """创建和管理当前会话的结构化任务列表。

使用此工具来跟踪多步骤任务的进度。
每个任务需要包含：content（描述）、status（pending/in_progress/completed）、
activeForm（进行中的动词形式）。

使用场景：
  - 接到复杂任务时，先分解为子任务
  - 开始某项工作时，更新状态为 in_progress
  - 完成某项工作时，更新状态为 completed
  - 用户随时可以看到工作进度""",
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "任务列表，全量替换（不是增量更新）",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "任务描述（做什么）"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "任务状态"
                        },
                        "activeForm": {
                            "type": "string",
                            "description": "进行中时的动词形式，如 '正在分析需求'"
                        }
                    },
                    "required": ["content", "status"]
                }
            }
        },
        "required": ["todos"]
    }
}


# ════════════════════════════════════════════════════════════
# 第四部分：Agent 循环（带 TodoWrite 工具）
# ════════════════════════════════════════════════════════════

# 全局 TodoManager 实例
# 在实际应用中，每个会话应该有独立的 TodoManager
todo_manager = TodoManager()


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    处理工具调用请求。

    当 Agent 决定调用 TodoWrite 工具时，Claude SDK 会返回
    一个 tool_use 内容块。我们需要：
      1. 解析工具名称和输入参数
      2. 执行对应的逻辑
      3. 将结果作为 tool_result 返回

    Args:
        tool_name: 工具名称（如 "TodoWrite"）
        tool_input: 工具输入参数（已解析为字典）

    Returns:
        str: 工具执行结果，将作为 tool_result 发回给模型
    """
    if tool_name == "TodoWrite":
        todos = tool_input.get("todos", [])
        return todo_manager.update(todos)
    return f"Unknown tool: {tool_name}"


def run_agent(query: str) -> str:
    """
    运行带 TodoWrite 工具的 Agent。

    Agent 循环的核心流程：
      1. 将用户消息和工具定义发送给模型
      2. 模型返回响应（可能是文本，也可能是工具调用）
      3. 如果是工具调用 → 执行工具 → 将结果发回模型 → 回到步骤 2
      4. 如果是文本 → 返回给用户

    这个循环就是所谓的 "agentic loop"（代理循环）。
    模型可以多次调用工具，直到它决定给出最终回答。

    Args:
        query: 用户的输入问题

    Returns:
        str: Agent 的最终回答
    """
    messages = [{"role": "user", "content": query}]

    while True:
        # 调用 Claude API，传入消息和工具定义
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=f"""你是一个编程助手，工作在 {os.getcwd()} 目录。
使用 TodoWrite 工具来管理任务进度。
接到复杂任务时：
1. 先用 TodoWrite 分解为子任务
2. 工作过程中更新任务状态（pending → in_progress → completed）
3. 完成后标记任务为 completed

用中文回复。""",
            tools=[TODO_WRITE_TOOL],
            messages=messages,
        )

        # 检查模型是否要调用工具
        # stop_reason == "tool_use" 表示模型想要调用工具
        if response.stop_reason == "tool_use":
            # 收集所有文本内容和工具调用
            tool_results = []
            assistant_content = response.content

            for block in response.content:
                if block.type == "tool_use":
                    # 执行工具调用
                    result = handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # 将助手消息和工具结果都加入对话历史
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
            # 继续循环，让模型处理工具结果
            continue

        # stop_reason == "end_turn" 表示模型给出了最终回答
        # 提取文本内容
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        return final_text


# ════════════════════════════════════════════════════════════
# 第五部分：程序入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  U05 - TodoWrite（任务管理工具）演示")
    print("=" * 60)

    # ── 演示 1：基本任务管理（无需 API 调用）──
    print("\n── 演示 1：Agent 分解任务 ──\n")
    print("模拟场景：用户要求 Agent 开发一个 Web 应用\n")

    # Agent 接到任务后，首先分解为子任务
    todo_manager.update([
        {"content": "分析需求文档", "status": "completed", "activeForm": "正在分析需求文档"},
        {"content": "设计数据库 schema", "status": "in_progress", "activeForm": "正在设计数据库 schema"},
        {"content": "实现 API 接口", "status": "pending", "activeForm": ""},
        {"content": "编写前端页面", "status": "pending", "activeForm": ""},
        {"content": "编写测试用例", "status": "pending", "activeForm": ""},
    ])
    print("Agent 创建了任务列表：")
    print(todo_manager.get_progress())

    # ── 演示 2：更新进度 ──
    print("\n\n── 演示 2：Agent 更新进度 ──\n")
    print("Agent 完成了数据库设计，开始实现 API...\n")

    todo_manager.update([
        {"content": "分析需求文档", "status": "completed", "activeForm": ""},
        {"content": "设计数据库 schema", "status": "completed", "activeForm": ""},
        {"content": "实现 API 接口", "status": "in_progress", "activeForm": "正在实现 API 接口"},
        {"content": "编写前端页面", "status": "pending", "activeForm": ""},
        {"content": "编写测试用例", "status": "pending", "activeForm": ""},
    ])
    print(todo_manager.get_progress())

    # ── 演示 3：查询当前任务 ──
    print("\n\n── 演示 3：查询当前任务状态 ──\n")

    active = todo_manager.get_active_task()
    if active:
        print(f"当前正在进行: {active.content}")
        print(f"  activeForm: {active.activeForm}")

    pending = todo_manager.get_pending_tasks()
    print(f"待处理任务: {len(pending)} 个")
    for t in pending:
        print(f"  - {t.content}")

    # ── 演示 4：JSON Schema 展示 ──
    print("\n\n── 演示 4：TodoWrite 工具的 JSON Schema ──\n")
    print("这是传递给 Claude API 的工具定义：\n")
    print(json.dumps(TODO_WRITE_TOOL, indent=2, ensure_ascii=False))

    # ── 交互模式 ──
    print("\n\n" + "=" * 60)
    print("  交互模式")
    print("  输入任务让 Agent 使用 TodoWrite 管理进度")
    print("  输入 q 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\033[36mu05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit"):
            print("bye!")
            break

        try:
            # 重置任务列表
            todo_manager = TodoManager()
            response = run_agent(query)
            print(response)
            print(f"\n当前进度:\n{todo_manager.get_progress()}")
        except Exception as e:
            print(f"Error: {e}")
        print()
