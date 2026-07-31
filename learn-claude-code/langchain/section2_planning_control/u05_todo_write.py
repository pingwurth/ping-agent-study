"""
U05 - TodoWrite（任务管理）
===========================
本文件演示 **TodoWrite** 工具：如何在 Agent 运行中管理任务列表。
使用 LangChain @tool 装饰器定义，LangGraph State 管理状态。

核心概念：
  1. TodoWrite 让 Agent 能够将复杂任务分解为可跟踪的子任务
  2. 每个任务有三种状态：pending（待处理）、in_progress（进行中）、completed（完成）
  3. Agent 可以动态添加、更新、删除任务
  4. 任务列表帮助 Agent 保持方向感，避免在长对话中迷失

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  将 todo 状态嵌入 LangGraph 的 State 中：                 │
  │                                                          │
  │  class AgentState(TypedDict):                            │
  │      messages: list                                      │
  │      todos: list[dict]  ← 任务列表作为状态的一部分        │
  │                                                          │
  │  TodoWrite 工具通过修改 state["todos"] 来更新任务列表     │
  └──────────────────────────────────────────────────────────┘
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

model = get_model()


# ── 任务状态枚举 ──────────────────────────────────────────
class TaskStatus(str, Enum):
    """任务的三种状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


# ── 任务数据结构 ──────────────────────────────────────────
@dataclass
class Task:
    """
    单个任务的表示。

    - content: 任务描述（做什么）
    - status: 任务状态
    - activeForm: 进行中时的动词形式（显示在状态行上）
    """
    content: str
    status: TaskStatus = TaskStatus.PENDING
    activeForm: str = ""


# ── 任务管理器 ────────────────────────────────────────────
class TodoManager:
    """
    任务列表管理器。

    对应 Claude Code 的 TodoWrite 工具的功能：
      - 创建新任务列表
      - 更新任务状态
      - 查询当前任务进度
      - 将任务列表格式化为 system prompt 的一部分

    在 LangGraph 中，TodoManager 的状态可以嵌入到
    Agent 的 State 中，让模型在每轮思考时都能看到进度。
    """

    def __init__(self):
        self.todos: list[Task] = []

    def update(self, todos: list[dict]) -> str:
        """
        更新整个任务列表（全量更新，不是增量更新）。

        Args:
            todos: 任务列表，每个任务是 {"content": str, "status": str, "activeForm": str}

        Returns:
            str: 更新后的任务进度摘要
        """
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
          [✓] Fix login bug
          [→] Add user profile page
          [ ] Write unit tests
          进度: 1/3 完成
        """
        if not self.todos:
            return "No tasks defined."

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
        """获取当前正在进行的任务。"""
        for task in self.todos:
            if task.status == TaskStatus.IN_PROGRESS:
                return task
        return None

    def get_pending_tasks(self) -> list[Task]:
        """获取所有待处理的任务。"""
        return [t for t in self.todos if t.status == TaskStatus.PENDING]


# ── 全局 TodoManager 实例 ─────────────────────────────────
# 在实际 LangGraph 应用中，todo 状态应嵌入到 State 中
# 这里使用全局实例简化演示
todo_manager = TodoManager()


# ── TodoWrite 工具定义 ───────────────────────────────────
# 使用 LangChain 的 @tool 装饰器
# 模型调用此工具时，LangGraph 自动执行并返回结果
@tool
def todo_write(todos: list[dict]) -> str:
    """Create and manage a structured task list for the current session.

    Use this tool to track progress on multi-step tasks.
    Each todo should have: content (description), status (pending/in_progress/completed),
    and activeForm (present continuous verb form).

    Args:
        todos: List of todo items, each with content, status, and activeForm fields
    """
    return todo_manager.update(todos)


# ── 带 TodoWrite 的 Agent ────────────────────────────────
SYSTEM = f"""You are a coding agent at {os.getcwd()}.
Use tools to solve tasks. Track your progress using the TodoWrite tool.

When given a complex task:
1. First break it down into subtasks using TodoWrite
2. Update task status as you work (pending → in_progress → completed)
3. Mark tasks complete when done

Act, don't explain."""


def run_agent(query: str) -> str:
    """运行带任务管理的 Agent。"""
    agent = create_react_agent(
        model,
        [todo_write],
        prompt=SYSTEM,
    )

    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    last_message = result["messages"][-1]
    return last_message.content


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("TodoWrite 演示\n")

    # 演示任务管理器的基本操作（无需 API 调用）
    print("── Agent 分解任务 ──")
    todo_manager.update([
        {"content": "分析需求文档", "status": "completed", "activeForm": "分析需求文档"},
        {"content": "设计数据库 schema", "status": "in_progress", "activeForm": "设计数据库 schema"},
        {"content": "实现 API 接口", "status": "pending", "activeForm": "实现 API 接口"},
        {"content": "编写前端页面", "status": "pending", "activeForm": "编写前端页面"},
        {"content": "编写测试用例", "status": "pending", "activeForm": "编写测试用例"},
    ])
    print(todo_manager.get_progress())

    print("\n── Agent 更新进度 ──")
    todo_manager.update([
        {"content": "分析需求文档", "status": "completed", "activeForm": "分析需求文档"},
        {"content": "设计数据库 schema", "status": "completed", "activeForm": "设计数据库 schema"},
        {"content": "实现 API 接口", "status": "in_progress", "activeForm": "实现 API 接口"},
        {"content": "编写前端页面", "status": "pending", "activeForm": "编写前端页面"},
        {"content": "编写测试用例", "status": "pending", "activeForm": "编写测试用例"},
    ])
    print(todo_manager.get_progress())

    print("\n── 当前任务 ──")
    active = todo_manager.get_active_task()
    if active:
        print(f"正在进行: {active.content}")
    print(f"待处理: {len(todo_manager.get_pending_tasks())} 个任务")

    print("\n── 交互模式 ──")
    print("输入问题让 Agent 使用 TodoWrite 管理任务。输入 q 退出。\n")

    while True:
        try:
            query = input("\033[036mu05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit"):
            print("bye!")
            break

        try:
            response = run_agent(query)
            print(response)
            print(f"\n当前进度:\n{todo_manager.get_progress()}")
        except Exception as e:
            print(f"Error: {e}")
        print()
