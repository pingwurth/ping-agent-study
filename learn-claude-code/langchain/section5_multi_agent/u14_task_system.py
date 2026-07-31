"""
U14 - Task System（任务系统）
==============================
本文件演示 **Task System**：如何将复杂工作分解为可管理的任务单元。
使用 LangGraph StateGraph 实现任务 DAG。

核心概念：
  1. Task System 是协调多步骤工作的核心机制
  2. 每个任务有明确的输入、输出和状态
  3. 任务可以串行或并行执行
  4. 任务之间可以有依赖关系

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  每个任务 = LangGraph 图中的一个节点                      │
  │  依赖关系 = 节点之间的边                                  │
  │                                                          │
  │  graph.add_node("prepare", prepare_data)                 │
  │  graph.add_node("process", process_data)                 │
  │  graph.add_edge("prepare", "process")  # process 依赖    │
  │                                                          │
  │  LangGraph 自动处理依赖顺序和并行执行                     │
  └──────────────────────────────────────────────────────────┘

任务生命周期：
  Created → Pending → Running → Completed/Failed
                       ↓
                    (retry on failure)
"""

import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable


# ── 任务状态 ──────────────────────────────────────────────
class TaskState(str, Enum):
    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


# ── 任务定义 ──────────────────────────────────────────────
@dataclass
class Task:
    """
    任务的完整定义。

    Claude Code 中的 Task 包含：
      - 唯一标识符
      - 任务类型和描述
      - 输入参数
      - 执行结果
      - 状态信息
      - 依赖关系
      - 重试配置
    """
    task_id: str
    name: str
    task_type: str               # "bash" | "agent" | "edit" | "verify"
    description: str = ""
    input_data: dict = field(default_factory=dict)
    output_data: Any = None
    state: TaskState = TaskState.CREATED
    error: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    max_retries: int = 2
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


# ── 任务执行器接口 ────────────────────────────────────────
class TaskExecutor:
    """任务执行器的基类。"""

    def execute(self, task: Task) -> Any:
        raise NotImplementedError


class BashExecutor(TaskExecutor):
    """执行 shell 命令的任务执行器。"""

    def execute(self, task: Task) -> str:
        import subprocess
        command = task.input_data.get("command", "")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")
        return result.stdout


class PrintExecutor(TaskExecutor):
    """演示用的打印执行器。"""

    def execute(self, task: Task) -> str:
        message = task.input_data.get("message", task.name)
        print(f"  执行任务: {message}")
        return f"完成: {message}"


# ── 任务系统 ──────────────────────────────────────────────
class TaskSystem:
    """
    任务系统：管理和执行任务。

    Claude Code 的任务系统提供：
      - 任务创建和管理
      - 依赖解析
      - 并行执行
      - 失败重试
      - 进度跟踪

    与 LangGraph 的对应关系：
      TaskSystem.run_all()  →  StateGraph.compile().invoke()
      依赖关系              →  图的边
      并行执行              →  LangGraph 的并行节点
    """

    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.executors: dict[str, TaskExecutor] = {
            "bash": BashExecutor(),
            "print": PrintExecutor(),
        }
        self.execution_log: list[dict] = []

    def register_executor(self, task_type: str, executor: TaskExecutor):
        """注册任务执行器。"""
        self.executors[task_type] = executor

    def create_task(
        self,
        name: str,
        task_type: str,
        input_data: dict = None,
        dependencies: list[str] = None,
        description: str = "",
    ) -> Task:
        """创建一个新任务。"""
        task = Task(
            task_id=str(uuid.uuid4())[:8],
            name=name,
            task_type=task_type,
            description=description,
            input_data=input_data or {},
            dependencies=dependencies or [],
        )
        self.tasks[task.task_id] = task
        return task

    def get_ready_tasks(self) -> list[Task]:
        """获取所有可以执行的任务（依赖已满足）。"""
        ready = []
        for task in self.tasks.values():
            if task.state not in (TaskState.CREATED, TaskState.PENDING):
                continue

            deps_met = all(
                self.tasks.get(dep_id, Task("", "", "")).state == TaskState.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_met:
                ready.append(task)

        return ready

    def execute_task(self, task: Task) -> bool:
        """执行单个任务。"""
        executor = self.executors.get(task.task_type)
        if not executor:
            task.state = TaskState.FAILED
            task.error = f"No executor for type '{task.task_type}'"
            return False

        task.state = TaskState.RUNNING
        task.started_at = time.time()

        try:
            result = executor.execute(task)
            task.output_data = result
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()

            self.execution_log.append({
                "task_id": task.task_id,
                "name": task.name,
                "state": "completed",
                "duration": task.completed_at - task.started_at,
            })
            return True

        except Exception as e:
            task.error = str(e)
            task.retry_count += 1

            if task.retry_count < task.max_retries:
                task.state = TaskState.RETRYING
                return self.execute_task(task)
            else:
                task.state = TaskState.FAILED
                task.completed_at = time.time()

                self.execution_log.append({
                    "task_id": task.task_id,
                    "name": task.name,
                    "state": "failed",
                    "error": str(e),
                })
                return False

    def run_all(self) -> dict:
        """
        执行所有任务（自动处理依赖顺序）。

        这个方法的逻辑与 LangGraph 的图执行类似：
          ① 找到所有就绪的节点（依赖已满足）
          ② 执行它们（可以并行）
          ③ 重复直到所有节点完成
        """
        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            ready = self.get_ready_tasks()
            if not ready:
                break

            for task in ready:
                task.state = TaskState.PENDING
                self.execute_task(task)

            iteration += 1

        completed = sum(1 for t in self.tasks.values() if t.state == TaskState.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.state == TaskState.FAILED)

        return {
            "total": len(self.tasks),
            "completed": completed,
            "failed": failed,
            "log": self.execution_log,
        }

    def print_status(self):
        """打印所有任务的状态。"""
        for task in self.tasks.values():
            deps = ", ".join(task.dependencies) if task.dependencies else "无"
            print(f"  [{task.state.value:10s}] {task.name} (依赖: {deps})")
            if task.error:
                print(f"             错误: {task.error}")


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Task System 演示\n")

    system = TaskSystem()

    # 创建有依赖关系的任务
    print("── 创建任务 ──")

    t1 = system.create_task(
        name="准备数据",
        task_type="print",
        input_data={"message": "从数据库加载原始数据"},
    )

    t2 = system.create_task(
        name="数据清洗",
        task_type="print",
        input_data={"message": "清理和标准化数据"},
        dependencies=[t1.task_id],
    )

    t3 = system.create_task(
        name="数据分析",
        task_type="print",
        input_data={"message": "执行统计分析"},
        dependencies=[t2.task_id],
    )

    t4 = system.create_task(
        name="验证结果",
        task_type="print",
        input_data={"message": "检查分析结果的正确性"},
        dependencies=[t3.task_id],
    )

    t5 = system.create_task(
        name="生成报告",
        task_type="print",
        input_data={"message": "生成最终报告"},
        dependencies=[t4.task_id],
    )

    print(f"  创建了 {len(system.tasks)} 个任务\n")

    print("── 任务状态（执行前）──")
    system.print_status()

    print("\n── 执行任务 ──")
    result = system.run_all()

    print("\n── 任务状态（执行后）──")
    system.print_status()

    print(f"\n── 执行摘要 ──")
    print(f"  总任务: {result['total']}")
    print(f"  完成: {result['completed']}")
    print(f"  失败: {result['failed']}")
