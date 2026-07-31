"""
U14 - Task System（任务系统）
==============================
本文件演示 Claude Code 的 **Task System**：如何将复杂工作分解为可管理的任务单元。

核心概念：
  1. Task System 是协调多步骤工作的核心机制
  2. 每个任务有明确的输入、输出和状态
  3. 任务之间可以有依赖关系（DAG - 有向无环图）
  4. 系统自动解析依赖顺序，按拓扑序执行
  5. 支持失败重试机制

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code 使用 TodoWrite 工具跟踪任务进度：            │
  │                                                          │
  │  TodoWrite(todos=[                                       │
  │      {"id": "1", "content": "准备数据", "status": "..."},│
  │      {"id": "2", "content": "处理数据", "status": "...", │
  │       "dependencies": ["1"]},                            │
  │      {"id": "3", "content": "生成报告", "status": "...", │
  │       "dependencies": ["2"]},                            │
  │  ])                                                      │
  └──────────────────────────────────────────────────────────┘

任务生命周期：
  Created → Pending → Running → Completed
                       ↓
                    Failed → Retrying → Running

依赖解析：
  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
  │ 准备 │ ──→ │ 清洗 │ ──→ │ 分析 │ ──→ │ 验证 │ ──→ │ 报告 │
  │ 数据 │     │ 数据 │     │ 数据 │     │ 结果 │     │      │
  └──────┘     └──────┘     └──────┘     └──────┘     └──────┘

  系统按拓扑序执行：准备 → 清洗 → 分析 → 验证 → 报告

本文件是纯 Python 实现，不依赖 anthropic SDK。
使用 dataclass 和枚举模拟 Claude Code 的任务系统。
"""

import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable


# ══════════════════════════════════════════════════════════════
# 第一部分：任务状态枚举
# ══════════════════════════════════════════════════════════════

class TaskState(str, Enum):
    """
    任务的生命周期状态。

    状态转换：
      CREATED  → PENDING  → RUNNING  → COMPLETED
                          ↓
                       FAILED → RETRYING → RUNNING

    说明：
      - CREATED:   任务刚创建，还未加入执行队列
      - PENDING:   任务已加入队列，等待依赖满足
      - RUNNING:   任务正在执行中
      - COMPLETED: 任务成功完成
      - FAILED:    任务执行失败（且重试次数已用完）
      - RETRYING:  任务失败后正在重试
    """
    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


# ══════════════════════════════════════════════════════════════
# 第二部分：任务定义
# ══════════════════════════════════════════════════════════════

@dataclass
class Task:
    """
    任务的完整定义。

    对应 Claude Code 中 TodoWrite 工具的任务项：
    {
        "id": "task-001",
        "content": "准备数据",
        "status": "pending",
        "dependencies": ["task-000"]
    }

    字段说明：
      - task_id:       任务唯一标识符
      - name:          任务名称（人类可读）
      - task_type:     任务类型，决定使用哪个执行器
      - description:   任务描述
      - input_data:    任务输入参数（字典）
      - output_data:   任务执行结果
      - state:         当前状态
      - error:         错误信息（失败时）
      - dependencies:  依赖的任务 ID 列表
      - max_retries:   最大重试次数
      - retry_count:   当前重试次数
      - created_at:    创建时间戳
      - started_at:    开始执行时间戳
      - completed_at:  完成时间戳
    """
    task_id: str
    name: str
    task_type: str               # "bash" | "print" | "agent" | "edit" | "verify"
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


# ══════════════════════════════════════════════════════════════
# 第三部分：任务执行器
# ══════════════════════════════════════════════════════════════

class TaskExecutor:
    """
    任务执行器的基类。

    执行器是 Task System 的核心组件：
      - 每种 task_type 对应一个执行器
      - 执行器负责实际执行任务逻辑
      - 通过 register_executor() 注册到系统

    Claude Code 中的执行器类型：
      - bash:   执行 shell 命令
      - read:   读取文件
      - write:  写入文件
      - edit:   编辑文件
      - agent:  调用子 Agent
    """

    def execute(self, task: Task) -> Any:
        """
        执行任务。子类必须实现此方法。

        Args:
            task: 要执行的任务

        Returns:
            Any: 任务执行结果

        Raises:
            Exception: 执行失败时抛出异常
        """
        raise NotImplementedError


class BashExecutor(TaskExecutor):
    """
    执行 shell 命令的任务执行器。

    对应 Claude Code 的 Bash 工具：
      - 通过 subprocess 执行命令
      - 捕获 stdout 和 stderr
      - 非零退出码视为失败
    """

    def execute(self, task: Task) -> str:
        import subprocess
        command = task.input_data.get("command", "")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")
        return result.stdout


class PrintExecutor(TaskExecutor):
    """
    演示用的打印执行器。

    不执行实际操作，只打印消息。
    用于演示 Task System 的工作流程。
    """

    def execute(self, task: Task) -> str:
        message = task.input_data.get("message", task.name)
        print(f"    执行任务: {message}")
        return f"完成: {message}"


# ══════════════════════════════════════════════════════════════
# 第四部分：任务系统
# ══════════════════════════════════════════════════════════════

class TaskSystem:
    """
    任务系统：管理和执行任务的 DAG（有向无环图）。

    核心功能：
      1. register_executor() - 注册任务执行器
      2. create_task()       - 创建任务
      3. get_ready_tasks()   - 获取可执行的任务（依赖已满足）
      4. execute_task()      - 执行单个任务（含重试）
      5. run_all()           - 执行所有任务（自动处理依赖顺序）
      6. print_status()      - 打印任务状态

    执行流程（run_all）：
      ① 找到所有就绪的节点（依赖已满足且未执行）
      ② 执行它们
      ③ 重复直到所有节点完成或没有可执行的节点
      ④ 返回执行结果统计

    这个流程类似于 LangGraph 的图执行：
      - get_ready_tasks() = 找到所有入度为 0 的节点
      - execute_task()    = 执行节点
      - run_all()         = 拓扑排序 + 顺序执行
    """

    def __init__(self):
        # 任务字典：task_id → Task
        self.tasks: dict[str, Task] = {}
        # 执行器字典：task_type → TaskExecutor
        self.executors: dict[str, TaskExecutor] = {
            "bash": BashExecutor(),
            "print": PrintExecutor(),
        }
        # 执行日志
        self.execution_log: list[dict] = []

    def register_executor(self, task_type: str, executor: TaskExecutor):
        """
        注册任务执行器。

        通过 task_type 将任务路由到对应的执行器。

        Args:
            task_type: 任务类型标识符
            executor:  执行器实例
        """
        self.executors[task_type] = executor

    def create_task(
        self,
        name: str,
        task_type: str,
        input_data: dict = None,
        dependencies: list[str] = None,
        description: str = "",
    ) -> Task:
        """
        创建一个新任务。

        Args:
            name:         任务名称
            task_type:    任务类型（决定使用哪个执行器）
            input_data:   任务输入参数
            dependencies: 依赖的任务 ID 列表
            description:  任务描述

        Returns:
            Task: 创建的任务对象
        """
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
        """
        获取所有可以执行的任务。

        判断条件：
          1. 任务状态为 CREATED 或 PENDING
          2. 所有依赖任务的状态都是 COMPLETED

        这是 DAG 执行的核心逻辑：
          - 找到所有"入度为 0"的未执行节点
          - 这些节点的依赖已经全部完成
          - 可以安全地并行执行

        Returns:
            list[Task]: 可执行的任务列表
        """
        ready = []
        for task in self.tasks.values():
            # 只考虑 CREATED 或 PENDING 状态的任务
            if task.state not in (TaskState.CREATED, TaskState.PENDING):
                continue

            # 检查所有依赖是否已完成
            deps_met = all(
                self.tasks.get(dep_id, Task("", "", "")).state == TaskState.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_met:
                ready.append(task)

        return ready

    def execute_task(self, task: Task) -> bool:
        """
        执行单个任务。

        执行流程：
          1. 查找对应的执行器
          2. 更新状态为 RUNNING
          3. 调用执行器执行
          4. 成功：更新为 COMPLETED
          5. 失败：重试（如果还有重试次数）或标记为 FAILED

        Args:
            task: 要执行的任务

        Returns:
            bool: 是否执行成功
        """
        # 查找执行器
        executor = self.executors.get(task.task_type)
        if not executor:
            task.state = TaskState.FAILED
            task.error = f"No executor for type '{task.task_type}'"
            return False

        # 更新状态
        task.state = TaskState.RUNNING
        task.started_at = time.time()

        try:
            # 执行任务
            result = executor.execute(task)
            task.output_data = result
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()

            # 记录执行日志
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

            # 检查是否可以重试
            if task.retry_count < task.max_retries:
                # 标记为重试中，然后递归调用
                task.state = TaskState.RETRYING
                return self.execute_task(task)
            else:
                # 重试次数用完，标记为失败
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

        算法：类似拓扑排序
          ① 找到所有就绪的节点（依赖已满足）
          ② 执行它们
          ③ 重复直到所有节点完成或没有可执行的节点
          ④ 返回执行结果统计

        防止无限循环：最多迭代 100 次。

        Returns:
            dict: 执行结果统计
                - total:     总任务数
                - completed: 完成的任务数
                - failed:    失败的任务数
                - log:       执行日志
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

        # 统计结果
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
            state_str = task.state.value.rjust(10)
            print(f"  [{state_str}] {task.name} (依赖: {deps})")
            if task.error:
                print(f"               错误: {task.error}")


# ══════════════════════════════════════════════════════════════
# 第五部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U14 - Task System 任务系统演示")
    print("=" * 60)

    system = TaskSystem()

    # ── 创建有依赖关系的任务 ──────────────────────────────
    # 这 5 个任务形成一条依赖链：
    #   准备数据 → 数据清洗 → 数据分析 → 验证结果 → 生成报告
    print("\n── 创建任务（5 个，形成依赖链）──")

    t1 = system.create_task(
        name="准备数据",
        task_type="print",
        input_data={"message": "从数据库加载原始数据"},
    )
    print(f"  创建: {t1.name} (id: {t1.task_id})")

    t2 = system.create_task(
        name="数据清洗",
        task_type="print",
        input_data={"message": "清理和标准化数据"},
        dependencies=[t1.task_id],  # 依赖 t1
    )
    print(f"  创建: {t2.name} (id: {t2.task_id}, 依赖: {t1.task_id})")

    t3 = system.create_task(
        name="数据分析",
        task_type="print",
        input_data={"message": "执行统计分析"},
        dependencies=[t2.task_id],  # 依赖 t2
    )
    print(f"  创建: {t3.name} (id: {t3.task_id}, 依赖: {t2.task_id})")

    t4 = system.create_task(
        name="验证结果",
        task_type="print",
        input_data={"message": "检查分析结果的正确性"},
        dependencies=[t3.task_id],  # 依赖 t3
    )
    print(f"  创建: {t4.name} (id: {t4.task_id}, 依赖: {t3.task_id})")

    t5 = system.create_task(
        name="生成报告",
        task_type="print",
        input_data={"message": "生成最终报告"},
        dependencies=[t4.task_id],  # 依赖 t4
    )
    print(f"  创建: {t5.name} (id: {t5.task_id}, 依赖: {t4.task_id})")

    print(f"\n  共创建 {len(system.tasks)} 个任务")

    # ── 执行前状态 ────────────────────────────────────────
    print("\n── 任务状态（执行前）──")
    system.print_status()

    # ── 执行所有任务 ──────────────────────────────────────
    print("\n── 执行任务（按依赖顺序）──")
    result = system.run_all()

    # ── 执行后状态 ────────────────────────────────────────
    print("\n── 任务状态（执行后）──")
    system.print_status()

    # ── 执行摘要 ──────────────────────────────────────────
    print(f"\n── 执行摘要 ──")
    print(f"  总任务: {result['total']}")
    print(f"  完成:   {result['completed']}")
    print(f"  失败:   {result['failed']}")

    # ── Claude Code 说明 ──────────────────────────────────
    print("\n── Claude Code Task System 机制说明 ──")
    print("""
    Claude Code 使用 TodoWrite 工具跟踪多步骤任务：

    1. 创建任务列表：
       TodoWrite(todos=[
           {"id": "1", "content": "准备数据", "status": "pending"},
           {"id": "2", "content": "处理数据", "status": "pending",
            "dependencies": ["1"]},
       ])

    2. 任务状态更新：
       - pending    → 进行中
       - completed  → 已完成
       - in_progress → 正在执行

    3. Agent 可以：
       - 创建任务计划
       - 按依赖顺序执行
       - 跟踪进度
       - 处理失败和重试
    """)
