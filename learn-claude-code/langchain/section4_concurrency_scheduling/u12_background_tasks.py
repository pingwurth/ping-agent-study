"""
U12 - Background Tasks（后台任务）
===================================
本文件演示 **后台任务** 机制：如何在后台执行长时间运行的任务。
使用 Python asyncio 实现异步任务管理。

核心概念：
  1. 后台任务允许 Agent 在等待任务完成的同时继续工作
  2. 适用于长时间运行的命令（如构建、测试、部署）
  3. 任务在后台运行，完成后通过回调机制告知 Agent

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  使用 Python asyncio 管理后台任务：                       │
  │                                                          │
  │  async def run_background(task):                         │
  │      result = await agent.ainvoke(...)                   │
  │      return result                                       │
  │                                                          │
  │  LangGraph 的 agent.ainvoke() 支持异步执行               │
  │  可以与 asyncio.gather() 结合并行执行多个任务            │
  └──────────────────────────────────────────────────────────┘

后台任务 vs 前台任务：
  ┌──────────────────────────────────────────────────────────┐
  │  前台任务（Foreground）                                   │
  │  - Agent 等待任务完成                                     │
  │  - 阻塞当前对话                                          │
  │  - 适用于快速命令（< 2 分钟）                              │
  ├──────────────────────────────────────────────────────────┤
  │  后台任务（Background）                                   │
  │  - 任务在后台运行                                        │
  │  - Agent 可以继续其他工作                                  │
  │  - 完成后通过回调/通知机制告知                             │
  │  - 适用于长时间命令（> 2 分钟）                            │
  └──────────────────────────────────────────────────────────┘
"""

import os
import time
import uuid
import asyncio
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any


# ── 任务状态 ──────────────────────────────────────────────
class TaskState(str, Enum):
    """后台任务的状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# ── 后台任务数据结构 ──────────────────────────────────────
@dataclass
class BackgroundTask:
    """
    后台任务的表示。

    Claude Code 中后台任务的信息：
      - task_id: 唯一标识符
      - command: 执行的命令
      - state: 当前状态
      - output: 任务输出
      - exit_code: 退出码
    """
    task_id: str
    command: str
    state: TaskState = TaskState.PENDING
    output: str = ""
    exit_code: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False)


# ── 后台任务管理器 ────────────────────────────────────────
class BackgroundTaskManager:
    """
    管理后台任务的执行和生命周期。

    Claude Code 的 Bash tool 支持 run_in_background 参数：
    {
        "name": "Bash",
        "input": {
            "command": "npm run build",
            "run_in_background": true
        }
    }

    支持两种执行模式：
      1. 线程模式（threading）：适合 subprocess 命令
      2. 异步模式（asyncio）：适合 LangGraph Agent 调用
    """

    def __init__(self):
        self.tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()
        self._callbacks: dict[str, Callable] = {}

    def submit(
        self,
        command: str,
        callback: Callable[[BackgroundTask], None] = None,
    ) -> str:
        """
        提交一个后台任务（线程模式）。

        Args:
            command: 要执行的命令
            callback: 任务完成时的回调函数

        Returns:
            str: 任务 ID
        """
        task_id = str(uuid.uuid4())[:8]
        task = BackgroundTask(task_id=task_id, command=command)

        with self._lock:
            self.tasks[task_id] = task

        if callback:
            self._callbacks[task_id] = callback

        thread = threading.Thread(
            target=self._run_task,
            args=(task,),
            daemon=True,
        )
        thread.start()

        return task_id

    def _run_task(self, task: BackgroundTask):
        """在后台线程中执行任务。"""
        task.state = TaskState.RUNNING
        task.start_time = time.time()

        try:
            process = subprocess.Popen(
                task.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            task.process = process

            stdout, _ = process.communicate(timeout=300)
            task.output = stdout
            task.exit_code = process.returncode

            task.state = TaskState.COMPLETED if process.returncode == 0 else TaskState.FAILED

        except subprocess.TimeoutExpired:
            task.state = TaskState.FAILED
            task.output = "Error: Task timed out after 300 seconds"
            task.exit_code = -1
            if task.process:
                task.process.kill()

        except Exception as e:
            task.state = TaskState.FAILED
            task.output = f"Error: {e}"
            task.exit_code = -1

        finally:
            task.end_time = time.time()

        callback = self._callbacks.get(task.task_id)
        if callback:
            callback(task)

    def get_status(self, task_id: str) -> Optional[BackgroundTask]:
        """获取任务状态。"""
        return self.tasks.get(task_id)

    def get_output(self, task_id: str) -> str:
        """获取任务输出。"""
        task = self.tasks.get(task_id)
        if not task:
            return f"Error: task '{task_id}' not found"
        return task.output

    def stop(self, task_id: str) -> str:
        """停止一个正在运行的任务。"""
        task = self.tasks.get(task_id)
        if not task:
            return f"Error: task '{task_id}' not found"

        if task.state == TaskState.RUNNING and task.process:
            task.process.kill()
            task.state = TaskState.STOPPED
            task.end_time = time.time()
            return f"Task '{task_id}' stopped"

        return f"Task '{task_id}' is not running (state: {task.state})"

    def list_tasks(self) -> list[BackgroundTask]:
        """列出所有任务。"""
        return list(self.tasks.values())

    def wait_for(self, task_id: str, timeout: float = 300) -> BackgroundTask:
        """等待任务完成。"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task '{task_id}' not found")

        start = time.time()
        while task.state in (TaskState.PENDING, TaskState.RUNNING):
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for task '{task_id}'")
            time.sleep(0.5)

        return task


# ── 异步任务管理器（LangGraph 集成）──────────────────────
class AsyncTaskManager:
    """
    异步任务管理器，适合与 LangGraph Agent 集成。

    使用 asyncio 管理后台任务，支持：
      - agent.ainvoke() 异步调用
      - asyncio.gather() 并行执行
      - 回调通知
    """

    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """获取或创建事件循环。"""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    async def submit_async(
        self,
        coro,
        task_id: str = None,
        callback: Callable = None,
    ) -> str:
        """
        提交一个异步任务。

        Args:
            coro: 协程对象（如 agent.ainvoke(...)）
            task_id: 任务 ID（可选，自动生成）
            callback: 完成回调

        Returns:
            str: 任务 ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        self.tasks[task_id] = {
            "task_id": task_id,
            "state": TaskState.RUNNING,
            "start_time": time.time(),
            "result": None,
            "error": None,
        }

        try:
            result = await coro
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["state"] = TaskState.COMPLETED
        except Exception as e:
            self.tasks[task_id]["error"] = str(e)
            self.tasks[task_id]["state"] = TaskState.FAILED
        finally:
            self.tasks[task_id]["end_time"] = time.time()

        if callback:
            callback(self.tasks[task_id])

        return task_id

    async def gather(self, coros: list) -> list:
        """并行执行多个协程。"""
        return await asyncio.gather(*coros, return_exceptions=True)

    def run_async(self, coro):
        """在同步代码中运行异步任务。"""
        loop = self._get_loop()
        return loop.run_until_complete(coro)


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("后台任务演示\n")

    manager = BackgroundTaskManager()

    # 提交后台任务
    print("── 提交后台任务 ──")

    task1_id = manager.submit(
        "echo 'Hello from background task 1' && sleep 1 && echo 'Task 1 done'",
        callback=lambda t: print(f"\n  [回调] 任务 {t.task_id} 完成: {t.state}")
    )
    print(f"  任务 1 已提交: {task1_id}")

    task2_id = manager.submit(
        "echo 'Starting task 2' && sleep 2 && echo 'Task 2 done'",
    )
    print(f"  任务 2 已提交: {task2_id}")

    # 查看任务状态
    print("\n── 任务状态 ──")
    for task in manager.list_tasks():
        print(f"  [{task.task_id}] {task.state.value}: {task.command[:50]}...")

    # 等待任务完成
    print("\n── 等待任务完成 ──")
    completed = manager.wait_for(task1_id)
    print(f"  任务 1 输出:\n{completed.output}")

    print(f"\n── 任务 2 输出 ──")
    task2 = manager.wait_for(task2_id)
    print(f"  {task2.output}")
    print(f"  耗时: {task2.end_time - task2.start_time:.1f} 秒")

    # 演示异步任务管理器
    print("\n── 异步任务管理器（LangGraph 集成）──")
    print("  AsyncTaskManager 支持 agent.ainvoke() 异步调用")
    print("  可以与 asyncio.gather() 结合并行执行多个 Agent 任务")
