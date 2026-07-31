"""
U12 - Background Tasks（后台任务）
===================================
本文件演示 Claude Code 的 **后台任务** 机制：如何在后台执行长时间运行的命令。

核心概念：
  1. Claude Code 的 Bash 工具支持 run_in_background 参数
  2. 后台任务允许 Agent 在等待任务完成的同时继续工作
  3. 适用于长时间运行的命令（如构建、测试、部署）
  4. 任务在后台运行，完成后通过回调机制通知

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  当用户要求运行长时间命令时，Claude Code 会：              │
  │                                                          │
  │  1. 使用 Bash tool 的 run_in_background=true 参数        │
  │  2. 创建子进程在后台执行命令                              │
  │  3. 返回任务 ID 给 Agent                                 │
  │  4. Agent 可以继续处理其他工作                            │
  │  5. 通过 Monitor 工具监听任务输出                         │
  │  6. 任务完成时收到通知                                    │
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

本文件是纯 Python 实现，不依赖 anthropic SDK。
使用 subprocess + threading 模拟 Claude Code 的后台任务机制。
"""

import os
import time
import uuid
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


# ══════════════════════════════════════════════════════════════
# 第一部分：任务状态枚举
# ══════════════════════════════════════════════════════════════

class TaskState(str, Enum):
    """
    后台任务的状态枚举。

    Claude Code 中后台任务的状态转换：
      PENDING → RUNNING → COMPLETED
                   ↓         ↑
                   └→ FAILED ┘
                   ↓
                STOPPED（用户手动停止）

    状态说明：
      - PENDING:   任务已创建，等待执行
      - RUNNING:   任务正在执行中
      - COMPLETED: 任务成功完成（exit_code == 0）
      - FAILED:    任务执行失败（exit_code != 0 或超时）
      - STOPPED:   任务被用户手动停止
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# ══════════════════════════════════════════════════════════════
# 第二部分：后台任务数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class BackgroundTask:
    """
    后台任务的数据表示。

    对应 Claude Code 中 Bash tool 的 run_in_background 模式：
    {
        "name": "Bash",
        "input": {
            "command": "npm run build",
            "run_in_background": true
        }
    }

    字段说明：
      - task_id:     任务的唯一标识符（UUID 前 8 位）
      - command:     要执行的 shell 命令
      - state:       当前任务状态
      - output:      任务的标准输出（合并了 stderr）
      - exit_code:   进程退出码（0 表示成功）
      - start_time:  任务开始时间戳
      - end_time:    任务结束时间戳
      - process:     子进程对象（用于停止任务）
    """
    task_id: str
    command: str
    state: TaskState = TaskState.PENDING
    output: str = ""
    exit_code: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    # process 字段不参与 repr，避免打印子进程对象
    process: Optional[subprocess.Popen] = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════
# 第三部分：后台任务管理器
# ══════════════════════════════════════════════════════════════

class BackgroundTaskManager:
    """
    后台任务管理器：管理后台任务的执行和生命周期。

    模拟 Claude Code 的后台任务机制：
      1. submit()     - 提交后台任务，创建子线程执行
      2. _run_task()  - 在后台线程中通过 subprocess.Popen 执行命令
      3. get_status() - 查询任务状态
      4. get_output() - 获取任务输出
      5. stop()       - 停止正在运行的任务
      6. list_tasks() - 列出所有任务
      7. wait_for()   - 等待任务完成

    Claude Code 的实际行为：
      - Bash tool 设置 run_in_background=true 时，命令在后台执行
      - Agent 可以通过 Monitor 工具监听任务的 stdout 输出
      - 任务完成时会收到通知（包含退出码和输出）
      - Agent 可以使用 TaskStop 工具停止后台任务

    线程安全：
      - 使用 threading.Lock 保护共享状态
      - 每个任务在独立的 daemon 线程中运行
    """

    def __init__(self):
        # 任务字典：task_id → BackgroundTask
        self.tasks: dict[str, BackgroundTask] = {}
        # 线程锁：保护 tasks 字典的并发访问
        self._lock = threading.Lock()
        # 回调函数：任务完成时触发
        self._callbacks: dict[str, Callable] = {}

    def submit(
        self,
        command: str,
        callback: Callable[[BackgroundTask], None] = None,
    ) -> str:
        """
        提交一个后台任务。

        对应 Claude Code 的 Bash tool 调用：
          Bash(command="npm run build", run_in_background=true)

        实现流程：
          1. 生成唯一的 task_id
          2. 创建 BackgroundTask 对象
          3. 注册回调函数（可选）
          4. 创建 daemon 线程执行 _run_task()
          5. 启动线程并返回 task_id

        Args:
            command:  要执行的 shell 命令
            callback: 任务完成时的回调函数，接收 BackgroundTask 参数

        Returns:
            str: 任务的唯一标识符（8 位 UUID）
        """
        # 生成短 UUID 作为任务 ID
        task_id = str(uuid.uuid4())[:8]
        task = BackgroundTask(task_id=task_id, command=command)

        # 线程安全地添加任务到字典
        with self._lock:
            self.tasks[task_id] = task

        # 注册回调函数
        if callback:
            self._callbacks[task_id] = callback

        # 创建 daemon 线程（daemon 线程在主程序退出时自动终止）
        thread = threading.Thread(
            target=self._run_task,
            args=(task,),
            daemon=True,
        )
        thread.start()

        return task_id

    def _run_task(self, task: BackgroundTask):
        """
        在后台线程中执行任务。

        这是实际执行命令的核心方法：
          1. 更新任务状态为 RUNNING
          2. 使用 subprocess.Popen 创建子进程
          3. 合并 stdout 和 stderr
          4. 根据退出码设置 COMPLETED 或 FAILED
          5. 触发回调函数

        异常处理：
          - TimeoutExpired: 命令执行超过 300 秒
          - Exception: 其他执行错误
          - 两种情况都会将任务标记为 FAILED
        """
        # 更新状态：PENDING → RUNNING
        task.state = TaskState.RUNNING
        task.start_time = time.time()

        try:
            # 创建子进程
            # shell=True: 通过 shell 执行命令（支持管道、重定向等）
            # stdout=PIPE: 捕获标准输出
            # stderr=STDOUT: 将标准错误合并到标准输出
            # text=True: 以文本模式读取输出（自动解码）
            process = subprocess.Popen(
                task.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            task.process = process

            # 等待进程完成，最多 300 秒（5 分钟）
            stdout, _ = process.communicate(timeout=300)
            task.output = stdout
            task.exit_code = process.returncode

            # 根据退出码判断成功或失败
            # exit_code == 0 表示成功
            task.state = (
                TaskState.COMPLETED if process.returncode == 0
                else TaskState.FAILED
            )

        except subprocess.TimeoutExpired:
            # 命令执行超时
            task.state = TaskState.FAILED
            task.output = "Error: Task timed out after 300 seconds"
            task.exit_code = -1
            # 强制终止超时的进程
            if task.process:
                task.process.kill()

        except Exception as e:
            # 其他执行错误
            task.state = TaskState.FAILED
            task.output = f"Error: {e}"
            task.exit_code = -1

        finally:
            # 无论成功失败，记录结束时间
            task.end_time = time.time()

        # 触发回调函数（如果注册了的话）
        callback = self._callbacks.get(task.task_id)
        if callback:
            callback(task)

    def get_status(self, task_id: str) -> Optional[BackgroundTask]:
        """
        获取任务状态。

        对应 Claude Code 的任务状态查询：
          - 返回 BackgroundTask 对象，包含状态、输出等信息
          - 如果任务不存在，返回 None

        Args:
            task_id: 任务 ID

        Returns:
            Optional[BackgroundTask]: 任务对象，不存在则返回 None
        """
        return self.tasks.get(task_id)

    def get_output(self, task_id: str) -> str:
        """
        获取任务的输出内容。

        对应 Claude Code 的 Monitor 工具：
          - 返回任务的标准输出
          - 如果任务不存在，返回错误信息

        Args:
            task_id: 任务 ID

        Returns:
            str: 任务输出内容
        """
        task = self.tasks.get(task_id)
        if not task:
            return f"Error: task '{task_id}' not found"
        return task.output

    def stop(self, task_id: str) -> str:
        """
        停止一个正在运行的任务。

        对应 Claude Code 的 TaskStop 工具：
          - 终止正在运行的子进程
          - 将任务状态设置为 STOPPED

        Args:
            task_id: 任务 ID

        Returns:
            str: 操作结果消息
        """
        task = self.tasks.get(task_id)
        if not task:
            return f"Error: task '{task_id}' not found"

        # 只有 RUNNING 状态的任务才能被停止
        if task.state == TaskState.RUNNING and task.process:
            task.process.kill()
            task.state = TaskState.STOPPED
            task.end_time = time.time()
            return f"Task '{task_id}' stopped"

        return f"Task '{task_id}' is not running (state: {task.state})"

    def list_tasks(self) -> list[BackgroundTask]:
        """
        列出所有任务。

        对应 Claude Code 的任务列表功能：
          - 返回所有已提交的任务
          - 包括已完成、运行中、失败的任务

        Returns:
            list[BackgroundTask]: 所有任务的列表
        """
        return list(self.tasks.values())

    def wait_for(self, task_id: str, timeout: float = 300) -> BackgroundTask:
        """
        等待任务完成。

        阻塞当前线程，直到任务完成或超时。
        每 0.5 秒检查一次任务状态。

        Args:
            task_id: 任务 ID
            timeout: 最大等待时间（秒），默认 300 秒

        Returns:
            BackgroundTask: 完成的任务对象

        Raises:
            ValueError: 任务不存在
            TimeoutError: 等待超时
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task '{task_id}' not found")

        start = time.time()
        # 轮询等待：每 0.5 秒检查一次状态
        while task.state in (TaskState.PENDING, TaskState.RUNNING):
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for task '{task_id}'")
            time.sleep(0.5)

        return task


# ══════════════════════════════════════════════════════════════
# 第四部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U12 - Background Tasks 后台任务演示")
    print("=" * 60)

    manager = BackgroundTaskManager()

    # ── 提交后台任务 ──────────────────────────────────────
    print("\n── 提交后台任务 ──")

    # 任务 1：带回调函数的后台任务
    # 回调函数会在任务完成时自动触发
    task1_id = manager.submit(
        "echo 'Hello from background task 1' && sleep 1 && echo 'Task 1 done'",
        callback=lambda t: print(f"\n  [回调] 任务 {t.task_id} 完成: {t.state.value}")
    )
    print(f"  任务 1 已提交: {task1_id}")

    # 任务 2：不带回调的后台任务
    task2_id = manager.submit(
        "echo 'Starting task 2' && sleep 2 && echo 'Task 2 done'",
    )
    print(f"  任务 2 已提交: {task2_id}")

    # ── 查看任务状态 ──────────────────────────────────────
    print("\n── 任务状态 ──")
    for task in manager.list_tasks():
        # 显示任务 ID、状态和命令（截取前 50 字符）
        print(f"  [{task.task_id}] {task.state.value}: {task.command[:50]}...")

    # ── 等待任务完成 ──────────────────────────────────────
    print("\n── 等待任务完成 ──")

    # wait_for() 会阻塞直到任务完成
    completed = manager.wait_for(task1_id)
    print(f"  任务 1 输出:\n{completed.output}")

    print(f"\n── 任务 2 输出 ──")
    task2 = manager.wait_for(task2_id)
    print(f"  {task2.output}")
    # 计算任务耗时
    print(f"  耗时: {task2.end_time - task2.start_time:.1f} 秒")

    # ── 演示说明 ──────────────────────────────────────────
    print("\n── Claude Code 后台任务机制说明 ──")
    print("""
    Claude Code 的后台任务通过 Bash tool 实现：

    1. 提交后台任务：
       Bash(command="npm run build", run_in_background=true)
       → 返回任务 ID，Agent 可以继续工作

    2. 监听任务输出：
       Monitor(command="tail -f build.log", description="构建进度")
       → 每行 stdout 输出作为一个事件通知

    3. 停止后台任务：
       TaskStop(task_id="abc123")
       → 终止正在运行的后台进程

    4. 任务完成通知：
       → 后台任务完成时自动通知 Agent
       → 包含退出码和最终输出
    """)
