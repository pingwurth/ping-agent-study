"""
U13 - Cron Scheduler（定时任务调度器）
========================================
本文件演示 **定时任务调度** 机制：如何设置定时执行的任务。
使用标准 cron 表达式 + LangGraph Agent 作为执行器。

核心概念：
  1. CronCreate 工具允许 Agent 设置定时任务
  2. 使用标准的 5 字段 cron 表达式
  3. 支持一次性任务和循环任务
  4. 任务只在当前会话中有效（会话结束即消失）

LangGraph 集成：
  ┌──────────────────────────────────────────────────────────┐
  │  Cron 调度器的执行器可以是 LangGraph Agent：              │
  │                                                          │
  │  def agent_executor(prompt: str):                        │
  │      result = agent.invoke({"messages": [prompt]})       │
  │      return result                                       │
  │                                                          │
  │  scheduler.set_executor(agent_executor)                  │
  │  scheduler.start()  # 后台线程定时触发 agent              │
  └──────────────────────────────────────────────────────────┘

Cron 表达式格式：
  ┌───────────── 分钟 (0-59)
  │ ┌─────────── 小时 (0-23)
  │ │ ┌───────── 日 (1-31)
  │ │ │ ┌─────── 月 (1-12)
  │ │ │ │ ┌───── 星期 (0-6, 0=周日)
  │ │ │ │ │
  * * * * *
"""

import os
import time
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional


# ── Cron 表达式解析器 ─────────────────────────────────────
class CronField:
    """
    解析单个 cron 字段。

    支持的格式：
      - *: 任意值
      - N: 具体值（如 5）
      - */N: 每隔 N（如 */5 表示每 5 分钟）
      - N-M: 范围（如 1-5）
      - N,M: 列表（如 1,3,5）
    """

    def __init__(self, field_str: str, min_val: int, max_val: int):
        self.field_str = field_str
        self.min_val = min_val
        self.max_val = max_val
        self.values = self._parse(field_str)

    def _parse(self, s: str) -> set[int]:
        values = set()
        for part in s.split(","):
            if part == "*":
                values.update(range(self.min_val, self.max_val + 1))
            elif part.startswith("*/"):
                step = int(part[2:])
                values.update(range(self.min_val, self.max_val + 1, step))
            elif "-" in part:
                start, end = part.split("-")
                values.update(range(int(start), int(end) + 1))
            else:
                values.add(int(part))
        return values

    def matches(self, value: int) -> bool:
        return value in self.values


@dataclass
class CronExpression:
    """
    标准 5 字段 cron 表达式。

    示例：
      */5 * * * *    → 每 5 分钟
      0 9 * * 1-5   → 工作日上午 9 点
      30 14 15 * *  → 每月 15 号下午 2:30
    """
    minute: CronField
    hour: CronField
    day: CronField
    month: CronField
    weekday: CronField

    @classmethod
    def parse(cls, expr: str) -> "CronExpression":
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expr}. Expected 5 fields.")

        return cls(
            minute=CronField(parts[0], 0, 59),
            hour=CronField(parts[1], 0, 23),
            day=CronField(parts[2], 1, 31),
            month=CronField(parts[3], 1, 12),
            weekday=CronField(parts[4], 0, 6),
        )

    def matches(self, dt: datetime) -> bool:
        return (
            self.minute.matches(dt.minute)
            and self.hour.matches(dt.hour)
            and self.day.matches(dt.day)
            and self.month.matches(dt.month)
            and self.weekday.matches(dt.weekday())
        )


# ── 定时任务 ──────────────────────────────────────────────
@dataclass
class ScheduledTask:
    """
    一个定时任务。

    Claude Code 的 CronCreate 参数：
    {
        "cron": "*/5 * * * *",
        "prompt": "检查部署状态",
        "recurring": true,
        "durable": false
    }
    """
    task_id: str
    cron_expr: CronExpression
    prompt: str
    recurring: bool = True
    durable: bool = False
    last_run: Optional[float] = None
    run_count: int = 0
    active: bool = True


# ── Cron 调度器 ───────────────────────────────────────────
class CronScheduler:
    """
    定时任务调度器。

    Claude Code 的定时任务特性：
      - 只在会话期间有效（除非 durable=true）
      - 任务在 REPL 空闲时触发
      - 循环任务 7 天后自动过期
      - 可以通过 CronDelete 删除
      - 可以通过 CronList 查看所有任务

    与 LangGraph 集成：
      scheduler.set_executor(lambda prompt: agent.invoke({"messages": [prompt]}))
    """

    def __init__(self):
        self.tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[Callable[[str], None]] = None

    def set_executor(self, executor: Callable[[str], None]):
        """
        设置任务执行器。

        可以设置为 LangGraph Agent 的调用函数：
          scheduler.set_executor(
              lambda prompt: agent.invoke({"messages": [HumanMessage(content=prompt)]})
          )
        """
        self._executor = executor

    def create(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = False,
    ) -> str:
        """创建一个定时任务。"""
        task_id = str(uuid.uuid4())[:8]
        cron_expr = CronExpression.parse(cron)

        task = ScheduledTask(
            task_id=task_id,
            cron_expr=cron_expr,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
        )

        self.tasks[task_id] = task
        return task_id

    def delete(self, task_id: str) -> bool:
        """删除一个定时任务。"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        """列出所有定时任务。"""
        return list(self.tasks.values())

    def start(self):
        """启动调度器（在后台线程中运行）。"""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止调度器。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        """调度主循环，每分钟检查一次。"""
        while self._running:
            now = datetime.now()
            tasks_to_remove = []

            for task_id, task in self.tasks.items():
                if not task.active:
                    continue

                if not task.cron_expr.matches(now):
                    continue

                if task.last_run and (time.time() - task.last_run) < 60:
                    continue

                self._execute_task(task)

                if not task.recurring:
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self.tasks[task_id]

            time.sleep(60 - datetime.now().second)

    def _execute_task(self, task: ScheduledTask):
        """执行一个定时任务。"""
        task.last_run = time.time()
        task.run_count += 1

        if self._executor:
            try:
                self._executor(task.prompt)
            except Exception as e:
                print(f"[Cron] Error executing task {task.task_id}: {e}")


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Cron 调度器演示\n")

    # 演示 cron 表达式解析
    print("── Cron 表达式解析 ──")
    test_exprs = [
        "*/5 * * * *",
        "0 9 * * 1-5",
        "30 14 15 * *",
        "0 0 * * 0",
    ]
    for expr_str in test_exprs:
        expr = CronExpression.parse(expr_str)
        now = datetime.now()
        print(f"  {expr_str:20s} → 当前匹配: {expr.matches(now)}")

    # 演示调度器
    print("\n── 调度器演示 ──")
    scheduler = CronScheduler()

    def my_executor(prompt: str):
        print(f"  [执行] {prompt} (时间: {datetime.now().strftime('%H:%M:%S')})")

    scheduler.set_executor(my_executor)

    task1 = scheduler.create("*/1 * * * *", "每分钟检查一次", recurring=True)
    task2 = scheduler.create("*/2 * * * *", "每两分钟报告状态", recurring=True)
    task3 = scheduler.create("* * * * *", "一次性提醒", recurring=False)

    print(f"  已创建 3 个任务")
    print(f"  任务数: {len(scheduler.list_tasks())}")

    print("\n── 任务列表 ──")
    for task in scheduler.list_tasks():
        print(f"  [{task.task_id}] {task.prompt}")

    print("\n── 与 LangGraph Agent 集成 ──")
    print("  scheduler.set_executor(")
    print("      lambda prompt: agent.invoke({'messages': [HumanMessage(content=prompt)]})")
    print("  )")
    print("  scheduler.start()  # 后台线程定时触发 agent")

    print("\n  (调度器 API 演示完成，未启动实际调度)")
