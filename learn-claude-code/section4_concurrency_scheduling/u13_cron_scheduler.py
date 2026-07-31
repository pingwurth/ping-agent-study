"""
U13 - Cron Scheduler（定时任务调度器）
========================================
本文件演示 Claude Code 的 **定时任务调度** 机制。

核心概念：
  1. Claude Code 支持通过 CronCreate 工具设置定时任务
  2. 使用标准的 5 字段 cron 表达式
  3. 支持一次性任务和循环任务
  4. 任务只在当前会话中有效（会话结束即消失）
  5. 循环任务 7 天后自动过期

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  CronCreate 工具参数：                                    │
  │  {                                                       │
  │      "cron": "*/5 * * * *",    ← cron 表达式             │
  │      "prompt": "检查部署状态",  ← 要执行的提示词          │
  │      "recurring": true,         ← 是否循环执行            │
  │      "durable": false           ← 是否持久化              │
  │  }                                                       │
  │                                                          │
  │  CronDelete 工具：删除定时任务                            │
  │  CronList 工具：列出所有定时任务                          │
  └──────────────────────────────────────────────────────────┘

Cron 表达式格式：
  ┌───────────── 分钟 (0-59)
  │ ┌─────────── 小时 (0-23)
  │ │ ┌───────── 日 (1-31)
  │ │ │ ┌─────── 月 (1-12)
  │ │ │ │ ┌───── 星期 (0-6, 0=周日)
  │ │ │ │ │
  * * * * *

  支持的语法：
    *        → 任意值
    N        → 具体值（如 5）
    */N      → 每隔 N（如 */5 表示每 5 分钟）
    N-M      → 范围（如 1-5 表示 1 到 5）
    N,M      → 列表（如 1,3,5）

  常用示例：
    */5 * * * *    → 每 5 分钟
    0 9 * * 1-5   → 工作日上午 9 点
    30 14 15 * *  → 每月 15 号下午 2:30
    0 0 * * 0     → 每周日子夜

本文件是纯 Python 实现，不依赖 anthropic SDK。
使用标准库的 threading 实现定时调度。
"""

import os
import time
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional


# ══════════════════════════════════════════════════════════════
# 第一部分：Cron 表达式解析器
# ══════════════════════════════════════════════════════════════

class CronField:
    """
    解析单个 cron 字段。

    Cron 表达式由 5 个字段组成，每个字段可以是：
      - *        任意值（匹配所有可能的值）
      - N        具体值（如 5 表示第 5 分钟）
      - */N      步长值（如 */5 表示每 5 个单位）
      - N-M      范围（如 1-5 表示 1 到 5）
      - N,M      列表（如 1,3,5 表示第 1、3、5 个单位）

    解析过程：
      1. 按逗号分割，处理列表语法
      2. 对每个部分判断类型（*, */N, N-M, N）
      3. 展开为具体的值集合

    示例：
      CronField("*/5", 0, 59)  → {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}
      CronField("1-5", 0, 23)  → {1, 2, 3, 4, 5}
      CronField("1,3,5", 0, 6) → {1, 3, 5}
    """

    def __init__(self, field_str: str, min_val: int, max_val: int):
        """
        初始化 CronField。

        Args:
            field_str: cron 字段的字符串表示（如 "*/5", "1-5", "1,3,5"）
            min_val:   字段的最小值（如分钟为 0，小时为 0）
            max_val:   字段的最大值（如分钟为 59，小时为 23）
        """
        self.field_str = field_str
        self.min_val = min_val
        self.max_val = max_val
        # 解析字段字符串，生成所有匹配的值
        self.values = self._parse(field_str)

    def _parse(self, s: str) -> set[int]:
        """
        解析 cron 字段字符串。

        解析规则：
          1. 按逗号分割多个值
          2. 对每个部分：
             - "*"     → 所有值 [min_val, max_val]
             - "*/N"   → 步长值，从 min_val 开始，每隔 N
             - "N-M"   → 范围值，从 N 到 M
             - "N"     → 单个值

        Returns:
            set[int]: 所有匹配的值的集合
        """
        values = set()
        for part in s.split(","):
            if part == "*":
                # 通配符：匹配所有值
                values.update(range(self.min_val, self.max_val + 1))
            elif part.startswith("*/"):
                # 步长值：如 */5 表示每 5 个单位
                step = int(part[2:])
                values.update(range(self.min_val, self.max_val + 1, step))
            elif "-" in part:
                # 范围值：如 1-5 表示 1 到 5
                start, end = part.split("-")
                values.update(range(int(start), int(end) + 1))
            else:
                # 单个值：如 5
                values.add(int(part))
        return values

    def matches(self, value: int) -> bool:
        """
        检查给定值是否匹配此字段。

        Args:
            value: 要检查的值（如当前分钟数）

        Returns:
            bool: 是否匹配
        """
        return value in self.values


@dataclass
class CronExpression:
    """
    标准 5 字段 cron 表达式。

    5 个字段分别表示：
      - minute:  分钟 (0-59)
      - hour:    小时 (0-23)
      - day:     日 (1-31)
      - month:   月 (1-12)
      - weekday: 星期 (0-6, 0=周日)

    示例：
      */5 * * * *    → 每 5 分钟
      0 9 * * 1-5   → 工作日上午 9 点
      30 14 15 * *  → 每月 15 号下午 2:30
      0 0 * * 0     → 每周日子夜
    """
    minute: CronField
    hour: CronField
    day: CronField
    month: CronField
    weekday: CronField

    @classmethod
    def parse(cls, expr: str) -> "CronExpression":
        """
        解析 cron 表达式字符串。

        Args:
            expr: 5 字段的 cron 表达式（如 "*/5 * * * *"）

        Returns:
            CronExpression: 解析后的表达式对象

        Raises:
            ValueError: 字段数量不是 5
        """
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: {expr}. Expected 5 fields."
            )

        return cls(
            minute=CronField(parts[0], 0, 59),
            hour=CronField(parts[1], 0, 23),
            day=CronField(parts[2], 1, 31),
            month=CronField(parts[3], 1, 12),
            weekday=CronField(parts[4], 0, 6),
        )

    def matches(self, dt: datetime) -> bool:
        """
        检查给定时间是否匹配此 cron 表达式。

        所有 5 个字段都必须匹配才算整体匹配。

        Args:
            dt: 要检查的时间

        Returns:
            bool: 是否匹配
        """
        return (
            self.minute.matches(dt.minute)
            and self.hour.matches(dt.hour)
            and self.day.matches(dt.day)
            and self.month.matches(dt.month)
            and self.weekday.matches(dt.weekday())
        )


# ══════════════════════════════════════════════════════════════
# 第二部分：定时任务定义
# ══════════════════════════════════════════════════════════════

@dataclass
class ScheduledTask:
    """
    一个定时任务的定义。

    对应 Claude Code 的 CronCreate 工具参数：
    {
        "cron": "*/5 * * * *",    ← cron_expr
        "prompt": "检查部署状态",  ← prompt
        "recurring": true,         ← recurring
        "durable": false           ← durable
    }

    字段说明：
      - task_id:    任务唯一标识符
      - cron_expr:  解析后的 cron 表达式对象
      - prompt:     任务执行时的提示词
      - recurring:  是否循环执行（False 表示一次性任务）
      - durable:    是否持久化（跨会话保留）
      - last_run:   上次执行时间戳
      - run_count:  已执行次数
      - active:     是否激活（可以暂停任务）
    """
    task_id: str
    cron_expr: CronExpression
    prompt: str
    recurring: bool = True
    durable: bool = False
    last_run: Optional[float] = None
    run_count: int = 0
    active: bool = True


# ══════════════════════════════════════════════════════════════
# 第三部分：Cron 调度器
# ══════════════════════════════════════════════════════════════

class CronScheduler:
    """
    定时任务调度器。

    模拟 Claude Code 的定时任务机制：
      - 只在会话期间有效（除非 durable=true）
      - 任务在后台线程中定时检查和执行
      - 循环任务 7 天后自动过期
      - 可以通过 CronDelete 删除任务
      - 可以通过 CronList 查看所有任务

    实现原理：
      1. 后台线程每分钟检查一次所有任务
      2. 对每个任务，检查其 cron 表达式是否匹配当前时间
      3. 匹配则执行任务（调用 executor）
      4. 非循环任务执行一次后自动删除

    executor（执行器）：
      - 可以是任何 Callable[[str], None]
      - 通常设置为 LangGraph Agent 的调用函数：
        scheduler.set_executor(
            lambda prompt: agent.invoke({"messages": [prompt]})
        )
    """

    def __init__(self):
        # 任务字典：task_id → ScheduledTask
        self.tasks: dict[str, ScheduledTask] = {}
        # 调度器运行状态
        self._running = False
        # 后台调度线程
        self._thread: Optional[threading.Thread] = None
        # 任务执行器（可注入 LangGraph Agent）
        self._executor: Optional[Callable[[str], None]] = None

    def set_executor(self, executor: Callable[[str], None]):
        """
        设置任务执行器。

        执行器是任务触发时调用的函数。
        可以设置为打印函数（演示用）或 Agent 调用函数。

        示例：
          # 简单打印
          scheduler.set_executor(lambda prompt: print(f"执行: {prompt}"))

          # LangGraph Agent
          scheduler.set_executor(
              lambda prompt: agent.invoke({"messages": [HumanMessage(content=prompt)]})
          )

        Args:
            executor: 接收 prompt 字符串的可调用对象
        """
        self._executor = executor

    def create(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = False,
    ) -> str:
        """
        创建一个定时任务。

        对应 Claude Code 的 CronCreate 工具。

        Args:
            cron:      5 字段的 cron 表达式（如 "*/5 * * * *"）
            prompt:    任务执行时的提示词
            recurring: 是否循环执行
            durable:   是否持久化

        Returns:
            str: 任务 ID
        """
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
        """
        删除一个定时任务。

        对应 Claude Code 的 CronDelete 工具。

        Args:
            task_id: 任务 ID

        Returns:
            bool: 是否成功删除
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        """
        列出所有定时任务。

        对应 Claude Code 的 CronList 工具。

        Returns:
            list[ScheduledTask]: 所有定时任务的列表
        """
        return list(self.tasks.values())

    def start(self):
        """
        启动调度器。

        在后台线程中运行调度主循环。
        每分钟检查一次所有任务的 cron 表达式。
        """
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        停止调度器。

        设置运行标志为 False，等待调度线程结束。
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        """
        调度主循环。

        每分钟检查一次所有任务：
          1. 遍历所有激活的任务
          2. 检查 cron 表达式是否匹配当前时间
          3. 防止同一分钟内重复执行（检查 last_run）
          4. 匹配则执行任务
          5. 非循环任务执行后删除
          6. 睡眠到下一分钟的第 0 秒
        """
        while self._running:
            now = datetime.now()
            tasks_to_remove = []

            for task_id, task in self.tasks.items():
                # 跳过未激活的任务
                if not task.active:
                    continue

                # 检查 cron 表达式是否匹配当前时间
                if not task.cron_expr.matches(now):
                    continue

                # 防止同一分钟内重复执行
                # （因为 _run_loop 可能在同一分钟内多次检查）
                if task.last_run and (time.time() - task.last_run) < 60:
                    continue

                # 执行任务
                self._execute_task(task)

                # 非循环任务执行后标记删除
                if not task.recurring:
                    tasks_to_remove.append(task_id)

            # 删除已执行的非循环任务
            for task_id in tasks_to_remove:
                del self.tasks[task_id]

            # 睡眠到下一分钟的第 0 秒
            # 例如：当前是 12:30:45，则睡眠 15 秒到 12:31:00
            time.sleep(60 - datetime.now().second)

    def _execute_task(self, task: ScheduledTask):
        """
        执行一个定时任务。

        更新任务的执行统计，然后调用 executor。

        Args:
            task: 要执行的任务
        """
        task.last_run = time.time()
        task.run_count += 1

        if self._executor:
            try:
                self._executor(task.prompt)
            except Exception as e:
                print(f"[Cron] Error executing task {task.task_id}: {e}")


# ══════════════════════════════════════════════════════════════
# 第四部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U13 - Cron Scheduler 定时任务调度器演示")
    print("=" * 60)

    # ── 演示 cron 表达式解析 ──────────────────────────────
    print("\n── Cron 表达式解析 ──")

    test_exprs = [
        ("*/5 * * * *",    "每 5 分钟"),
        ("0 9 * * 1-5",    "工作日上午 9 点"),
        ("30 14 15 * *",   "每月 15 号下午 2:30"),
        ("0 0 * * 0",      "每周日子夜"),
        ("*/1 * * * *",    "每分钟"),
        ("0 */2 * * *",    "每 2 小时"),
    ]

    for expr_str, description in test_exprs:
        expr = CronExpression.parse(expr_str)
        now = datetime.now()
        matches = expr.matches(now)
        print(f"  {expr_str:20s} → {description:20s} → 当前匹配: {matches}")

    # ── 演示调度器 API ────────────────────────────────────
    print("\n── 调度器 API 演示 ──")

    scheduler = CronScheduler()

    # 设置简单的打印执行器（演示用）
    def my_executor(prompt: str):
        print(f"  [执行] {prompt} (时间: {datetime.now().strftime('%H:%M:%S')})")

    scheduler.set_executor(my_executor)

    # 创建定时任务
    task1 = scheduler.create("*/1 * * * *", "每分钟检查一次", recurring=True)
    task2 = scheduler.create("*/2 * * * *", "每两分钟报告状态", recurring=True)
    task3 = scheduler.create("* * * * *", "一次性提醒", recurring=False)

    print(f"  已创建 3 个任务")

    # 列出所有任务
    print("\n── 任务列表 ──")
    for task in scheduler.list_tasks():
        recurring_str = "循环" if task.recurring else "一次性"
        print(f"  [{task.task_id}] {task.prompt} ({recurring_str})")

    print(f"\n  任务总数: {len(scheduler.list_tasks())}")

    # 删除任务
    scheduler.delete(task3)
    print(f"  删除一次性任务后: {len(scheduler.list_tasks())} 个")

    # ── Claude Code 集成说明 ──────────────────────────────
    print("\n── Claude Code 定时任务机制说明 ──")
    print("""
    Claude Code 的定时任务通过 CronCreate/CronDelete/CronList 工具管理：

    1. 创建定时任务：
       CronCreate(cron="*/5 * * * *", prompt="检查部署状态", recurring=true)
       → Agent 会每 5 分钟自动执行 "检查部署状态"

    2. 定时任务的限制：
       - 只在会话期间有效（会话结束任务消失）
       - 循环任务 7 天后自动过期
       - 在 REPL 空闲时触发

    3. 删除定时任务：
       CronDelete(task_id="abc123")
       → 立即停止并删除任务

    4. 查看定时任务：
       CronList()
       → 列出所有活跃的定时任务

    5. 与 Agent 集成：
       scheduler.set_executor(
           lambda prompt: agent.invoke({"messages": [HumanMessage(content=prompt)]})
       )
       → 每次定时触发时，Agent 会收到 prompt 并执行
    """)

    print("  (调度器 API 演示完成，未启动实际调度)")
