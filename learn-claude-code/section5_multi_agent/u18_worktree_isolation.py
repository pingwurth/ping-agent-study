"""
U18 - Worktree Isolation（工作树隔离）
========================================
本文件演示 Claude Code 的 **Worktree Isolation** 机制：如何在隔离环境中执行任务。

核心概念：
  1. Git worktree 允许在同一个仓库中同时有多个工作目录
  2. 每个 worktree 有独立的分支和文件状态
  3. 代理在 worktree 中工作，不影响主工作目录
  4. 工作完成后可以合并或丢弃

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code 提供两个 worktree 工具：                     │
  │                                                          │
  │  EnterWorktree:                                          │
  │    - 创建新的 worktree 或进入已有的                       │
  │    - 在 .claude/worktrees/ 目录下创建工作目录             │
  │    - 自动创建新分支                                       │
  │    - 切换会话的工作目录到 worktree                        │
  │                                                          │
  │  ExitWorktree:                                           │
  │    - action="keep"   → 保留 worktree 和分支              │
  │    - action="remove" → 删除 worktree 和分支              │
  │    - 恢复会话的工作目录到原始位置                         │
  └──────────────────────────────────────────────────────────┘

Worktree 的好处：
  ┌──────────────────────────────────────────────────────────┐
  │  1. 多个代理可以并行工作，互不干扰                        │
  │     - Agent A 在 worktree-auth 中实现认证                 │
  │     - Agent B 在 worktree-api 中重构 API                  │
  │                                                          │
  │  2. 实验性修改可以安全丢弃                                │
  │     - 尝试新方案，不满意就 remove                         │
  │     - 不会影响主分支                                      │
  │                                                          │
  │  3. 主分支保持稳定                                        │
  │     - 所有修改都在隔离的分支中                            │
  │     - 只有审查通过后才合并                                │
  │                                                          │
  │  4. 每个代理有独立的文件状态                              │
  │     - 不同的文件修改                                      │
  │     - 不同的未提交更改                                    │
  └──────────────────────────────────────────────────────────┘

目录结构：
  repo/
  ├── .git/
  ├── src/
  ├── .claude/
  │   └── worktrees/
  │       ├── feature-auth/    ← worktree 1
  │       │   ├── src/
  │       │   └── ...
  │       └── feature-api/     ← worktree 2
  │           ├── src/
  │           └── ...

本文件是纯 Python 实现，不依赖 anthropic SDK。
使用 subprocess 调用 git 命令实现 worktree 管理。
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════
# 第一部分：Worktree 管理器
# ══════════════════════════════════════════════════════════════

class WorktreeManager:
    """
    Git Worktree 管理器。

    封装了 git worktree 命令，提供以下功能：
      1. create()        - 创建新的 worktree
      2. list_worktrees() - 列出所有 worktree
      3. get_path()       - 获取 worktree 路径
      4. remove()         - 删除 worktree
      5. merge()          - 合并 worktree 分支到目标分支

    Claude Code 的 worktree 操作：
      - EnterWorktree → create() + 切换工作目录
      - ExitWorktree  → remove() 或保留

    目录约定：
      - worktree 创建在 .claude/worktrees/ 目录下
      - 每个 worktree 有独立的分支（与 worktree 同名）
      - 分支基于当前 HEAD 创建
    """

    def __init__(self, repo_path: str):
        """
        初始化 worktree 管理器。

        Args:
            repo_path: Git 仓库的根目录路径
        """
        self.repo_path = Path(repo_path).resolve()
        # worktree 存放目录
        self.worktrees_dir = self.repo_path / ".claude" / "worktrees"
        # 当前激活的 worktree 名称
        self.active_worktree: Optional[str] = None

    def _run_git(self, *args) -> subprocess.CompletedProcess:
        """
        执行 git 命令。

        在仓库根目录下执行 git 命令，捕获输出。

        Args:
            *args: git 命令参数

        Returns:
            subprocess.CompletedProcess: 命令执行结果
        """
        return subprocess.run(
            ["git"] + list(args),
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

    def create(self, name: str) -> str:
        """
        创建一个新的 worktree。

        对应 Claude Code 的 EnterWorktree 工具（创建新 worktree 时）。

        创建流程：
          1. 检查 worktree 是否已存在
          2. 创建 .claude/worktrees/ 目录
          3. 执行 git worktree add -b <name> <path>
          4. 设置 active_worktree

        Args:
            name: worktree 名称（同时也是分支名）

        Returns:
            str: worktree 的路径

        Raises:
            ValueError: worktree 已存在
            RuntimeError: git 命令执行失败
        """
        worktree_path = self.worktrees_dir / name

        # 检查是否已存在
        if worktree_path.exists():
            raise ValueError(f"Worktree '{name}' already exists")

        # 创建父目录
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

        # 执行 git worktree add
        # -b: 创建新分支
        result = self._run_git(
            "worktree", "add",
            "-b", name,          # 创建新分支
            str(worktree_path),  # worktree 路径
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        self.active_worktree = name
        return str(worktree_path)

    def list_worktrees(self) -> list[dict]:
        """
        列出所有 worktree。

        对应 Claude Code 的 git worktree list 命令。

        使用 --porcelain 格式解析输出：
          worktree /path/to/worktree
          HEAD abc1234...
          branch refs/heads/branch-name

        Returns:
            list[dict]: worktree 信息列表，每个包含 path, branch, head
        """
        result = self._run_git("worktree", "list", "--porcelain")
        if result.returncode != 0:
            return []

        worktrees = []
        current = {}
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                if current:
                    worktrees.append(current)
                current = {"path": line.split(" ", 1)[1]}
            elif line.startswith("branch "):
                current["branch"] = line.split(" ", 1)[1]
            elif line.startswith("HEAD "):
                current["head"] = line.split(" ", 1)[1]

        if current:
            worktrees.append(current)

        return worktrees

    def get_path(self, name: str) -> Optional[str]:
        """
        获取 worktree 的路径。

        Args:
            name: worktree 名称

        Returns:
            Optional[str]: worktree 路径，不存在则返回 None
        """
        path = self.worktrees_dir / name
        if path.exists():
            return str(path)
        return None

    def remove(self, name: str, keep_branch: bool = False) -> str:
        """
        删除一个 worktree。

        对应 Claude Code 的 ExitWorktree(action="remove")。

        删除流程：
          1. 执行 git worktree remove <path> --force
          2. 删除对应的分支（除非 keep_branch=True）
          3. 清除 active_worktree（如果是当前激活的）

        Args:
            name:        worktree 名称
            keep_branch: 是否保留分支

        Returns:
            str: 操作结果消息
        """
        worktree_path = self.worktrees_dir / name

        if not worktree_path.exists():
            return f"Worktree '{name}' not found"

        # 删除 worktree
        result = self._run_git("worktree", "remove", str(worktree_path), "--force")
        if result.returncode != 0:
            return f"Failed to remove worktree: {result.stderr}"

        # 删除分支（除非指定保留）
        if not keep_branch:
            self._run_git("branch", "-D", name)

        # 清除激活状态
        if self.active_worktree == name:
            self.active_worktree = None

        return f"Worktree '{name}' removed"

    def merge(self, name: str, target_branch: str = "main") -> str:
        """
        将 worktree 的分支合并到目标分支。

        合并流程：
          1. 切换到目标分支
          2. 执行 git merge <worktree-branch>

        Args:
            name:          worktree 名称（也是分支名）
            target_branch: 目标分支（默认 "main"）

        Returns:
            str: 操作结果消息
        """
        # 切换到目标分支
        result = self._run_git("checkout", target_branch)
        if result.returncode != 0:
            return f"Failed to checkout {target_branch}: {result.stderr}"

        # 合并分支
        result = self._run_git("merge", name)
        if result.returncode != 0:
            return f"Merge failed: {result.stderr}"

        return f"Branch '{name}' merged into '{target_branch}'"


# ══════════════════════════════════════════════════════════════
# 第二部分：隔离执行器
# ══════════════════════════════════════════════════════════════

class IsolatedExecutor:
    """
    在隔离的 worktree 中执行任务的执行器。

    使用场景：
      - 多个代理并行开发不同功能
      - 实验性修改（可以安全丢弃）
      - 代码审查（在隔离环境中检查代码）

    执行流程：
      1. 创建 worktree
      2. 在 worktree 中执行任务函数
      3. 返回结果
      4. 自动清理（可选）
    """

    def __init__(self, repo_path: str):
        """
        初始化隔离执行器。

        Args:
            repo_path: Git 仓库路径
        """
        self.manager = WorktreeManager(repo_path)

    def execute_in_isolation(
        self,
        name: str,
        task_fn: callable,
        auto_cleanup: bool = True,
    ) -> dict:
        """
        在隔离的 worktree 中执行任务。

        Args:
            name:         worktree 名称
            task_fn:      要执行的函数，接收 worktree_path 作为参数
            auto_cleanup: 完成后是否自动清理

        Returns:
            dict: 执行结果
                - worktree: worktree 路径
                - result:   任务函数的返回值
                - cleaned:  是否已清理
        """
        # 创建 worktree
        worktree_path = self.manager.create(name)

        try:
            # 在 worktree 中执行任务
            result = task_fn(worktree_path)
            return {
                "worktree": worktree_path,
                "result": result,
                "cleaned": False,
            }
        finally:
            # 自动清理
            if auto_cleanup:
                self.manager.remove(name)

    def parallel_execute(self, tasks: list[dict]) -> list[dict]:
        """
        并行在多个 worktree 中执行任务。

        注意：这里简化为顺序执行，实际可以使用 threading 并行。

        Args:
            tasks: 任务列表，每个包含 name 和 task_fn

        Returns:
            list[dict]: 每个任务的执行结果
        """
        results = []
        for task in tasks:
            result = self.execute_in_isolation(
                name=task["name"],
                task_fn=task["task_fn"],
            )
            results.append(result)
        return results


# ══════════════════════════════════════════════════════════════
# 第三部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U18 - Worktree Isolation 工作树隔离演示")
    print("=" * 60)

    # ── Worktree 管理 API 演示 ────────────────────────────
    print("\n── Worktree 管理 API ──")
    print("""
    # 创建 worktree
    manager = WorktreeManager("/path/to/repo")
    path = manager.create("feature-auth")
    # → 在 .claude/worktrees/feature-auth/ 创建新工作目录
    # → 创建 feature-auth 分支

    # 列出 worktree
    worktrees = manager.list_worktrees()
    # → [{"path": "...", "branch": "refs/heads/feature-auth", "head": "..."}]

    # 获取 worktree 路径
    path = manager.get_path("feature-auth")
    # → "/path/to/repo/.claude/worktrees/feature-auth"

    # 删除 worktree
    manager.remove("feature-auth")
    # → 删除 worktree 目录和分支

    # 保留分支删除 worktree
    manager.remove("feature-auth", keep_branch=True)
    # → 只删除 worktree 目录，保留分支
    """)

    # ── 隔离执行器演示 ────────────────────────────────────
    print("── 隔离执行器 API ──")
    print("""
    # 在隔离环境中执行任务
    executor = IsolatedExecutor("/path/to/repo")
    result = executor.execute_in_isolation(
        name="test-feature",
        task_fn=lambda path: f"在 {path} 中完成了工作",
        auto_cleanup=True,
    )
    # → {"worktree": "...", "result": "...", "cleaned": True}

    # 并行在多个 worktree 中执行
    results = executor.parallel_execute([
        {"name": "feature-auth", "task_fn": lambda p: implement_auth(p)},
        {"name": "feature-api", "task_fn": lambda p: refactor_api(p)},
    ])
    """)

    # ── 合并演示 ──────────────────────────────────────────
    print("── 合并操作 ──")
    print("""
    # 合并 worktree 分支到主分支
    manager.merge("feature-auth", "main")
    # → git checkout main
    # → git merge feature-auth
    """)

    # ── Claude Code Worktree 机制说明 ─────────────────────
    print("\n── Claude Code Worktree 机制说明 ──")
    print("""
    Claude Code 提供 EnterWorktree 和 ExitWorktree 两个工具：

    1. EnterWorktree - 创建/进入 worktree：
       EnterWorktree(name="feature-auth")
       → 在 .claude/worktrees/feature-auth/ 创建新工作目录
       → 创建 feature-auth 分支（基于当前 HEAD）
       → 切换会话的工作目录

       EnterWorktree(path=".claude/worktrees/existing-wt")
       → 进入已存在的 worktree
       → 切换会话的工作目录

    2. ExitWorktree - 离开 worktree：
       ExitWorktree(action="keep")
       → 保留 worktree 和分支在磁盘上
       → 恢复会话的工作目录到原始位置

       ExitWorktree(action="remove")
       → 删除 worktree 目录和分支
       → 恢复会话的工作目录

    3. 使用场景：
       - 多个 Agent 并行开发不同功能
       - 实验性修改（不满意就 remove）
       - 代码审查（在隔离环境中检查）
       - 长时间任务（不影响主工作目录）

    4. 目录结构：
       repo/
       ├── .git/
       ├── src/
       └── .claude/
           └── worktrees/
               ├── feature-auth/  ← Agent A 的工作目录
               └── feature-api/   ← Agent B 的工作目录
    """)
