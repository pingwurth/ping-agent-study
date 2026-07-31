"""
U18 - Worktree Isolation（工作树隔离）
========================================
本文件演示 **Worktree Isolation** 机制：如何在隔离环境中执行任务。
使用 LangGraph 子图封装隔离执行。

核心概念：
  1. Git worktree 允许在同一个仓库中同时有多个工作目录
  2. 每个 worktree 有独立的分支和文件状态
  3. 代理在 worktree 中工作，不影响主工作目录
  4. 工作完成后可以合并或丢弃

LangGraph 集成：
  ┌──────────────────────────────────────────────────────────┐
  │  将 worktree 隔离执行封装为 LangGraph 子图节点：          │
  │                                                          │
  │  def isolated_node(state):                               │
  │      executor = IsolatedExecutor(repo_path)              │
  │      result = executor.execute_in_isolation(             │
  │          name="feature-x",                               │
  │          task_fn=lambda path: agent.invoke(...),         │
  │      )                                                   │
  │      return {"result": result}                           │
  └──────────────────────────────────────────────────────────┘
"""

import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Worktree 管理器 ───────────────────────────────────────
class WorktreeManager:
    """
    Git Worktree 管理器。

    Claude Code 的 worktree 操作：
      - EnterWorktree: 创建新的 worktree 并切换进去
      - ExitWorktree: 离开 worktree（保留或删除）
      - 在 worktree 中的所有文件操作都是隔离的
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.worktrees_dir = self.repo_path / ".claude" / "worktrees"
        self.active_worktree: Optional[str] = None

    def _run_git(self, *args) -> subprocess.CompletedProcess:
        """执行 git 命令。"""
        return subprocess.run(
            ["git"] + list(args),
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

    def create(self, name: str) -> str:
        """创建一个新的 worktree。"""
        worktree_path = self.worktrees_dir / name

        if worktree_path.exists():
            raise ValueError(f"Worktree '{name}' already exists")

        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

        result = self._run_git(
            "worktree", "add",
            "-b", name,
            str(worktree_path),
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        self.active_worktree = name
        return str(worktree_path)

    def list_worktrees(self) -> list[dict]:
        """列出所有 worktree。"""
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
        """获取 worktree 的路径。"""
        path = self.worktrees_dir / name
        if path.exists():
            return str(path)
        return None

    def remove(self, name: str, keep_branch: bool = False) -> str:
        """删除一个 worktree。"""
        worktree_path = self.worktrees_dir / name

        if not worktree_path.exists():
            return f"Worktree '{name}' not found"

        result = self._run_git("worktree", "remove", str(worktree_path), "--force")
        if result.returncode != 0:
            return f"Failed to remove worktree: {result.stderr}"

        if not keep_branch:
            self._run_git("branch", "-D", name)

        if self.active_worktree == name:
            self.active_worktree = None

        return f"Worktree '{name}' removed"

    def merge(self, name: str, target_branch: str = "main") -> str:
        """将 worktree 的分支合并到目标分支。"""
        result = self._run_git("checkout", target_branch)
        if result.returncode != 0:
            return f"Failed to checkout {target_branch}: {result.stderr}"

        result = self._run_git("merge", name)
        if result.returncode != 0:
            return f"Merge failed: {result.stderr}"

        return f"Branch '{name}' merged into '{target_branch}'"


# ── 隔离的代理执行器 ──────────────────────────────────────
class IsolatedExecutor:
    """
    在隔离的 worktree 中执行任务的代理。

    使用场景：
      - 多个代理并行开发不同功能
      - 实验性修改（可以安全丢弃）
      - 代码审查（在隔离环境中检查代码）
    """

    def __init__(self, repo_path: str):
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
            name: worktree 名称
            task_fn: 要执行的函数，接收 worktree_path 作为参数
            auto_cleanup: 完成后是否自动清理

        Returns:
            dict: {"worktree": str, "result": Any, "cleaned": bool}
        """
        worktree_path = self.manager.create(name)

        try:
            result = task_fn(worktree_path)
            return {
                "worktree": worktree_path,
                "result": result,
                "cleaned": False,
            }
        finally:
            if auto_cleanup:
                self.manager.remove(name)

    def parallel_execute(self, tasks: list[dict]) -> list[dict]:
        """并行在多个 worktree 中执行任务。"""
        results = []
        for task in tasks:
            result = self.execute_in_isolation(
                name=task["name"],
                task_fn=task["task_fn"],
            )
            results.append(result)
        return results


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Worktree Isolation 工作树隔离演示\n")

    print("── Worktree 管理 API ──")
    print("""
    # 创建 worktree
    manager = WorktreeManager("/path/to/repo")
    path = manager.create("feature-auth")

    # 列出 worktree
    worktrees = manager.list_worktrees()

    # 在隔离环境中执行
    executor = IsolatedExecutor("/path/to/repo")
    result = executor.execute_in_isolation(
        name="test-feature",
        task_fn=lambda path: f"在 {path} 中完成了工作",
    )

    # 合并到主分支
    manager.merge("feature-auth", "main")

    # 删除 worktree
    manager.remove("feature-auth")
    """)

    print("── LangGraph 集成 ──")
    print("""
    # 将 worktree 隔离执行封装为 LangGraph 节点
    def isolated_coding_node(state):
        executor = IsolatedExecutor(state["repo_path"])
        result = executor.execute_in_isolation(
            name=f"feature-{state['task_id']}",
            task_fn=lambda path: run_coding_agent(path, state["task"]),
        )
        return {"result": result}

    # 在主图中使用
    graph.add_node("isolated_coding", isolated_coding_node)
    """)

    print("── Worktree 隔离的好处 ──")
    print("  1. 多个代理可以并行工作，互不干扰")
    print("  2. 实验性修改可以安全丢弃")
    print("  3. 主分支保持稳定")
    print("  4. 每个代理有独立的文件状态")
