"""
U11 - Memory（记忆系统）
=========================
本文件演示 **记忆系统**：跨会话持久化存储重要信息。
使用 Anthropic SDK 和纯 Python 实现。

核心概念：
  1. 记忆系统让 Agent 能在不同会话之间保持知识
  2. 与上下文压缩不同，记忆是持久化到文件系统的
  3. 记忆分为多种类型：用户偏好、项目信息、反馈等
  4. 记忆存储在 ~/.claude/projects/<project>/memory/ 目录下

记忆 vs 上下文：
  ┌──────────────────────────────────────────────────────────┐
  │  上下文（Context）                                        │
  │  - 当前会话的对话历史                                     │
  │  - 会话结束即丢失                                         │
  │  - 可以被压缩                                             │
  ├──────────────────────────────────────────────────────────┤
  │  记忆（Memory）                                           │
  │  - 跨会话持久化存储                                       │
  │  - 存储在文件系统中                                       │
  │  - 只存储关键信息                                         │
  │  - 有结构化的分类（user/project/feedback/reference）       │
  └──────────────────────────────────────────────────────────┘

存储结构：
  ~/.claude/projects/<project>/memory/
  ├── MEMORY.md              # 索引文件（每条一行摘要）
  ├── user_role.md           # 用户角色记忆
  ├── project_goals.md       # 项目目标记忆
  ├── feedback_testing.md    # 测试相关反馈
  └── reference_linear.md    # Linear 工具引用

每个记忆文件格式（Markdown + YAML frontmatter）：
  ---
  name: user-role
  description: 用户是后端工程师，熟悉 Python 和 Go
  type: user
  ---

  用户是一位后端工程师，主要使用 Python 和 Go。
  对前端技术不太熟悉，解释时需要多用后端的类比。
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


# ── 记忆类型常量 ────────────────────────────────────────────
class MemoryType:
    """
    记忆类型常量。

    Claude Code 的记忆系统将记忆分为四类：
      - USER:      用户个人信息（角色、偏好、技能水平）
      - PROJECT:   项目相关信息（目标、架构、技术栈）
      - FEEDBACK:  用户反馈（对 Agent 行为的纠正）
      - REFERENCE: 外部资源引用（文档链接、工具配置）
    """
    USER = "user"            # 用户信息：角色、偏好、习惯
    PROJECT = "project"      # 项目信息：目标、架构、约束
    FEEDBACK = "feedback"    # 用户反馈：纠正、偏好调整
    REFERENCE = "reference"  # 外部资源：文档、链接、配置


# ── 记忆数据结构 ────────────────────────────────────────────
@dataclass
class Memory:
    """
    单条记忆的数据结构。

    每条记忆对应一个独立的 markdown 文件，使用 YAML frontmatter 存储元数据。

    Attributes:
        name:         记忆的唯一标识名（用于文件名和索引）
        description:  记忆的简短描述（显示在索引中）
        memory_type:  记忆类型（user/project/feedback/reference）
        content:      记忆的详细内容（markdown 正文）
        created_at:   创建时间戳
        updated_at:   最后更新时间戳
    """
    name: str
    description: str
    memory_type: str
    content: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ── 记忆管理器 ──────────────────────────────────────────────
class MemoryManager:
    """
    记忆系统的管理器。

    Claude Code 的记忆系统工作流程：
      ① 会话开始时：加载 MEMORY.md 索引文件
      ② 需要记忆时：根据索引读取具体的记忆文件
      ③ 学到新信息时：创建或更新记忆文件
      ④ 记忆过时后：更新或删除记忆文件

    索引文件 MEMORY.md 格式：
      - [user-role](user_role.md) — 用户是后端工程师
      - [project-goals](project_goals.md) — Q3 API 性能优化
      - [feedback-testing](feedback_testing.md) — 不要 mock 数据库
    """

    def __init__(self, memory_dir: str = None):
        """
        初始化记忆管理器。

        Args:
            memory_dir: 记忆存储目录路径。
                       默认为 ~/.claude/memory（Claude Code 的默认位置）
        """
        if memory_dir is None:
            memory_dir = os.path.expanduser("~/.claude/memory")
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 内存中的记忆缓存：name -> Memory 对象
        self.memories: dict[str, Memory] = {}

        # 启动时加载索引
        self._load_index()

    def _index_path(self) -> Path:
        """返回 MEMORY.md 索引文件的路径。"""
        return self.memory_dir / "MEMORY.md"

    def _load_index(self):
        """
        加载 MEMORY.md 索引文件。

        索引格式：- [name](filename.md) — description

        解析流程：
          1. 读取 MEMORY.md 的每一行
          2. 解析 markdown 链接格式提取 name 和 filename
          3. 读取对应的记忆文件内容
          4. 从 frontmatter 提取 memory_type
          5. 构建 Memory 对象并缓存
        """
        index_path = self._index_path()
        if not index_path.exists():
            return

        for line in index_path.read_text().splitlines():
            line = line.strip()
            # 解析格式：- [name](filename.md) — description
            if line.startswith("- [") and "](" in line:
                name = line.split("]")[0][3:]           # 提取 name
                filename = line.split("(")[1].split(")")[0]  # 提取 filename
                # 提取 description（在 — 后面）
                description = line.split("—")[-1].strip() if "—" in line else ""

                # 读取记忆文件
                filepath = self.memory_dir / filename
                if filepath.exists():
                    content = filepath.read_text()
                    memory_type = self._extract_type(content)
                    self.memories[name] = Memory(
                        name=name,
                        description=description,
                        memory_type=memory_type,
                        content=content,
                    )

    def _extract_type(self, content: str) -> str:
        """
        从 markdown 文件的 YAML frontmatter 中提取 type 字段。

        Frontmatter 格式：
          ---
          name: xxx
          type: user
          ---
        """
        if content.startswith("---"):
            try:
                # 提取两个 --- 之间的内容
                frontmatter = content.split("---")[1]
                for line in frontmatter.splitlines():
                    if line.strip().startswith("type:"):
                        return line.split(":")[1].strip()
            except (IndexError, ValueError):
                pass
        return MemoryType.USER  # 默认为 user 类型

    def save(self, memory: Memory) -> str:
        """
        保存一条记忆。

        流程：
          1. 生成文件名（name 小写，空格替换为下划线）
          2. 写入 markdown 文件（带 frontmatter）
          3. 更新 MEMORY.md 索引
          4. 更新内存缓存

        Args:
            memory: 要保存的 Memory 对象

        Returns:
            str: 保存结果消息
        """
        # 生成文件名：user-role -> user_role.md
        filename = f"{memory.name.lower().replace(' ', '_')}.md"
        filepath = self.memory_dir / filename

        # 构建带 frontmatter 的 markdown 内容
        content = f"""---
name: {memory.name}
description: {memory.description}
type: {memory.memory_type}
---

{memory.content}
"""
        # 写入文件
        filepath.write_text(content)

        # 更新索引
        self._update_index(memory, filename)

        # 更新缓存
        memory.updated_at = time.time()
        self.memories[memory.name] = memory

        return f"Memory saved: {memory.name}"

    def _update_index(self, memory: Memory, filename: str):
        """
        更新 MEMORY.md 索引文件。

        如果记忆已存在则更新对应行，否则追加新行。

        Args:
            memory: 记忆对象
            filename: 记忆文件名
        """
        index_path = self._index_path()
        lines = []
        found = False

        # 读取现有索引，查找是否已有同名记忆
        if index_path.exists():
            for line in index_path.read_text().splitlines():
                if f"[{memory.name}]" in line:
                    # 更新已有条目
                    lines.append(f"- [{memory.name}]({filename}) — {memory.description}")
                    found = True
                else:
                    lines.append(line)

        # 新记忆：追加到末尾
        if not found:
            lines.append(f"- [{memory.name}]({filename}) — {memory.description}")

        index_path.write_text("\n".join(lines) + "\n")

    def recall(self, query: str) -> list[Memory]:
        """
        根据查询召回相关记忆。

        简化实现：按关键词匹配（name、description、content）。
        Claude Code 实际使用更复杂的语义匹配（embedding 相似度）。

        Args:
            query: 搜索关键词

        Returns:
            list[Memory]: 匹配的记忆列表
        """
        results = []
        query_lower = query.lower()
        for memory in self.memories.values():
            if (query_lower in memory.name.lower()
                    or query_lower in memory.description.lower()
                    or query_lower in memory.content.lower()):
                results.append(memory)
        return results

    def list_all(self) -> list[Memory]:
        """列出所有记忆。"""
        return list(self.memories.values())

    def delete(self, name: str) -> str:
        """
        删除一条记忆。

        流程：
          1. 从内存缓存中移除
          2. 删除对应的 markdown 文件
          3. 重建 MEMORY.md 索引

        Args:
            name: 要删除的记忆名称

        Returns:
            str: 删除结果消息
        """
        if name in self.memories:
            # 删除文件
            filename = f"{name.lower().replace(' ', '_')}.md"
            filepath = self.memory_dir / filename
            if filepath.exists():
                filepath.unlink()

            # 从缓存移除
            del self.memories[name]

            # 重建索引（因为删除后行号变了）
            self._rebuild_index()
            return f"Memory deleted: {name}"
        return f"Memory not found: {name}"

    def _rebuild_index(self):
        """
        重建 MEMORY.md 索引文件。

        删除记忆后需要重建索引，因为中间可能有空行。
        """
        lines = []
        for memory in self.memories.values():
            filename = f"{memory.name.lower().replace(' ', '_')}.md"
            lines.append(f"- [{memory.name}]({filename}) — {memory.description}")

        # 如果没有记忆，写入空内容
        content = "\n".join(lines) + "\n" if lines else ""
        self._index_path().write_text(content)


# ── 程序入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  Memory 记忆系统演示（Anthropic SDK 版）")
    print("=" * 60)
    print()

    # 使用临时目录演示，避免污染真实的 ~/.claude/memory
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(memory_dir=tmpdir)

        # ── 保存记忆 ──────────────────────────────────────
        # 记忆 1：用户角色信息
        manager.save(Memory(
            name="user-role",
            description="用户是后端工程师，熟悉 Python 和 Go",
            memory_type=MemoryType.USER,
            content="用户是一位后端工程师，主要使用 Python 和 Go。\n对前端技术不太熟悉，解释时需要多用后端的类比。",
        ))

        # 记忆 2：项目目标
        manager.save(Memory(
            name="project-goals",
            description="Q3 目标：API 性能优化",
            memory_type=MemoryType.PROJECT,
            content="项目 Q3 目标是优化 API 响应时间，目标是 p99 < 200ms。\n主要瓶颈在数据库查询和缓存策略。",
        ))

        # 记忆 3：用户反馈
        manager.save(Memory(
            name="feedback-testing",
            description="不要 mock 数据库，使用真实数据库测试",
            memory_type=MemoryType.FEEDBACK,
            content="用户强调：集成测试必须使用真实的数据库，不要 mock。\n原因：之前 mock 测试通过但生产环境迁移失败。",
        ))

        # ── 列出所有记忆 ──────────────────────────────────
        print("── 所有记忆 ──")
        for m in manager.list_all():
            print(f"  [{m.memory_type}] {m.name}: {m.description}")

        # ── 召回记忆（关键词搜索）─────────────────────────
        print("\n── 搜索 '测试' ──")
        results = manager.recall("测试")
        for m in results:
            print(f"  找到: {m.name}")

        # ── 查看 MEMORY.md 索引文件 ───────────────────────
        print(f"\n── MEMORY.md 索引内容 ──")
        index_path = Path(tmpdir) / "MEMORY.md"
        print(index_path.read_text())

        # ── 查看单个记忆文件 ──────────────────────────────
        print("── 记忆文件示例 (user_role.md) ──")
        print((Path(tmpdir) / "user_role.md").read_text())

        # ── 演示删除记忆 ─────────────────────────────────
        print("── 删除记忆 'feedback-testing' ──")
        result = manager.delete("feedback-testing")
        print(f"  {result}")
        print(f"  剩余记忆数: {len(manager.list_all())}")

        # ── 更新记忆 ─────────────────────────────────────
        print("\n── 更新记忆 'user-role' ──")
        manager.save(Memory(
            name="user-role",
            description="用户是全栈工程师，熟悉 Python、Go 和 React",
            memory_type=MemoryType.USER,
            content="用户是一位全栈工程师，后端用 Python 和 Go，前端用 React。\n最近在学习 TypeScript。",
        ))
        print("  更新后的 MEMORY.md:")
        print(index_path.read_text())
