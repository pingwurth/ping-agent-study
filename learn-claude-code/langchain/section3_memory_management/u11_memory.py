"""
U11 - Memory（记忆系统）
=========================
本文件演示 **记忆系统**：跨会话持久化存储重要信息。
使用 LangChain 的 BaseChatMessageHistory 接口实现。

核心概念：
  1. 记忆系统让 Agent 能在不同会话之间保持知识
  2. 与上下文压缩不同，记忆是持久化到文件系统的
  3. 记忆分为多种类型：用户偏好、项目信息、反馈等
  4. 记忆存储在 ~/.claude/projects/<project>/memory/ 目录下

LangChain 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  LangChain 提供 BaseChatMessageHistory 接口：             │
  │                                                          │
  │  class FileBasedMemory(BaseChatMessageHistory):          │
  │      def add_message(self, message): ...                 │
  │      def clear(self): ...                                │
  │                                                          │
  │  可以与 LangChain 的 ConversationChain 集成              │
  │  也可以独立使用                                          │
  └──────────────────────────────────────────────────────────┘

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
"""

import os
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# ── 记忆类型 ──────────────────────────────────────────────
class MemoryType:
    """记忆类型常量"""
    USER = "user"            # 用户信息
    PROJECT = "project"      # 项目信息
    FEEDBACK = "feedback"    # 用户反馈
    REFERENCE = "reference"  # 外部资源引用


# ── 记忆数据结构 ──────────────────────────────────────────
@dataclass
class Memory:
    """
    单条记忆的表示。

    Claude Code 中每条记忆是一个独立的 markdown 文件，包含 frontmatter：
    ---
    name: user-role
    description: 用户是后端工程师，熟悉 Python 和 Go
    type: user
    ---

    用户是一位后端工程师，主要使用 Python 和 Go。
    """
    name: str
    description: str
    memory_type: str
    content: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ── LangChain 集成：基于文件的记忆 ────────────────────────
class FileBasedMemory(BaseChatMessageHistory):
    """
    基于文件系统的 LangChain 记忆实现。

    实现 BaseChatMessageHistory 接口，可以与 LangChain 的
    ConversationChain、ChatMessageHistory 等组件集成。

    存储格式：每个会话的消息存储为 JSONL 文件。
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._messages: list[BaseMessage] = []
        self._load()

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """添加一条消息到记忆。"""
        self._messages.append(message)
        self._save_message(message)

    def clear(self) -> None:
        """清空所有消息。"""
        self._messages = []
        if self.file_path.exists():
            self.file_path.unlink()

    def _load(self):
        """从文件加载消息。"""
        if not self.file_path.exists():
            return

        import json
        for line in self.file_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data["type"] == "human":
                    self._messages.append(HumanMessage(content=data["content"]))
                elif data["type"] == "ai":
                    self._messages.append(AIMessage(content=data["content"]))
            except (json.JSONDecodeError, KeyError):
                continue

    def _save_message(self, message: BaseMessage):
        """追加一条消息到文件。"""
        import json
        msg_type = "human" if isinstance(message, HumanMessage) else "ai"
        data = {"type": msg_type, "content": message.content, "timestamp": time.time()}
        with open(self.file_path, "a") as f:
            f.write(json.dumps(data) + "\n")


# ── 记忆管理器（Claude Code 风格）────────────────────────
class MemoryManager:
    """
    记忆系统的管理器。

    Claude Code 的记忆系统工作流程：
      ① 会话开始时：加载 MEMORY.md 索引文件
      ② 需要记忆时：根据索引读取具体的记忆文件
      ③ 学到新信息时：创建或更新记忆文件
      ④ 记忆过时后：更新或删除记忆文件

    存储结构：
      ~/.claude/projects/<project>/memory/
      ├── MEMORY.md              # 索引文件（每条一行摘要）
      ├── user_role.md           # 用户角色记忆
      ├── project_goals.md       # 项目目标记忆
      ├── feedback_testing.md    # 测试相关反馈
      └── reference_linear.md    # Linear 工具引用
    """

    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            memory_dir = os.path.expanduser("~/.claude/memory")
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memories: dict[str, Memory] = {}
        self._load_index()

    def _index_path(self) -> Path:
        """MEMORY.md 索引文件的路径。"""
        return self.memory_dir / "MEMORY.md"

    def _load_index(self):
        """加载 MEMORY.md 索引。"""
        index_path = self._index_path()
        if not index_path.exists():
            return

        for line in index_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("- [") and "](" in line:
                name = line.split("]")[0][3:]
                filename = line.split("(")[1].split(")")[0]
                description = line.split("—")[-1].strip() if "—" in line else ""

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
        """从 markdown 文件的 frontmatter 中提取 type 字段。"""
        if content.startswith("---"):
            try:
                frontmatter = content.split("---")[1]
                for line in frontmatter.splitlines():
                    if line.strip().startswith("type:"):
                        return line.split(":")[1].strip()
            except (IndexError, ValueError):
                pass
        return MemoryType.USER

    def save(self, memory: Memory) -> str:
        """保存一条记忆。"""
        filename = f"{memory.name.lower().replace(' ', '_')}.md"
        filepath = self.memory_dir / filename

        content = f"""---
name: {memory.name}
description: {memory.description}
type: {memory.memory_type}
---

{memory.content}
"""
        filepath.write_text(content)
        self._update_index(memory, filename)
        memory.updated_at = time.time()
        self.memories[memory.name] = memory
        return f"Memory saved: {memory.name}"

    def _update_index(self, memory: Memory, filename: str):
        """更新 MEMORY.md 索引文件。"""
        index_path = self._index_path()
        lines = []
        found = False

        if index_path.exists():
            for line in index_path.read_text().splitlines():
                if f"[{memory.name}]" in line:
                    lines.append(f"- [{memory.name}]({filename}) — {memory.description}")
                    found = True
                else:
                    lines.append(line)

        if not found:
            lines.append(f"- [{memory.name}]({filename}) — {memory.description}")

        index_path.write_text("\n".join(lines) + "\n")

    def recall(self, query: str) -> list[Memory]:
        """
        根据查询召回相关记忆。

        简化实现：按关键词匹配。
        Claude Code 实际使用更复杂的语义匹配。
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
        """删除一条记忆。"""
        if name in self.memories:
            filename = f"{name.lower().replace(' ', '_')}.md"
            filepath = self.memory_dir / filename
            if filepath.exists():
                filepath.unlink()
            del self.memories[name]
            self._rebuild_index()
            return f"Memory deleted: {name}"
        return f"Memory not found: {name}"

    def _rebuild_index(self):
        """重建 MEMORY.md 索引。"""
        lines = []
        for memory in self.memories.values():
            filename = f"{memory.name.lower().replace(' ', '_')}.md"
            lines.append(f"- [{memory.name}]({filename}) — {memory.description}")
        self._index_path().write_text("\n".join(lines) + "\n" if lines else "")


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Memory 记忆系统演示\n")

    # 使用临时目录演示
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(memory_dir=tmpdir)

        # 保存用户信息记忆
        manager.save(Memory(
            name="user-role",
            description="用户是后端工程师，熟悉 Python 和 Go",
            memory_type=MemoryType.USER,
            content="用户是一位后端工程师，主要使用 Python 和 Go。\n对前端技术不太熟悉，解释时需要多用后端的类比。",
        ))

        # 保存项目信息记忆
        manager.save(Memory(
            name="project-goals",
            description="Q3 目标：API 性能优化",
            memory_type=MemoryType.PROJECT,
            content="项目 Q3 目标是优化 API 响应时间，目标是 p99 < 200ms。\n主要瓶颈在数据库查询和缓存策略。",
        ))

        # 保存用户反馈记忆
        manager.save(Memory(
            name="feedback-testing",
            description="不要 mock 数据库，使用真实数据库测试",
            memory_type=MemoryType.FEEDBACK,
            content="用户强调：集成测试必须使用真实的数据库，不要 mock。\n原因：之前 mock 测试通过但生产环境迁移失败。",
        ))

        # 列出所有记忆
        print("── 所有记忆 ──")
        for m in manager.list_all():
            print(f"  [{m.memory_type}] {m.name}: {m.description}")

        # 召回记忆
        print("\n── 搜索 '测试' ──")
        results = manager.recall("测试")
        for m in results:
            print(f"  找到: {m.name}")

        # 查看 MEMORY.md 索引
        print(f"\n── MEMORY.md 内容 ──")
        index_path = Path(tmpdir) / "MEMORY.md"
        print(index_path.read_text())

        # 查看记忆文件
        print("── 记忆文件示例 (user_role.md) ──")
        print((Path(tmpdir) / "user_role.md").read_text())

        # 演示 LangChain FileBasedMemory
        print("\n── LangChain FileBasedMemory ──")
        chat_memory = FileBasedMemory(os.path.join(tmpdir, "chat_history.jsonl"))
        chat_memory.add_message(HumanMessage(content="你好"))
        chat_memory.add_message(AIMessage(content="你好！有什么可以帮你的？"))
        chat_memory.add_message(HumanMessage(content="帮我写个排序算法"))
        chat_memory.add_message(AIMessage(content="好的，这是一个快速排序实现..."))

        print(f"  消息数: {len(chat_memory.messages)}")
        for msg in chat_memory.messages:
            print(f"    [{msg.__class__.__name__}] {msg.content[:40]}...")
