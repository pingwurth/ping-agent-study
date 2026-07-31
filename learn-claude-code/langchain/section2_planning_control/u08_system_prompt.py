"""
U08 - System Prompt（系统提示词）
=================================
本文件演示 **System Prompt** 机制：如何通过系统提示词定义 Agent 的行为。
使用 LangChain ChatPromptTemplate 实现。

核心概念：
  1. System Prompt 定义了 Agent 的角色、能力和行为规范
  2. 它在每轮对话中都会被发送给模型，但不会显示在对话历史中
  3. Claude Code 的 System Prompt 非常复杂，包含多个层次

LangChain 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  使用 ChatPromptTemplate 动态组装 system prompt：         │
  │                                                          │
  │  prompt = ChatPromptTemplate.from_messages([             │
  │      SystemMessagePromptTemplate.from_template(          │
  │          "You are a coding agent at {working_dir}..."    │
  │      ),                                                  │
  │      MessagesPlaceholder("messages"),                    │
  │  ])                                                      │
  │                                                          │
  │  优点：                                                   │
  │    - 模板化：支持变量注入                                  │
  │    - 组合化：多个 template 可以拼接                        │
  │    - 类型安全：SystemMessage / HumanMessage / AIMessage   │
  └──────────────────────────────────────────────────────────┘
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder

model = get_model()


# ── System Prompt 模板 ───────────────────────────────────
# 使用 LangChain 的 SystemMessagePromptTemplate 定义各个 section
# 每个 section 是一个独立的模板，最终拼接为完整的 system prompt

SECTION_ROLE = SystemMessagePromptTemplate.from_template(
    """You are an interactive agent that helps users with software engineering tasks.
You are operating in the directory: {working_dir}
Platform: {platform}
Current git branch: {git_branch}"""
)

SECTION_TOOL_GUIDE = SystemMessagePromptTemplate.from_template(
    """## Tool Usage Guidelines
- Use dedicated tools instead of bash when available
- Read files with the Read tool, not cat/head/tail
- Edit files with the Edit tool, not sed/awk
- Search files with Glob and Grep tools
- Use Bash only for system commands that require shell execution"""
)

SECTION_BEHAVIOR = SystemMessagePromptTemplate.from_template(
    """## Behavior Guidelines
- Be concise and direct in responses
- Prefer editing existing files over creating new ones
- Don't add features beyond what was asked
- Don't add comments to code you didn't change
- Verify before destructive operations"""
)


def build_system_prompt(
    working_dir: str,
    platform: str = "linux",
    git_branch: str = "main",
    rules: list[str] = None,
    extra_sections: dict[str, str] = None,
) -> str:
    """
    动态构建 System Prompt。

    Claude Code 在每次 API 调用时都会重新构建 system prompt，
    因为环境信息（如 git 状态、当前目录）可能随时变化。

    Args:
        working_dir: 当前工作目录
        platform: 操作系统平台
        git_branch: 当前 git 分支
        rules: 从 CLAUDE.md 和 rules/ 加载的规则列表
        extra_sections: 额外的 prompt section

    Returns:
        str: 完整的 system prompt
    """
    sections = []

    # Section 1: 基础角色定义
    sections.append(SECTION_ROLE.format(
        working_dir=working_dir,
        platform=platform,
        git_branch=git_branch,
    ))

    # Section 2: 工具使用指南
    sections.append(SECTION_TOOL_GUIDE.format())

    # Section 3: 行为规范
    sections.append(SECTION_BEHAVIOR.format())

    # Section 4: 规则
    if rules:
        rules_text = "\n".join(rules)
        sections.append(f"## Rules\n{rules_text}")

    # Section 5: 额外 sections
    if extra_sections:
        for title, content in extra_sections.items():
            sections.append(f"## {title}\n{content}")

    return "\n\n".join(sections)


# ── 动态上下文注入 ────────────────────────────────────────
class SystemPromptBuilder:
    """
    System Prompt 构建器。

    Claude Code 的 system prompt 不是静态的，而是每次调用时动态构建。
    动态部分包括：
      - 当前日期
      - Git 状态（分支、未提交的变更）
      - 可用的 MCP 服务器
      - 活动的 Skills
      - Todo 列表状态
      - 用户的 CLAUDE.md 规则

    在 LangChain 中，可以将 SystemPromptBuilder 的输出
    作为 ChatPromptTemplate 的 system message 内容。
    """

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.rules: list[str] = []
        self.extra_sections: dict[str, str] = {}

    def add_rule(self, rule: str):
        """添加一条规则。"""
        self.rules.append(rule)

    def add_section(self, title: str, content: str):
        """添加一个额外的 section。"""
        self.extra_sections[title] = content

    def build(self) -> str:
        """构建完整的 system prompt。"""
        return build_system_prompt(
            working_dir=self.working_dir,
            rules=self.rules,
            extra_sections=self.extra_sections,
        )

    def to_prompt_template(self) -> ChatPromptTemplate:
        """
        转换为 LangChain 的 ChatPromptTemplate。

        将构建好的 system prompt 作为 system message，
        加上 MessagesPlaceholder 用于对话历史。
        """
        system_content = self.build()
        return ChatPromptTemplate.from_messages([
            ("system", system_content),
            MessagesPlaceholder("messages"),
        ])


# ── 规则加载示例 ──────────────────────────────────────────
def load_rules_example() -> list[str]:
    """
    模拟 Claude Code 加载规则的过程。

    Claude Code 从以下位置加载规则：
      1. 项目根目录的 CLAUDE.md
      2. 项目子目录的 CLAUDE.md
      3. ~/.claude/CLAUDE.md（全局规则）
      4. ~/.claude/rules/ 中的规则文件
    """
    rules = []

    # 模拟从 CLAUDE.md 加载
    rules.append("- 遵循 PEP 8 编码规范")
    rules.append("- 测试覆盖率不低于 80%")
    rules.append("- 不要在代码中硬编码密钥")

    # 模拟从 rules/common/ 加载
    rules.append("- 优先使用不可变数据结构")
    rules.append("- 函数不超过 50 行")
    rules.append("- 文件不超过 800 行")

    # 模拟从 rules/python/ 加载
    rules.append("- 使用 type annotations")
    rules.append("- 使用 black 格式化代码")

    return rules


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("System Prompt 构建演示\n")

    # 创建构建器
    builder = SystemPromptBuilder(working_dir="/home/user/my-project")

    # 添加规则
    for rule in load_rules_example():
        builder.add_rule(rule)

    # 添加额外 section
    builder.add_section("Current Date", "Today's date is 2026-07-31.")
    builder.add_section("Git Status", "Current branch: main\nNo uncommitted changes.")

    # 构建并输出
    prompt = builder.build()
    print("── 生成的 System Prompt ──")
    print(prompt)
    print(f"\n── Prompt 长度: {len(prompt)} 字符 ──")

    # 转换为 LangChain ChatPromptTemplate
    print("\n── 转换为 ChatPromptTemplate ──")
    template = builder.to_prompt_template()
    print(f"模板消息数: {len(template.messages)}")
    for msg in template.messages:
        print(f"  - {msg.__class__.__name__}")
