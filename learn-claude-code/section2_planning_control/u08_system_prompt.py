"""
U08 - System Prompt（系统提示词）
=================================
本文件演示 **System Prompt** 机制：如何通过系统提示词定义 Agent 的行为。
使用原生 Anthropic SDK 实现。

核心概念：
  1. System Prompt 定义了 Agent 的角色、能力和行为规范
  2. 它在每轮对话中都会被发送给模型，但不会显示在对话历史中
  3. Claude Code 的 System Prompt 非常复杂，包含多个层次

Claude Code 的 System Prompt 结构：
  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code 的 system prompt 包含以下部分：              │
  │                                                          │
  │  1. 角色定义：你是谁，能做什么                            │
  │  2. 工具指南：如何使用各种工具                            │
  │  3. 行为规范：回复风格、安全约束                          │
  │  4. 环境信息：当前目录、平台、git 状态                    │
  │  5. 规则（Rules）：从 CLAUDE.md 加载的项目规则            │
  │  6. 技能（Skills）：当前激活的技能                        │
  │  7. MCP 服务器：可用的外部工具                            │
  │  8. 日期和上下文：当前日期、会话信息                      │
  │                                                          │
  │  这些部分在每次 API 调用时动态组装。                      │
  └──────────────────────────────────────────────────────────┘

为什么 System Prompt 如此重要？
  - 它是 Agent 行为的"宪法"，定义了所有行为边界
  - 好的 system prompt 能让 Agent 更准确、更安全地工作
  - Claude Code 的 system prompt 经过精心设计和反复迭代
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ============================================================
# 初始化客户端
# ============================================================
client, MODEL = create_client()


# ════════════════════════════════════════════════════════════
# 第一部分：System Prompt 的各个 Section
# ════════════════════════════════════════════════════════════

# Claude Code 的 system prompt 由多个 section 组成。
# 每个 section 负责定义 Agent 的一个方面。
# 这种模块化设计使得 system prompt 易于维护和扩展。

SECTION_ROLE = """You are an interactive agent that helps users with software engineering tasks.
You are operating in the directory: {working_dir}
Platform: {platform}
Current git branch: {git_branch}"""

SECTION_TOOL_GUIDE = """## Tool Usage Guidelines
- Use dedicated tools instead of bash when available
- Read files with the Read tool, not cat/head/tail
- Edit files with the Edit tool, not sed/awk
- Search files with Glob and Grep tools
- Use Bash only for system commands that require shell execution
- Use WebFetch for fetching web content
- Use WebSearch for searching the web"""

SECTION_BEHAVIOR = """## Behavior Guidelines
- Be concise and direct in responses
- Prefer editing existing files over creating new ones
- Don't add features beyond what was asked
- Don't add comments to code you didn't change
- Verify before destructive operations
- Ask for clarification when requirements are ambiguous
- Use Chinese when the user communicates in Chinese"""

SECTION_SAFETY = """## Safety Rules
- Never expose secrets or credentials in output
- Don't execute destructive commands without confirmation
- Validate inputs before processing
- Handle errors gracefully, don't crash
- Log important operations for auditability"""


# ════════════════════════════════════════════════════════════
# 第二部分：build_system_prompt() - 动态构建
# ════════════════════════════════════════════════════════════

def build_system_prompt(
    working_dir: str,
    platform: str = "linux",
    git_branch: str = "main",
    rules: list[str] = None,
    extra_sections: dict[str, str] = None,
) -> str:
    """
    动态构建完整的 System Prompt。

    Claude Code 在每次 API 调用时都会重新构建 system prompt，
    因为环境信息（如 git 状态、当前目录）可能随时变化。

    构建顺序：
      1. 角色定义（包含环境信息）
      2. 工具使用指南
      3. 行为规范
      4. 安全规则
      5. 项目规则（从 CLAUDE.md 加载）
      6. 额外 sections（如日期、技能等）

    Args:
        working_dir:     当前工作目录
        platform:        操作系统平台（linux / darwin / win32）
        git_branch:      当前 git 分支名
        rules:           从 CLAUDE.md 和 rules/ 加载的规则列表
        extra_sections:  额外的 prompt section（标题 → 内容）

    Returns:
        str: 完整的 system prompt 文本
    """
    sections = []

    # Section 1: 角色定义（包含环境信息）
    sections.append(SECTION_ROLE.format(
        working_dir=working_dir,
        platform=platform,
        git_branch=git_branch,
    ))

    # Section 2: 工具使用指南
    sections.append(SECTION_TOOL_GUIDE)

    # Section 3: 行为规范
    sections.append(SECTION_BEHAVIOR)

    # Section 4: 安全规则
    sections.append(SECTION_SAFETY)

    # Section 5: 项目规则
    # 在 Claude Code 中，规则从以下位置加载：
    #   - 项目根目录的 CLAUDE.md
    #   - ~/.claude/CLAUDE.md（全局规则）
    #   - ~/.claude/rules/ 中的规则文件
    if rules:
        rules_text = "\n".join(f"- {rule}" for rule in rules)
        sections.append(f"## Rules\n{rules_text}")

    # Section 6: 额外 sections
    # 用于注入动态信息，如当前日期、git 状态等
    if extra_sections:
        for title, content in extra_sections.items():
            sections.append(f"## {title}\n{content}")

    return "\n\n".join(sections)


# ════════════════════════════════════════════════════════════
# 第三部分：SystemPromptBuilder - 构建器模式
# ════════════════════════════════════════════════════════════

class SystemPromptBuilder:
    """
    System Prompt 构建器。

    使用构建器模式（Builder Pattern）来组装 system prompt。
    这种设计使得 prompt 的构建过程更加灵活和可读。

    使用示例：
        builder = SystemPromptBuilder("/home/user/project")
        builder.add_rule("测试覆盖率不低于 80%")
        builder.add_section("Current Date", "2026-07-31")
        prompt = builder.build()

    Claude Code 中的动态部分包括：
      - 当前日期
      - Git 状态（分支、未提交的变更）
      - 可用的 MCP 服务器
      - 活动的 Skills
      - Todo 列表状态
      - 用户的 CLAUDE.md 规则
    """

    def __init__(self, working_dir: str):
        """
        初始化构建器。

        Args:
            working_dir: 当前工作目录
        """
        self.working_dir = working_dir
        self.platform = "linux"
        self.git_branch = "main"
        self.rules: list[str] = []
        self.extra_sections: dict[str, str] = {}

    def set_platform(self, platform: str) -> "SystemPromptBuilder":
        """设置平台信息（支持链式调用）。"""
        self.platform = platform
        return self

    def set_git_branch(self, branch: str) -> "SystemPromptBuilder":
        """设置 git 分支（支持链式调用）。"""
        self.git_branch = branch
        return self

    def add_rule(self, rule: str) -> "SystemPromptBuilder":
        """
        添加一条规则。

        规则通常从 CLAUDE.md 或 rules/ 目录加载。
        它们定义了 Agent 在此项目中应该遵循的约束。

        Args:
            rule: 规则文本

        Returns:
            self（支持链式调用）
        """
        self.rules.append(rule)
        return self

    def add_rules(self, rules: list[str]) -> "SystemPromptBuilder":
        """批量添加规则。"""
        self.rules.extend(rules)
        return self

    def add_section(self, title: str, content: str) -> "SystemPromptBuilder":
        """
        添加一个额外的 section。

        用于注入动态信息，如当前日期、git 状态等。

        Args:
            title:   section 标题
            content: section 内容

        Returns:
            self（支持链式调用）
        """
        self.extra_sections[title] = content
        return self

    def build(self) -> str:
        """
        构建完整的 system prompt。

        Returns:
            str: 完整的 system prompt 文本
        """
        return build_system_prompt(
            working_dir=self.working_dir,
            platform=self.platform,
            git_branch=self.git_branch,
            rules=self.rules,
            extra_sections=self.extra_sections,
        )


# ════════════════════════════════════════════════════════════
# 第四部分：规则加载示例
# ════════════════════════════════════════════════════════════

def load_rules_example() -> list[str]:
    """
    模拟 Claude Code 加载规则的过程。

    Claude Code 从以下位置加载规则（按优先级）：
      1. 项目根目录的 CLAUDE.md
         - 项目特定的规则和约定
         - 例如：编码风格、分支策略

      2. 项目子目录的 CLAUDE.md
         - 子目录特定的规则
         - 例如：frontend/ 目录有自己的规则

      3. ~/.claude/CLAUDE.md（全局规则）
         - 用户级别的偏好
         - 对所有项目生效

      4. ~/.claude/rules/ 中的规则文件
         - 按语言和领域组织的规则
         - 例如：rules/python/coding-style.md

    规则的优先级：子目录 > 项目根目录 > 全局

    Returns:
        list[str]: 模拟加载的规则列表
    """
    rules = []

    # 模拟从项目 CLAUDE.md 加载
    rules.append("遵循 PEP 8 编码规范")
    rules.append("测试覆盖率不低于 80%")
    rules.append("不要在代码中硬编码密钥")

    # 模拟从 rules/common/ 加载
    rules.append("优先使用不可变数据结构")
    rules.append("函数不超过 50 行")
    rules.append("文件不超过 800 行")

    # 模拟从 rules/python/ 加载
    rules.append("使用 type annotations")
    rules.append("使用 black 格式化代码")
    rules.append("使用 pytest 编写测试")

    return rules


# ════════════════════════════════════════════════════════════
# 第五部分：Agent 调用（使用构建的 system prompt）
# ════════════════════════════════════════════════════════════

def run_agent(query: str, system_prompt: str) -> str:
    """
    使用构建好的 system prompt 运行 Agent。

    Args:
        query:         用户输入
        system_prompt: 系统提示词

    Returns:
        str: Agent 的回答
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )

    result = ""
    for block in response.content:
        if hasattr(block, "text"):
            result += block.text
    return result


# ════════════════════════════════════════════════════════════
# 第六部分：程序入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  U08 - System Prompt（系统提示词）构建演示")
    print("=" * 60)

    # ── 演示 1：使用构建器创建 system prompt ──
    print("\n── 演示 1：使用 SystemPromptBuilder 构建 ──\n")

    builder = SystemPromptBuilder(working_dir="/home/user/my-project")

    # 设置环境信息
    builder.set_platform("linux").set_git_branch("feature/auth")

    # 添加规则（模拟从 CLAUDE.md 加载）
    for rule in load_rules_example():
        builder.add_rule(rule)

    # 添加动态 section
    today = datetime.now().strftime("%Y-%m-%d")
    builder.add_section("Current Date", f"Today's date is {today}.")
    builder.add_section("Git Status", "Current branch: feature/auth\n3 files changed, 120 insertions(+), 15 deletions(-)")

    # 构建并输出
    prompt = builder.build()
    print("生成的 System Prompt：")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print(f"\nPrompt 长度: {len(prompt)} 字符")

    # ── 演示 2：链式调用 ──
    print("\n\n── 演示 2：链式调用构建 ──\n")

    prompt2 = (
        SystemPromptBuilder("/home/user/api-server")
        .set_platform("darwin")
        .set_git_branch("main")
        .add_rule("使用 FastAPI 框架")
        .add_rule("所有接口必须有 OpenAPI 文档")
        .add_section("Current Date", f"Today's date is {today}.")
        .build()
    )
    print("链式调用构建的 prompt：")
    print("-" * 50)
    print(prompt2)
    print("-" * 50)

    # ── 演示 3：对比不同配置 ──
    print("\n\n── 演示 3：不同项目配置的对比 ──\n")

    # Python 项目
    python_prompt = (
        SystemPromptBuilder("/home/user/python-api")
        .add_rule("使用 Python 3.12+")
        .add_rule("使用 type annotations")
        .add_rule("使用 pytest 编写测试")
        .build()
    )

    # Web 前端项目
    web_prompt = (
        SystemPromptBuilder("/home/user/react-app")
        .add_rule("使用 TypeScript")
        .add_rule("使用 React 18+")
        .add_rule("使用 Vitest 编写测试")
        .add_rule("CSS 使用 Tailwind CSS")
        .build()
    )

    print(f"Python 项目 prompt: {len(python_prompt)} 字符")
    print(f"Web 前端项目 prompt: {len(web_prompt)} 字符")
    print("\n两个项目的 system prompt 结构相同，但规则不同，")
    print("这使得 Agent 能够适应不同的项目环境。")

    # ── 演示 4：prompt 的层次结构 ──
    print("\n\n── 演示 4：System Prompt 的层次结构 ──\n")
    print("""
    System Prompt 的组装过程：

    ┌─────────────────────────────────────┐
    │  Section 1: 角色定义                 │  ← 基础层：你是谁
    │  "You are an interactive agent..."  │
    ├─────────────────────────────────────┤
    │  Section 2: 工具指南                 │  ← 能力层：能做什么
    │  "Use dedicated tools..."           │
    ├─────────────────────────────────────┤
    │  Section 3: 行为规范                 │  ← 行为层：怎么做
    │  "Be concise and direct..."         │
    ├─────────────────────────────────────┤
    │  Section 4: 安全规则                 │  ← 约束层：不能做什么
    │  "Never expose secrets..."          │
    ├─────────────────────────────────────┤
    │  Section 5: 项目规则                 │  ← 项目层：项目特定约束
    │  "- 测试覆盖率 >= 80%"              │
    │  "- 使用 type annotations"          │
    ├─────────────────────────────────────┤
    │  Section 6: 动态信息                 │  ← 上下文层：当前状态
    │  "Current Date: 2026-07-31"         │
    │  "Git branch: feature/auth"         │
    └─────────────────────────────────────┘

    每一层都为 Agent 的行为提供了指导和约束。
    这种分层设计使得 system prompt 既有通用性，又有灵活性。
    """)

    # ── 交互模式 ──
    print("=" * 60)
    print("  交互模式")
    print("  输入问题，使用构建好的 system prompt 运行 Agent")
    print("  输入 q 退出")
    print("=" * 60 + "\n")

    # 使用第一个演示构建的 prompt
    while True:
        try:
            query = input("\033[36mu08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "quit", "exit"):
            print("bye!")
            break

        try:
            response = run_agent(query, prompt)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print()
