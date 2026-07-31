"""
U07 - Skills（技能系统）
========================
本文件演示 **Skills** 机制：预定义的专业知识包。
使用 LangChain PromptTemplate 实现。

核心概念：
  1. Skill 是预定义的专业知识和行为模板
  2. 用户可以通过 /skill-name 触发特定技能
  3. 技能被展开为完整的 prompt，指导 Agent 的行为
  4. 技能可以引用外部资源（文档、最佳实践等）

LangChain 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  每个 Skill = 一个 ChatPromptTemplate                    │
  │                                                          │
  │  commit_skill = ChatPromptTemplate.from_messages([       │
  │      ("system", "你正在执行 /commit 技能..."),             │
  │      MessagesPlaceholder("messages"),                    │
  │  ])                                                      │
  │                                                          │
  │  激活技能时，用对应的 prompt 替换默认 prompt               │
  └──────────────────────────────────────────────────────────┘

技能 vs 规则（Rules）：
  - 规则：通用的标准和约定（如"80% 测试覆盖率"）
  - 技能：特定任务的详细操作指南（如"如何用 pytest 写测试"）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

model = get_model()


# ── 技能定义 ──────────────────────────────────────────────
# 技能本质上是一个结构化的 prompt 模板
# 使用 LangChain 的 ChatPromptTemplate 定义
SKILLS = {
    "commit": {
        "name": "commit",
        "description": "生成规范的 git commit 消息",
        "trigger": "用户要求提交代码时",
        "prompt": ChatPromptTemplate.from_messages([
            ("system", """你正在执行 /commit 技能。按以下步骤操作：

1. 运行 `git status` 查看所有变更
2. 运行 `git diff` 查看具体修改内容
3. 运行 `git log --oneline -5` 查看最近的提交风格
4. 分析变更的性质（feat/fix/refactor/docs/test/chore）
5. 生成符合 Conventional Commits 格式的提交消息
6. 格式：<type>: <description>

类型说明：
  feat: 新功能
  fix: Bug 修复
  refactor: 重构
  docs: 文档更新
  test: 测试相关
  chore: 构建/工具变更"""),
            MessagesPlaceholder("messages"),
        ]),
    },
    "review-pr": {
        "name": "review-pr",
        "description": "审查 Pull Request",
        "trigger": "用户要求审查 PR 时",
        "prompt": ChatPromptTemplate.from_messages([
            ("system", """你正在执行 /review-pr 技能。按以下步骤审查：

1. 运行 `git diff main...HEAD` 查看所有变更
2. 检查以下方面：
   - 安全性：是否有硬编码密钥、SQL 注入、XSS 等
   - 代码质量：函数是否过长、嵌套是否过深
   - 测试覆盖：新功能是否有测试
   - 性能：是否有 N+1 查询、缺失索引等
3. 按严重程度分类：
   - CRITICAL: 安全漏洞（必须修复）
   - HIGH: Bug 或重大问题（应该修复）
   - MEDIUM: 可维护性问题（建议修复）
   - LOW: 风格建议（可选）
4. 输出结构化的审查报告"""),
            MessagesPlaceholder("messages"),
        ]),
    },
    "tdd-workflow": {
        "name": "tdd-workflow",
        "description": "测试驱动开发工作流",
        "trigger": "用户要求实现新功能或修复 bug 时",
        "prompt": ChatPromptTemplate.from_messages([
            ("system", """你正在执行 /tdd-workflow 技能。严格按 TDD 流程执行：

RED 阶段（先写测试）：
1. 根据需求编写失败的测试用例
2. 运行测试，确认它确实失败
3. 测试应该描述期望的行为

GREEN 阶段（最小实现）：
4. 编写最少的代码让测试通过
5. 不要添加测试未覆盖的功能
6. 运行测试，确认它通过

REFACTOR 阶段（重构优化）：
7. 在测试保护下重构代码
8. 提取重复逻辑
9. 改善命名和结构
10. 运行测试，确认仍然通过

验证：
11. 检查测试覆盖率 >= 80%"""),
            MessagesPlaceholder("messages"),
        ]),
    },
    "security-review": {
        "name": "security-review",
        "description": "安全审查",
        "trigger": "涉及认证、支付、用户数据的代码变更",
        "prompt": ChatPromptTemplate.from_messages([
            ("system", """你正在执行 /security-review 技能。按 OWASP Top 10 检查：

1. 注入攻击：SQL/NoSQL/OS/LDAP 注入
2. 认证失效：会话管理、密码策略
3. 敏感数据暴露：加密、密钥管理
4. XXE：XML 外部实体攻击
5. 访问控制：权限检查、IDOR
6. 安全配置：默认配置、错误信息泄露
7. XSS：反射型/存储型/DOM 型
8. 反序列化：不安全的反序列化
9. 已知漏洞：过时的依赖
10. 日志和监控：安全事件记录

输出安全审查报告，标注风险等级。"""),
            MessagesPlaceholder("messages"),
        ]),
    },
}


# ── 技能管理器 ────────────────────────────────────────────
class SkillManager:
    """
    管理和触发技能。

    Claude Code 中技能的工作流程：
      ① 用户输入 /skill-name 或 Agent 识别匹配的技能
      ② SkillManager 查找对应的技能定义
      ③ 将技能的 prompt 展开并注入到对话中
      ④ Agent 按照技能的指导执行任务

    在 LangChain 中，技能激活就是选择对应的 ChatPromptTemplate。
    """

    def __init__(self):
        self.skills = SKILLS.copy()
        self.active_skill = None

    def list_skills(self) -> list[dict]:
        """列出所有可用技能。"""
        return [
            {"name": s["name"], "description": s["description"]}
            for s in self.skills.values()
        ]

    def activate(self, skill_name: str) -> ChatPromptTemplate:
        """
        激活一个技能。

        Args:
            skill_name: 技能名称

        Returns:
            ChatPromptTemplate: 技能的 prompt 模板
        """
        skill = self.skills.get(skill_name)
        if not skill:
            return None

        self.active_skill = skill_name
        return skill["prompt"]

    def detect_skill(self, user_input: str) -> str | None:
        """
        根据用户输入自动检测应该使用的技能。

        Args:
            user_input: 用户输入

        Returns:
            str | None: 匹配的技能名称，或 None
        """
        input_lower = user_input.lower()

        # 检查显式的 /skill-name 触发
        for name in self.skills:
            if f"/{name}" in input_lower:
                return name

        # 检查关键词匹配
        if any(word in input_lower for word in ["提交", "commit", "git commit"]):
            return "commit"
        if any(word in input_lower for word in ["审查", "review", "pr"]):
            return "review-pr"
        if any(word in input_lower for word in ["测试", "test", "tdd"]):
            return "tdd-workflow"
        if any(word in input_lower for word in ["安全", "security", "漏洞"]):
            return "security-review"

        return None


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Skills 技能系统演示\n")

    manager = SkillManager()

    # 列出所有技能
    print("── 可用技能 ──")
    for skill in manager.list_skills():
        print(f"  /{skill['name']}: {skill['description']}")

    # 演示技能激活
    print("\n── 激活 /tdd-workflow 技能 ──")
    prompt = manager.activate("tdd-workflow")
    if prompt:
        # 展示 prompt 模板的消息结构
        for msg in prompt.messages:
            print(f"  [{msg.__class__.__name__}] {str(msg)[:100]}...")

    # 演示自动检测
    print("\n── 自动检测技能 ──")
    test_inputs = [
        "帮我提交代码",
        "审查这个 PR",
        "为这个函数写测试",
        "检查一下安全性",
        "帮我写个排序算法",  # 无匹配技能
    ]
    for inp in test_inputs:
        detected = manager.detect_skill(inp)
        print(f"  '{inp}' → /{detected or '无匹配'}")
