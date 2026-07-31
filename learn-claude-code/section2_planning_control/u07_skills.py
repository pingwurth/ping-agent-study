"""
U07 - Skills（技能系统）
========================
本文件演示 **Skills** 机制：预定义的专业知识包。
使用原生 Anthropic SDK 实现。

核心概念：
  1. Skill 是预定义的专业知识和行为模板
  2. 用户可以通过 /skill-name 触发特定技能
  3. 技能被展开为完整的 prompt，指导 Agent 的行为
  4. 技能可以引用外部资源（文档、最佳实践等）

技能 vs 规则（Rules）：
  ┌──────────────────────────────────────────────────────────┐
  │  规则（Rules）：                                          │
  │    - 通用的标准和约定                                     │
  │    - 始终生效，不需要显式触发                              │
  │    - 例如："测试覆盖率不低于 80%"                          │
  │                                                          │
  │  技能（Skills）：                                         │
  │    - 特定任务的详细操作指南                                │
  │    - 需要显式触发或自动检测                                │
  │    - 例如："如何用 TDD 流程开发功能"                       │
  │                                                          │
  │  技能本质上是一段结构化的 prompt，                         │
  │  当被激活时，它会被注入到系统提示词中，                     │
  │  指导 Agent 按照特定的流程执行任务。                       │
  └──────────────────────────────────────────────────────────┘

Claude Code 中的 Skills：
  - 位于 .claude/skills/ 目录下的 Markdown 文件
  - 用户可以通过 /skill-name 命令触发
  - Agent 也可以根据上下文自动检测需要的技能
  - 技能文件会被展开为完整的 prompt 注入到对话中
"""

import os
import sys
import json
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ============================================================
# 初始化客户端
# ============================================================
client, MODEL = create_client()


# ════════════════════════════════════════════════════════════
# 第一部分：技能定义
# ════════════════════════════════════════════════════════════

# 每个技能包含以下字段：
#   - name:        技能名称（用于 /skill-name 触发）
#   - description: 简短描述（用于列表展示）
#   - trigger:     触发条件（什么时候应该使用这个技能）
#   - prompt:      技能的完整提示词（核心内容）
#
# prompt 是技能的核心：它定义了 Agent 在执行此技能时应该遵循的步骤。
# 这些提示词通常很长，包含详细的步骤说明和最佳实践。

SKILLS = {
    "commit": {
        "name": "commit",
        "description": "生成规范的 git commit 消息",
        "trigger": "用户要求提交代码时",
        "prompt": """你正在执行 /commit 技能。按以下步骤操作：

步骤 1：分析变更
- 运行 `git status` 查看所有变更的文件
- 运行 `git diff` 查看具体修改内容
- 运行 `git log --oneline -5` 查看最近的提交风格

步骤 2：确定提交类型
根据变更的性质选择合适的类型：
  feat:     新功能
  fix:      Bug 修复
  refactor: 重构（不改变功能）
  docs:     文档更新
  test:     测试相关
  chore:    构建/工具变更
  perf:     性能优化
  ci:       CI/CD 相关

步骤 3：生成提交消息
格式：<type>: <description>
  - description 用中文简明描述变更内容
  - 如果有多个不相关的变更，建议分开提交

步骤 4：执行提交
- 使用 `git add` 暂存相关文件
- 使用 `git commit` 提交
- 不要自动 push，除非用户明确要求""",
    },

    "review-pr": {
        "name": "review-pr",
        "description": "审查 Pull Request",
        "trigger": "用户要求审查 PR 时",
        "prompt": """你正在执行 /review-pr 技能。按以下步骤审查：

步骤 1：获取变更
- 运行 `git diff main...HEAD` 查看所有变更
- 运行 `git log main..HEAD` 查看提交历史

步骤 2：安全性检查（优先级最高）
- 是否有硬编码的密钥、密码、API Key
- 是否有 SQL 注入风险（字符串拼接查询）
- 是否有 XSS 漏洞（未转义的用户输入）
- 是否有路径遍历风险
- 是否缺少 CSRF 保护

步骤 3：代码质量检查
- 函数是否过长（建议 < 50 行）
- 文件是否过大（建议 < 800 行）
- 嵌套是否过深（建议 < 4 层）
- 是否有重复代码
- 命名是否清晰

步骤 4：测试覆盖检查
- 新功能是否有对应的测试
- 测试是否覆盖了边界情况
- 测试是否独立且可重复

步骤 5：输出审查报告
按严重程度分类：
  CRITICAL: 安全漏洞或数据丢失风险（必须修复）
  HIGH:     Bug 或重大质量问题（应该修复）
  MEDIUM:   可维护性问题（建议修复）
  LOW:      风格或次要建议（可选）""",
    },

    "tdd-workflow": {
        "name": "tdd-workflow",
        "description": "测试驱动开发工作流",
        "trigger": "用户要求实现新功能或修复 bug 时",
        "prompt": """你正在执行 /tdd-workflow 技能。严格按 TDD 流程执行：

═══ RED 阶段（先写测试）═══
1. 根据需求分析要测试的行为
2. 编写失败的测试用例
   - 使用 AAA 模式：Arrange - Act - Assert
   - 测试名称应该描述期望的行为
   - 包含正常路径和边界情况
3. 运行测试，确认它确实失败
   - 如果测试意外通过，说明需求理解有误

═══ GREEN 阶段（最小实现）═══
4. 编写最少的代码让测试通过
   - 不要添加测试未覆盖的功能
   - 不要考虑代码质量，只要能通过
5. 运行测试，确认它通过

═══ REFACTOR 阶段（重构优化）═══
6. 在测试保护下重构代码
   - 提取重复逻辑
   - 改善命名
   - 简化复杂逻辑
7. 运行测试，确认仍然通过

═══ 验证 ═══
8. 检查测试覆盖率 >= 80%
9. 确保所有测试都是绿色

关键原则：
  - 永远不要在没有失败测试的情况下写实现代码
  - 测试应该描述行为，而不是实现细节
  - 每次只让一个测试从红变绿""",
    },

    "security-review": {
        "name": "security-review",
        "description": "安全审查（OWASP Top 10）",
        "trigger": "涉及认证、支付、用户数据的代码变更",
        "prompt": """你正在执行 /security-review 技能。按 OWASP Top 10 逐项检查：

1. 注入攻击（Injection）
   - SQL 注入：检查所有数据库查询是否使用参数化
   - NoSQL 注入：检查 MongoDB 等查询
   - OS 命令注入：检查 subprocess/system 调用

2. 认证失效（Broken Authentication）
   - 密码策略：最小长度、复杂度要求
   - 会话管理：session 超时、安全 cookie 标志
   - 多因素认证：关键操作是否要求二次验证

3. 敏感数据暴露（Sensitive Data Exposure）
   - 传输加密：是否使用 HTTPS
   - 存储加密：密码是否使用 bcrypt/argon2
   - 日志脱敏：日志中是否包含敏感信息

4. XXE（XML External Entities）
   - XML 解析是否禁用外部实体

5. 访问控制（Broken Access Control）
   - IDOR：是否通过 ID 直接访问资源
   - 权限检查：每个接口是否验证权限

6. 安全配置错误（Security Misconfiguration）
   - 默认密码是否修改
   - 错误信息是否泄露实现细节
   - 不必要的服务是否关闭

7. XSS（Cross-Site Scripting）
   - 反射型：URL 参数是否转义
   - 存储型：用户输入是否净化
   - DOM 型：前端是否安全处理

8. 不安全的反序列化（Insecure Deserialization）
   - 是否反序列化不可信的数据

9. 已知漏洞的组件（Using Components with Known Vulnerabilities）
   - 依赖版本是否过时

10. 日志和监控不足（Insufficient Logging）
    - 安全事件是否记录
    - 是否有异常检测机制

输出安全审查报告，标注每个发现的风险等级。""",
    },
}


# ════════════════════════════════════════════════════════════
# 第二部分：SkillManager - 技能管理器
# ════════════════════════════════════════════════════════════

class SkillManager:
    """
    管理和触发技能。

    技能的工作流程：
      ① 用户输入 /skill-name 或 Agent 识别匹配的技能
      ② SkillManager 查找对应的技能定义
      ③ 将技能的 prompt 注入到系统提示词中
      ④ Agent 按照技能的指导执行任务

    关键方法：
      - list_skills():   列出所有可用技能
      - activate():      激活指定技能，返回其 prompt
      - detect_skill():  根据用户输入自动检测应该使用的技能
    """

    def __init__(self):
        self.skills = SKILLS.copy()
        self.active_skill: Optional[str] = None

    def list_skills(self) -> list[dict]:
        """
        列出所有可用技能。

        Returns:
            list[dict]: 技能列表，每个包含 name 和 description
        """
        return [
            {"name": s["name"], "description": s["description"]}
            for s in self.skills.values()
        ]

    def activate(self, skill_name: str) -> Optional[str]:
        """
        激活一个技能，返回其 prompt。

        激活后的 prompt 应该被注入到系统提示词中，
        让 Agent 在后续对话中按照技能的指导执行。

        Args:
            skill_name: 技能名称

        Returns:
            str 或 None: 技能的 prompt，如果技能不存在返回 None
        """
        skill = self.skills.get(skill_name)
        if not skill:
            return None

        self.active_skill = skill_name
        return skill["prompt"]

    def detect_skill(self, user_input: str) -> Optional[str]:
        """
        根据用户输入自动检测应该使用的技能。

        检测策略（按优先级）：
          1. 显式触发：检查是否包含 /skill-name
          2. 关键词匹配：检查是否包含相关关键词

        Args:
            user_input: 用户输入文本

        Returns:
            str 或 None: 匹配的技能名称，无匹配返回 None
        """
        input_lower = user_input.lower()

        # 检查显式的 /skill-name 触发
        for name in self.skills:
            if f"/{name}" in input_lower:
                return name

        # 关键词匹配
        # 注意：实际应用中应该使用更智能的匹配方式
        # 这里为了演示简单使用关键词
        keyword_map = {
            "commit": ["提交", "commit", "git commit", "git push"],
            "review-pr": ["审查", "review", "pr", "pull request", "代码审查"],
            "tdd-workflow": ["测试", "test", "tdd", "单元测试", "写测试"],
            "security-review": ["安全", "security", "漏洞", "owasp", "安全审查"],
        }

        for skill_name, keywords in keyword_map.items():
            if any(word in input_lower for word in keywords):
                return skill_name

        return None


# ════════════════════════════════════════════════════════════
# 第三部分：Agent 集成（技能注入到系统提示词）
# ════════════════════════════════════════════════════════════

def build_system_prompt_with_skill(skill_prompt: Optional[str] = None) -> str:
    """
    构建包含技能的系统提示词。

    当技能被激活时，它的 prompt 会被注入到系统提示词中。
    这样 Agent 在处理用户请求时，会遵循技能定义的流程。

    Args:
        skill_prompt: 技能的 prompt（如果有的话）

    Returns:
        str: 完整的系统提示词
    """
    base_prompt = f"""你是一个编程助手，工作在 {os.getcwd()} 目录。
用中文回复。"""

    if skill_prompt:
        return f"{base_prompt}\n\n{skill_prompt}"
    return base_prompt


def run_with_skill(query: str, skill_manager: SkillManager) -> str:
    """
    使用技能运行 Agent。

    流程：
      1. 检测是否需要激活技能
      2. 如果需要，激活技能并注入 prompt
      3. 调用 Claude API

    Args:
        query:          用户输入
        skill_manager:  技能管理器实例

    Returns:
        str: Agent 的回答
    """
    # 自动检测技能
    detected_skill = skill_manager.detect_skill(query)
    skill_prompt = None

    if detected_skill:
        print(f"  [自动检测] 激活技能: /{detected_skill}")
        skill_prompt = skill_manager.activate(detected_skill)

    # 构建系统提示词
    system_prompt = build_system_prompt_with_skill(skill_prompt)

    # 调用 Claude API
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )

    # 提取回答文本
    result = ""
    for block in response.content:
        if hasattr(block, "text"):
            result += block.text
    return result


# ════════════════════════════════════════════════════════════
# 第四部分：程序入口
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  U07 - Skills（技能系统）演示")
    print("=" * 60)

    manager = SkillManager()

    # ── 演示 1：列出所有技能 ──
    print("\n── 可用技能列表 ──\n")
    for skill in manager.list_skills():
        print(f"  /{skill['name']:<20} {skill['description']}")

    # ── 演示 2：激活技能并查看 prompt ──
    print("\n\n── 激活 /tdd-workflow 技能 ──\n")
    prompt = manager.activate("tdd-workflow")
    if prompt:
        print("技能 prompt 预览（前 500 字符）：")
        print("-" * 50)
        print(prompt[:500])
        print("-" * 50)
        print(f"总长度: {len(prompt)} 字符")

    # ── 演示 3：自动检测技能 ──
    print("\n\n── 自动检测技能演示 ──\n")
    test_inputs = [
        "帮我提交代码",
        "审查这个 PR",
        "为这个函数写测试",
        "检查一下安全性",
        "帮我写个排序算法",      # 无匹配技能
        "帮我做一次安全审查",    # 匹配 security-review
        "/commit",              # 显式触发
    ]

    for inp in test_inputs:
        detected = manager.detect_skill(inp)
        skill_name = f"/{detected}" if detected else "无匹配"
        print(f"  '{inp}' → {skill_name}")

    # ── 演示 4：技能 prompt 结构 ──
    print("\n\n── 所有技能的 prompt 结构 ──\n")
    for name, skill in SKILLS.items():
        print(f"  /{name}:")
        print(f"    描述: {skill['description']}")
        print(f"    触发: {skill['trigger']}")
        print(f"    prompt 长度: {len(skill['prompt'])} 字符")
        # 显示 prompt 的前几行
        first_lines = skill["prompt"].split("\n")[:3]
        for line in first_lines:
            print(f"    | {line}")
        print()
