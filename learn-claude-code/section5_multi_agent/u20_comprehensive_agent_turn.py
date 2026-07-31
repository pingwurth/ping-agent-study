"""
U20 - Comprehensive Agent Turn（完整的代理轮次）
==================================================
本文件演示一个 **完整的代理轮次** 是如何工作的。
整合所有前面章节的概念，使用 anthropic SDK 实现。

核心概念：
  1. 一个 Agent Turn = 接收用户输入 → 处理 → 输出响应
  2. 每个 turn 可能包含多轮工具调用
  3. 涉及前面所有章节的概念：工具、权限、Hook、任务管理等

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  一个完整的 Agent Turn 流程：                              │
  │                                                          │
  │  ① 接收用户输入                                          │
  │  ② 构建请求（system prompt + tools + messages）           │
  │  ③ 调用 Claude API (messages.create)                     │
  │  ④ 处理响应：                                             │
  │     - text → 返回给用户                                   │
  │     - tool_use → 执行工具                                 │
  │  ⑤ 执行工具（权限检查 + Hook）                            │
  │  ⑥ 将工具结果加入消息历史                                 │
  │  ⑦ 循环回到 ③                                            │
  │  ⑧ 输出最终响应                                           │
  └──────────────────────────────────────────────────────────┘

Agent Loop 的核心：
  while True:
      response = client.messages.create(
          model="claude-sonnet-4-20250514",
          system=system_prompt,
          messages=messages,
          tools=tools,
      )
      if response.stop_reason == "end_turn":
          break  # 代理完成
      if response.stop_reason == "tool_use":
          execute_tools(response.content)
          continue  # 继续循环

全课程总结：
  Section 1 - 工具执行:
    U01: Agent Loop     → messages.create + tool_use 循环
    U02: Tool Use       → JSON Schema 定义工具
    U03: Permission     → 工具调用前的权限检查
    U04: Hooks          → PreToolUse / PostToolUse 回调

  Section 2 - 规划控制:
    U05: TodoWrite      → 任务跟踪工具
    U06: Sub-Agent      → Agent 工具创建子代理
    U07: Skills         → 技能系统（预定义提示词）
    U08: System Prompt  → 动态组装系统提示词
    U09: Error Recovery → 工具执行失败的重试机制

  Section 3 - 记忆管理:
    U10: Context Compaction → 上下文压缩
    U11: Memory             → 持久化记忆

  Section 4 - 并发调度:
    U12: Background Tasks → 后台任务执行
    U13: Cron Scheduler   → 定时任务调度

  Section 5 - 多代理:
    U14: Task System      → 任务 DAG 执行
    U15: Agent Teams      → 多代理协作
    U16: Team Protocols   → 代理间通信协议
    U17: Autonomous Agents → 自主循环代理
    U18: Worktree Isolation → 隔离工作环境
    U19: MCP Tools        → 外部工具集成
    U20: Comprehensive Turn → 完整轮次（本文件）

本文件使用 anthropic SDK 实现完整的 Agent Turn。
"""

import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client


# ══════════════════════════════════════════════════════════════
# 第一部分：Agent 配置
# ══════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """
    Agent 的完整配置。

    Claude Code 在每个 turn 开始时会组装这些配置：
      - system_prompt: 系统提示词（含规则、环境信息、工具说明）
      - tools:         可用工具列表（内置 + MCP）
      - max_tokens:    最大输出 token 数
      - model:         使用的模型

    system_prompt 的组成：
      - CLAUDE.md 规则
      - 环境信息（操作系统、工作目录等）
      - 工具使用说明
      - 安全规则
    """
    system_prompt: str = ""
    tools: list = field(default_factory=list)
    max_tokens: int = 8000
    model: str = "claude-sonnet-4-20250514"


# ══════════════════════════════════════════════════════════════
# 第二部分：Agent Turn 实现
# ══════════════════════════════════════════════════════════════

class AgentTurn:
    """
    一个完整的 Agent Turn。

    实现了 Claude Code 的核心 Agent Loop：
      1. 接收用户输入
      2. 构建请求（system + tools + messages）
      3. 调用 Claude API
      4. 处理响应（文本 or 工具调用）
      5. 如果有工具调用，执行工具并继续循环
      6. 返回最终响应

    整合了所有前面章节的概念：
      - 工具执行（U01-U02）: 定义和执行工具
      - 权限控制（U03）: 工具调用前检查
      - Hook 系统（U04）: PreToolUse / PostToolUse
      - 任务管理（U05）: TodoWrite 工具
      - 上下文管理（U10）: 消息历史管理
    """

    def __init__(self, config: AgentConfig):
        """
        初始化 Agent Turn。

        Args:
            config: Agent 配置
        """
        self.config = config
        self.client, model = create_client()
        # 如果配置中使用默认模型，则用环境变量中的模型覆盖
        if self.config.model == "claude-sonnet-4-20250514":
            self.config.model = model
        # 统计信息
        self.tool_call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.turn_count = 0
        self.start_time = 0

    def process(self, user_input: str, max_turns: int = 10) -> str:
        """
        处理一个完整的用户输入。

        这是 Agent Turn 的核心方法，实现了完整的 Agent Loop：

        Args:
            user_input: 用户的输入文本
            max_turns:  最大循环次数（防止无限循环）

        Returns:
            str: Agent 的最终文本响应
        """
        self.start_time = time.time()
        self.turn_count = 0

        # 初始化消息历史
        messages = [{"role": "user", "content": user_input}]

        # Agent Loop
        for turn in range(max_turns):
            self.turn_count += 1

            # ① 调用 Claude API
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                tools=self.config.tools,
                messages=messages,
            )

            # ② 统计 token 使用
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            # ③ 处理响应
            # 提取文本内容
            text_content = self._extract_text(response.content)

            # 检查是否需要执行工具
            if response.stop_reason == "tool_use":
                # ④ 执行工具调用
                tool_results = self._execute_tools(response.content)

                # ⑤ 将助手响应和工具结果加入消息历史
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                # 继续循环
                continue

            # ⑥ 没有工具调用，返回最终响应
            return text_content

        # 达到最大轮次
        return text_content if text_content else "Max turns reached"

    def _extract_text(self, content: list) -> str:
        """
        从响应内容中提取文本。

        Claude API 的响应内容是一个 block 列表：
          - TextBlock: 文本内容
          - ToolUseBlock: 工具调用请求

        Args:
            content: 响应内容 block 列表

        Returns:
            str: 提取的文本内容
        """
        text_parts = []
        for block in content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    def _execute_tools(self, content: list) -> list[dict]:
        """
        执行响应中的所有工具调用。

        遍历响应内容，找到所有 ToolUseBlock，
        逐个执行并收集结果。

        Args:
            content: 响应内容 block 列表

        Returns:
            list[dict]: 工具结果列表（ToolMessage 格式）
        """
        tool_results = []

        for block in content:
            if block.type == "tool_use":
                # 执行单个工具
                result = self._execute_single_tool(
                    tool_name=block.name,
                    tool_input=block.input,
                    tool_use_id=block.id,
                )
                tool_results.append(result)

        return tool_results

    def _execute_single_tool(
        self,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str,
    ) -> dict:
        """
        执行单个工具调用。

        执行流程：
          1. 权限检查（U03）
          2. PreToolUse Hook（U04）
          3. 执行工具
          4. PostToolUse Hook（U04）
          5. 返回结果

        Args:
            tool_name:  工具名称
            tool_input: 工具参数
            tool_use_id: 工具调用 ID

        Returns:
            dict: ToolMessage 格式的结果
        """
        self.tool_call_count += 1

        # 权限检查（模拟）
        # 实际 Claude Code 会检查 allowedTools / blockedTools
        # print(f"  [权限检查] {tool_name}: 允许")

        # PreToolUse Hook（模拟）
        # 实际 Claude Code 会执行配置的 hook 命令
        # print(f"  [PreToolUse] {tool_name}")

        # 执行工具（简化实现）
        result_content = f"Executed {tool_name} with {json.dumps(tool_input)}"

        # 特殊处理某些工具
        if tool_name == "bash":
            result_content = self._execute_bash(tool_input.get("command", ""))
        elif tool_name == "read_file":
            result_content = self._execute_read_file(tool_input.get("path", ""))

        # PostToolUse Hook（模拟）
        # print(f"  [PostToolUse] {tool_name}")

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result_content,
        }

    def _execute_bash(self, command: str) -> str:
        """执行 bash 命令。"""
        import subprocess
        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True,
                timeout=30,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Error: {e}"

    def _execute_read_file(self, path: str) -> str:
        """读取文件内容。"""
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    def get_stats(self) -> dict:
        """
        获取本轮的统计信息。

        Returns:
            dict: 统计信息
                - turns:           循环次数
                - tool_calls:      工具调用次数
                - input_tokens:    输入 token 数
                - output_tokens:   输出 token 数
                - elapsed_seconds: 耗时
        """
        elapsed = time.time() - self.start_time
        return {
            "turns": self.turn_count,
            "tool_calls": self.tool_call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "elapsed_seconds": round(elapsed, 2),
        }


# ══════════════════════════════════════════════════════════════
# 第三部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U20 - Comprehensive Agent Turn 完整代理轮次演示")
    print("=" * 60)

    # ── 定义工具 ──────────────────────────────────────────
    # 使用 JSON Schema 定义工具（对应 U02）
    tools = [
        {
            "name": "bash",
            "description": "Run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run",
                    }
                },
                "required": ["command"],
            },
        },
        {
            "name": "read_file",
            "description": "Read file contents",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    ]

    # ── 构建 System Prompt ────────────────────────────────
    # 对应 U08: 动态组装系统提示词
    system_prompt = f"""You are a coding agent at {os.getcwd()}.
Use tools to solve tasks.

Available tools:
- bash: Execute shell commands
- read_file: Read file contents
- write_file: Write content to files

Be concise. Act, don't explain."""

    # ── 创建 Agent 配置 ───────────────────────────────────
    config = AgentConfig(
        system_prompt=system_prompt,
        tools=tools,
        max_tokens=8000,
    )

    # ── 展示 Agent Turn 流程 ──────────────────────────────
    print("\n── Agent Turn 流程 ──")
    print("""
    ┌─────────────────────────────────────────────┐
    │  用户输入: "列出当前目录的文件"               │
    ├─────────────────────────────────────────────┤
    │  ① 构建请求                                  │
    │     - system: coding agent prompt            │
    │     - tools: [bash, read_file, write_file]   │
    │     - messages: [user: "列出文件"]            │
    ├─────────────────────────────────────────────┤
    │  ② 调用 Claude API                           │
    │     response = client.messages.create(...)   │
    ├─────────────────────────────────────────────┤
    │  ③ 处理响应                                  │
    │     - stop_reason: "tool_use"                │
    │     - tool: bash("ls -la")                   │
    ├─────────────────────────────────────────────┤
    │  ④ 执行工具                                  │
    │     - 权限检查 (U03)                          │
    │     - PreToolUse Hook (U04)                   │
    │     - 执行: ls -la                            │
    │     - PostToolUse Hook (U04)                  │
    ├─────────────────────────────────────────────┤
    │  ⑤ 将结果加入消息历史                        │
    │     messages.append(tool_result)             │
    ├─────────────────────────────────────────────┤
    │  ⑥ 再次调用 Claude API                       │
    │     - 收到工具结果                            │
    │     - 生成最终回答                            │
    │     - stop_reason: "end_turn"                │
    ├─────────────────────────────────────────────┤
    │  ⑦ 输出给用户                                │
    │     "当前目录包含以下文件..."                 │
    └─────────────────────────────────────────────┘
    """)

    # ── 统计信息示例 ──────────────────────────────────────
    print("── 统计信息示例 ──")
    turn = AgentTurn(config)
    stats = turn.get_stats()
    print(f"  循环次数:     {stats['turns']}")
    print(f"  工具调用次数: {stats['tool_calls']}")
    print(f"  输入 Token:   {stats['input_tokens']}")
    print(f"  输出 Token:   {stats['output_tokens']}")
    print(f"  耗时:         {stats['elapsed_seconds']} 秒")

    # ── 全课程总结 ────────────────────────────────────────
    print("\n── 全课程总结 ──")
    print("""
    Section 1 - 工具执行:
      U01: Agent Loop     → messages.create + tool_use 循环
      U02: Tool Use       → JSON Schema 定义工具
      U03: Permission     → 工具调用前的权限检查
      U04: Hooks          → PreToolUse / PostToolUse 回调

    Section 2 - 规划控制:
      U05: TodoWrite      → 任务跟踪工具
      U06: Sub-Agent      → Agent 工具创建子代理
      U07: Skills         → 技能系统（预定义提示词）
      U08: System Prompt  → 动态组装系统提示词
      U09: Error Recovery → 工具执行失败的重试机制

    Section 3 - 记忆管理:
      U10: Context Compaction → 上下文压缩
      U11: Memory             → 持久化记忆

    Section 4 - 并发调度:
      U12: Background Tasks → 后台任务执行
      U13: Cron Scheduler   → 定时任务调度

    Section 5 - 多代理:
      U14: Task System      → 任务 DAG 执行
      U15: Agent Teams      → 多代理协作
      U16: Team Protocols   → 代理间通信协议
      U17: Autonomous Agents → 自主循环代理
      U18: Worktree Isolation → 隔离工作环境
      U19: MCP Tools        → 外部工具集成
      U20: Comprehensive Turn → 完整轮次（本文件）
    """)

    # ── Claude Code 核心机制总结 ──────────────────────────
    print("── Claude Code 核心机制总结 ──")
    print("""
    1. Agent Loop（核心循环）:
       while True:
           response = client.messages.create(...)
           if response.stop_reason == "end_turn":
               break
           execute_tools(response)
           messages.append(tool_results)

    2. 工具系统:
       - 内置工具: Bash, Read, Write, Edit, Glob, Grep, Agent...
       - MCP 工具: 通过 MCP 协议集成外部工具
       - 工具定义: JSON Schema 格式

    3. 权限控制:
       - allowedTools: 允许的工具列表
       - blockedTools: 禁止的工具列表
       - 每次工具调用前检查权限

    4. Hook 系统:
       - PreToolUse:  工具执行前（验证、参数修改）
       - PostToolUse: 工具执行后（格式化、检查）
       - Stop:        会话结束时（最终验证）

    5. 多代理系统:
       - Agent 工具: 创建子代理
       - SendMessage: 代理间通信
       - Worktree: 隔离工作环境

    6. 记忆系统:
       - 上下文压缩: 管理长对话
       - 持久化记忆: 跨会话保持
    """)
