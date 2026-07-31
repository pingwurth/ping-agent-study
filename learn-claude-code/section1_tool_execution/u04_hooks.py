#!/usr/bin/env python3
"""
u04_hooks.py - Hooks（钩子系统）实现
======================================
本文件演示 **Hooks（钩子系统）** 机制：在工具执行的特定时机自动运行代码。

核心概念：
  1. Hooks 是在工具执行生命周期中自动触发的脚本
  2. 三种类型：
     - PreToolUse: 工具执行前触发（可用于安全检查、参数验证）
     - PostToolUse: 工具执行后触发（可用于格式化、日志记录）
     - Stop: 会话结束时触发（可用于清理、最终验证）
  3. PreToolUse 的非零退出码会阻止工具执行（关键安全机制）
  4. Hooks 通过环境变量接收上下文信息（HOOK_TYPE, TOOL_NAME, FILE_PATH）

Claude Code 中的 Hooks 配置（.claude/settings.json）：
  {
    "hooks": {
      "PreToolUse": [
        {"matcher": "Write|Edit", "command": "prettier --check \"$FILE_PATH\""}
      ],
      "PostToolUse": [
        {"matcher": "Write|Edit", "command": "eslint --fix \"$FILE_PATH\""}
      ]
    }
  }

Hook 执行流程：
  模型请求工具 → 匹配 PreToolUse hooks → 执行 → 检查退出码
    → 退出码非 0: 阻止执行，返回错误
    → 退出码 0: 允许执行 → 执行工具 → 运行 PostToolUse hooks
"""

import os
import sys
import json
import time
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import create_client

# ── readline 中文修复 ──────────────────────────────────────
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

# ============================================================
# 初始化客户端
# ============================================================
client, MODEL = create_client()

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ============================================================
# Hook 类型常量
# ============================================================
# 对应 Claude Code 中的三种 Hook 类型
PRE_TOOL_USE = "PreToolUse"
POST_TOOL_USE = "PostToolUse"
STOP = "Stop"


# ============================================================
# Hook 管理器
# ============================================================
class HookManager:
    """
    管理和执行 Hooks 的核心类。

    注册格式：
      hooks.register("PreToolUse", "bash", "check_command.sh")
      hooks.register("PostToolUse", "Write|Edit", "prettier --write $FILE_PATH")

    matcher 支持 | 分隔多个工具名，如 "Write|Edit" 匹配这两个工具。
    """

    def __init__(self):
        # 按 Hook 类型分组存储
        self.hooks: dict[str, list[dict]] = {
            PRE_TOOL_USE: [],
            POST_TOOL_USE: [],
            STOP: [],
        }
        # 执行日志 —— 记录每次 Hook 执行的详细信息
        self.execution_log: list[dict] = []

    def register(self, hook_type: str, matcher: str, command: str) -> None:
        """
        注册一个 Hook。

        参数：
          hook_type: Hook 类型（PreToolUse / PostToolUse / Stop）
          matcher: 匹配工具名的模式（支持 | 分隔多个名称）
          command: 要执行的 shell 命令
        """
        if hook_type not in self.hooks:
            print(f"[警告] 未知的 Hook 类型: {hook_type}")
            return

        self.hooks[hook_type].append({
            "matcher": matcher,
            "command": command,
        })
        print(f"[Hook 注册] {hook_type} | 匹配: {matcher} | 命令: {command}")

    def _matches(self, matcher: str, tool_name: str) -> bool:
        """
        检查工具名称是否匹配 matcher 模式。

        匹配规则：
          - "bash" 匹配工具名 "bash"
          - "Write|Edit" 匹配工具名 "Write" 或 "Edit"
          - "*" 匹配所有工具
        """
        if matcher == "*":
            return True
        patterns = matcher.split("|")
        return tool_name in patterns

    def run_hooks(self, hook_type: str, tool_name: str = "",
                  file_path: str = "") -> dict:
        """
        执行指定类型的所有匹配 Hooks。

        环境变量（供 Hook 脚本使用）：
          - HOOK_TYPE: Hook 类型（PreToolUse / PostToolUse / Stop）
          - TOOL_NAME: 触发 Hook 的工具名
          - FILE_PATH: 操作的文件路径（如果有）

        返回值：
          {"allowed": bool, "output": str}
          - allowed: 是否允许继续执行（仅 PreToolUse 有意义）
          - output: Hook 脚本的输出

        关键行为：
          - PreToolUse: 退出码非 0 → allowed=False，阻止工具执行
          - PostToolUse/Stop: 退出码不影响流程，仅收集输出
          - 超时: 30 秒后强制终止
        """
        results = {"allowed": True, "output": ""}

        for hook in self.hooks.get(hook_type, []):
            # 检查是否匹配当前工具
            if not self._matches(hook["matcher"], tool_name):
                continue

            # 构建环境变量 —— Hook 脚本通过这些变量获取上下文
            env = os.environ.copy()
            env["HOOK_TYPE"] = hook_type
            env["TOOL_NAME"] = tool_name
            env["FILE_PATH"] = file_path

            try:
                result = subprocess.run(
                    hook["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=30,
                )

                # 记录执行日志
                log_entry = {
                    "hook_type": hook_type,
                    "tool_name": tool_name,
                    "command": hook["command"],
                    "exit_code": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "timestamp": time.time(),
                }
                self.execution_log.append(log_entry)

                print(f"  [Hook] {hook['command']} → 退出码: {result.returncode}")

                # PreToolUse 的关键行为：退出码非 0 表示阻止执行
                if hook_type == PRE_TOOL_USE and result.returncode != 0:
                    results["allowed"] = False
                    results["output"] = f"[Hook 阻止] {result.stderr or result.stdout}"
                    return results

                # 收集输出
                if result.stdout:
                    results["output"] += result.stdout

            except subprocess.TimeoutExpired:
                print(f"  [Hook] 超时: {hook['command']}")
                results["output"] += f"[Hook 超时: {hook['command']}]\n"
            except Exception as e:
                print(f"  [Hook] 错误: {e}")
                results["output"] += f"[Hook 错误: {e}]\n"

        return results


# ============================================================
# 工具定义
# ============================================================
BASH_TOOL = {
    "name": "bash",
    "description": "在终端中执行 bash 命令并返回输出结果。",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 bash 命令",
            }
        },
        "required": ["command"],
    },
}


def execute_bash(command: str) -> str:
    """执行 bash 命令。"""
    try:
        result = subprocess.run(
            command, shell=True,
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(命令执行成功，无输出)"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT] 命令执行超过 120 秒超时"
    except Exception as e:
        return f"[ERROR] 执行失败: {e}"


# ============================================================
# Agent 循环（带 Hook 支持）
# ============================================================

def print_response_content(response):
    """打印模型回复的内容块。"""
    for block in response.content:
        if block.type == "text":
            print(f"\n[模型回复] {block.text}")
        elif block.type == "tool_use":
            print(f"\n[工具调用] {block.name}({json.dumps(block.input, ensure_ascii=False)})")


def agent_loop(messages: list, hook_manager: HookManager) -> None:
    """
    带 Hook 支持的 Agent 循环。

    与 u01 的区别：
      - u01 直接执行工具
      - 这里在执行前运行 PreToolUse hooks
      - 执行后运行 PostToolUse hooks
      - PreToolUse hook 可以阻止工具执行

    执行顺序：
      ① 模型返回 tool_use
      ② 运行匹配的 PreToolUse hooks
      ③ 如果 PreToolUse 返回 allowed=False → 跳过执行
      ④ 如果 allowed=True → 执行工具
      ⑤ 运行匹配的 PostToolUse hooks
      ⑥ 将结果返回给模型
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            system=SYSTEM,
            tools=[BASH_TOOL],
            messages=messages,
        )

        print_response_content(response)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                command = block.input.get("command", "")

                # ① 运行 PreToolUse hooks
                print(f"\n[执行工具] {block.name}...")
                print(f"  [PreToolUse] 检查中...")
                pre_result = hook_manager.run_hooks(
                    PRE_TOOL_USE,
                    tool_name=block.name,
                    file_path=command,
                )

                if not pre_result["allowed"]:
                    # PreToolUse hook 阻止了执行
                    print(f"  [PreToolUse] 被阻止: {pre_result['output']}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": pre_result["output"],
                    })
                    continue

                # ② 执行工具
                if block.name == "bash":
                    output = execute_bash(command)
                else:
                    output = f"[ERROR] 未知工具: {block.name}"

                print(f"[工具输出]\n{output}")

                # ③ 运行 PostToolUse hooks
                print(f"  [PostToolUse] 执行中...")
                post_result = hook_manager.run_hooks(
                    POST_TOOL_USE,
                    tool_name=block.name,
                    file_path=command,
                )
                if post_result["output"]:
                    print(f"  [PostToolUse 输出] {post_result['output']}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user", "content": tool_results})

    # 会话结束时运行 Stop hooks
    print("\n[Stop] 运行结束钩子...")
    hook_manager.run_hooks(STOP)


# ============================================================
# 程序入口：交互式 REPL
# ============================================================
if __name__ == "__main__":
    print(f"[Hook 系统 Agent] 模型: {MODEL}")
    print(f"[系统提示] {SYSTEM}")
    print("[工具] bash（带 Hook 支持）")
    print("-" * 50)

    # 初始化 Hook 管理器
    hook_manager = HookManager()

    # 注册示例 Hooks
    # PreToolUse: 阻止删除根目录的命令
    hook_manager.register(
        PRE_TOOL_USE,
        matcher="bash",
        command='echo "$TOOL_NAME" | grep -q "rm -rf /" && exit 1 || exit 0',
    )

    # PostToolUse: 记录所有工具调用到日志文件
    hook_manager.register(
        POST_TOOL_USE,
        matcher="bash",
        command='echo "[$(date)] Tool: $TOOL_NAME | Command: $FILE_PATH" >> /tmp/agent_hook.log',
    )

    # Stop: 会话结束时的清理操作
    hook_manager.register(
        STOP,
        matcher="*",
        command='echo "[$(date)] Session ended" >> /tmp/agent_hook.log',
    )

    print("-" * 50)

    messages = []

    while True:
        try:
            user_input = input("\033[36mu04 >> \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("再见！")
                break

            messages.append({"role": "user", "content": user_input})
            agent_loop(messages, hook_manager)

        except KeyboardInterrupt:
            print("\n\n中断退出。")
            # 中断时也运行 Stop hooks
            hook_manager.run_hooks(STOP)
            break
        except EOFError:
            print("\n\n再见！")
            hook_manager.run_hooks(STOP)
            break

    # 打印 Hook 执行日志摘要
    if hook_manager.execution_log:
        print(f"\n[Hook 日志] 共执行 {len(hook_manager.execution_log)} 次 Hook")
        for entry in hook_manager.execution_log:
            status = "成功" if entry["exit_code"] == 0 else f"失败(退出码 {entry['exit_code']})"
            print(f"  {entry['hook_type']} | {entry['tool_name']} | {status}")
