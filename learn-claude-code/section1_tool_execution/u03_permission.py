#!/usr/bin/env python3
"""
u03_permission.py - 权限控制系统
==================================
本文件演示 **权限控制（Permission System）** 机制：如何在工具执行前进行安全审查。

核心概念：
  1. 不是所有工具调用都应该自动执行
  2. 危险操作（如 rm -rf、写入系统文件）需要用户确认
  3. 权限系统在工具执行前拦截，询问用户是否允许
  4. Claude Code 支持三种权限模式：自动允许、需确认、禁止

Claude Code 中的权限实现：
  - 每个工具有权限级别（auto、confirm、deny）
  - 用户可以通过 "always allow" 跳过后续同类确认
  - 某些操作（如删除文件）始终需要确认

权限检查流程：
  模型请求工具调用 → 检查是否危险 → 危险则询问用户 → 允许则执行，拒绝则跳过
"""

import os
import sys
import json
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
# 危险命令检测
# ============================================================
# Claude Code 有更完善的权限模型（基于工具类型 + 路径规则），
# 这里简化为命令模式匹配。

DANGEROUS_PATTERNS = [
    "rm -rf",        # 递归强制删除
    "rm -f /",       # 删除根目录文件
    "sudo rm",       # sudo 删除
    "mkfs",          # 格式化文件系统
    "dd if=",        # 直接磁盘写入
    "> /dev/sd",     # 覆写磁盘设备
    "chmod 777",     # 开放所有权限
    "curl | bash",   # 远程代码执行
    "wget | bash",   # 远程代码执行
    ":(){:|:&};:",   # fork bomb
]


def is_dangerous(command: str) -> bool:
    """
    检查命令是否为危险命令。

    实际的 Claude Code 使用更精细的规则：
      - 按工具类型（Read/Write/Bash/Edit）分类
      - 按路径模式匹配（如 /etc/*、~/.ssh/*）
      - 支持用户自定义规则
    """
    cmd_lower = command.lower().strip()
    return any(pattern in cmd_lower for pattern in DANGEROUS_PATTERNS)


def request_permission(command: str) -> bool:
    """
    请求用户确认是否允许执行命令。

    Claude Code 中的权限 UI 更丰富：
      - 显示完整的命令和上下文
      - 提供 "always allow" 选项（本会话内生效）
      - 记录用户的权限决策历史
    """
    print(f"\n\033[33m[权限检查] 检测到可能的危险命令\033[0m")
    print(f"\033[33m  命令: {command}\033[0m")

    while True:
        response = input(
            "\033[33m  允许执行？[y/n/a] (y=允许, n=拒绝, a=本会话自动允许): \033[0m"
        ).strip().lower()

        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        elif response == "a":
            # 设置自动允许标志，后续同类命令不再询问
            print("\033[33m  已设置本会话自动允许\033[0m")
            return "auto_approve"
        print("  请输入 y/n/a")


def execute_with_permission(command: str, auto_approve: bool = False) -> str:
    """
    带权限检查的命令执行。

    流程：
      1. 检查是否自动允许模式
      2. 如果不是自动允许，检查命令是否危险
      3. 危险命令需要用户确认
      4. 确认通过后执行命令
    """
    # 自动允许模式：跳过所有检查
    if auto_approve:
        return execute_bash(command)

    # 检查是否危险
    if is_dangerous(command):
        result = request_permission(command)
        if result == "auto_approve":
            # 用户选择自动允许，设置标志
            return "__SET_AUTO_APPROVE__"
        elif not result:
            return "[DENIED] 用户拒绝执行该命令"

    return execute_bash(command)


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


# ============================================================
# Agent 循环（带权限控制）
# ============================================================

def print_response_content(response):
    """打印模型回复的内容块。"""
    for block in response.content:
        if block.type == "text":
            print(f"\n[模型回复] {block.text}")
        elif block.type == "tool_use":
            print(f"\n[工具调用] {block.name}({json.dumps(block.input, ensure_ascii=False)})")


def agent_loop(messages: list, auto_approve: bool = False) -> None:
    """
    带权限控制的 Agent 循环。

    与 u01 的区别：
      - u01 直接执行所有工具调用
      - 这里在执行前进行权限检查
      - 危险命令会询问用户确认
      - 用户可以选择自动允许（auto_approve）

    auto_approve 参数：
      - False: 每次危险命令都询问（默认）
      - True: 跳过所有权限检查（用户之前选择了 'a'）
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
                print(f"\n[执行工具] {block.name}...")

                if block.name == "bash":
                    # 使用带权限检查的执行器
                    output = execute_with_permission(
                        block.input["command"],
                        auto_approve=auto_approve,
                    )

                    # 处理自动允许标志
                    if output == "__SET_AUTO_APPROVE__":
                        auto_approve = True
                        # 用户选择自动允许后，重新执行命令
                        output = execute_bash(block.input["command"])
                else:
                    output = f"[ERROR] 未知工具: {block.name}"

                print(f"[工具输出]\n{output}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user", "content": tool_results})

    return auto_approve


# ============================================================
# 程序入口：交互式 REPL
# ============================================================
if __name__ == "__main__":
    print(f"[权限控制 Agent] 模型: {MODEL}")
    print(f"[系统提示] {SYSTEM}")
    print("[工具] bash（带权限检查）")
    print("[提示] 输入危险命令（如 'rm -rf /'）会触发权限检查")
    print("-" * 50)

    messages = []
    auto_approve = False

    while True:
        try:
            user_input = input("\033[36mu03 >> \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("再见！")
                break

            messages.append({"role": "user", "content": user_input})
            # agent_loop 返回更新后的 auto_approve 状态
            auto_approve = agent_loop(messages, auto_approve=auto_approve)

        except KeyboardInterrupt:
            print("\n\n中断退出。")
            break
        except EOFError:
            print("\n\n再见！")
            break
