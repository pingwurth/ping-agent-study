"""
U16 - Team Protocols（团队协议）
=================================
本文件演示 Claude Code 代理团队的 **协作协议**：代理之间如何通信和协调。

核心概念：
  1. 协议定义了代理之间的交互规则
  2. 消息格式：代理间通信的标准格式
  3. 消息总线：代理间通信的基础设施
  4. 协作模式：请求-响应、审查循环、广播等

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code 代理间通信机制：                              │
  │                                                          │
  │  1. 主代理通过 SendMessage 工具向子代理发送消息           │
  │  2. 子代理完成后通过通知机制返回结果                      │
  │  3. 使用 Agent ID 标识消息来源和目标                      │
  │  4. 支持同步和异步消息传递                                │
  └──────────────────────────────────────────────────────────┘

协作模式：
  ┌──────────────────────────────────────────────────────────┐
  │  1. 请求-响应（Request-Response）                         │
  │     代理 A 发送请求 → 代理 B 处理 → 代理 B 返回响应      │
  │                                                          │
  │  2. 审查循环（Review Cycle）                              │
  │     作者提交代码 → 审查者审查 → 作者修改 → 审查者确认    │
  │                                                          │
  │  3. 广播（Broadcast）                                     │
  │     一个代理向所有代理发送消息（如状态更新）              │
  │                                                          │
  │  4. 管道（Pipeline）                                      │
  │     代理 A → 代理 B → 代理 C（链式传递）                 │
  └──────────────────────────────────────────────────────────┘

本文件是纯 Python 实现，不依赖 anthropic SDK。
使用消息队列和回调机制模拟代理间通信。
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import deque


# ══════════════════════════════════════════════════════════════
# 第一部分：消息类型和优先级
# ══════════════════════════════════════════════════════════════

class MessageType(str, Enum):
    """
    代理间通信的消息类型。

    Claude Code 中的消息类型：
      - TASK:     任务分配（主代理 → 子代理）
      - RESULT:   任务结果（子代理 → 主代理）
      - FEEDBACK: 反馈信息（审查者 → 作者）
      - QUERY:    查询请求（代理 A → 代理 B）
      - STATUS:   状态更新（广播给所有代理）
      - ERROR:    错误通知
    """
    TASK = "task"
    RESULT = "result"
    FEEDBACK = "feedback"
    QUERY = "query"
    STATUS = "status"
    ERROR = "error"


class Priority(str, Enum):
    """
    消息优先级。

    高优先级消息会被优先处理。
    Claude Code 中：
      - HIGH:   安全问题、构建失败
      - NORMAL: 普通任务分配
      - LOW:    状态更新、日志
    """
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# ══════════════════════════════════════════════════════════════
# 第二部分：消息结构
# ══════════════════════════════════════════════════════════════

@dataclass
class Message:
    """
    代理间通信的消息。

    对应 Claude Code 的 SendMessage 工具参数：
    {
        "to": "agent-id",
        "summary": "简短摘要",
        "message": "详细内容"
    }

    字段说明：
      - message_id:  消息唯一标识符
      - from_agent:  发送方代理名称
      - to_agent:    接收方代理名称
      - msg_type:    消息类型
      - content:     消息内容（可以是任意类型）
      - priority:    消息优先级
      - timestamp:   消息创建时间戳
      - in_reply_to: 回复的消息 ID（用于关联请求和响应）
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_agent: str = ""
    to_agent: str = ""
    msg_type: MessageType = MessageType.TASK
    content: Any = None
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    in_reply_to: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# 第三部分：消息总线
# ══════════════════════════════════════════════════════════════

class MessageBus:
    """
    消息总线：代理间通信的基础设施。

    提供以下功能：
      1. register_agent() - 注册代理到消息总线
      2. send()           - 发送消息到目标代理
      3. broadcast()      - 广播消息给所有代理
      4. receive()        - 从代理的队列中接收消息
      5. subscribe()      - 订阅代理的消息（回调）
      6. get_history()    - 获取消息历史

    实现原理：
      - 每个代理有一个消息队列（deque）
      - 发送消息时，放入目标代理的队列
      - 接收消息时，从自己的队列中取出
      - 订阅机制支持消息到达时的回调

    Claude Code 中的对应关系：
      - MessageBus    → Claude 的 Agent 通信机制
      - send()        → SendMessage 工具
      - broadcast()   → 状态更新通知
      - receive()     → 子代理完成通知
    """

    def __init__(self):
        # 代理消息队列：agent_name → deque[Message]
        self.queues: dict[str, deque] = {}
        # 消息历史（所有已发送的消息）
        self.history: list[Message] = []
        # 订阅者：agent_name → [callback, ...]
        self.subscribers: dict[str, list[callable]] = {}

    def register_agent(self, agent_name: str):
        """
        注册一个代理到消息总线。

        注册后，代理就有了自己的消息队列，可以收发消息。

        Args:
            agent_name: 代理名称（唯一标识）
        """
        if agent_name not in self.queues:
            self.queues[agent_name] = deque()

    def send(self, message: Message) -> bool:
        """
        发送消息到目标代理。

        流程：
          1. 检查目标代理是否已注册
          2. 将消息放入目标代理的队列
          3. 记录到消息历史
          4. 触发目标代理的订阅者回调

        Args:
            message: 要发送的消息

        Returns:
            bool: 是否发送成功
        """
        if message.to_agent not in self.queues:
            return False

        # 放入目标代理的队列
        self.queues[message.to_agent].append(message)
        # 记录历史
        self.history.append(message)

        # 触发订阅者回调
        for callback in self.subscribers.get(message.to_agent, []):
            callback(message)

        return True

    def broadcast(self, from_agent: str, msg_type: MessageType, content: Any):
        """
        广播消息给所有代理（除了发送者）。

        对应 Claude Code 的状态更新通知：
          - 一个代理完成任务后，通知所有其他代理
          - 用于同步团队状态

        Args:
            from_agent: 发送者名称
            msg_type:   消息类型
            content:    消息内容
        """
        for agent_name in self.queues:
            if agent_name != from_agent:
                self.send(Message(
                    from_agent=from_agent,
                    to_agent=agent_name,
                    msg_type=msg_type,
                    content=content,
                ))

    def receive(self, agent_name: str) -> Optional[Message]:
        """
        从代理的队列中接收消息。

        非阻塞操作：如果队列为空，立即返回 None。

        Args:
            agent_name: 代理名称

        Returns:
            Optional[Message]: 消息对象，队列为空则返回 None
        """
        if agent_name in self.queues and self.queues[agent_name]:
            return self.queues[agent_name].popleft()
        return None

    def subscribe(self, agent_name: str, callback: callable):
        """
        订阅代理的消息。

        当代理收到消息时，会调用注册的回调函数。
        用于实现实时通知机制。

        Args:
            agent_name: 要订阅的代理名称
            callback:   回调函数，接收 Message 参数
        """
        self.subscribers.setdefault(agent_name, []).append(callback)

    def get_history(self, agent_name: str = None) -> list[Message]:
        """
        获取消息历史。

        Args:
            agent_name: 如果指定，只返回与该代理相关的消息

        Returns:
            list[Message]: 消息列表
        """
        if agent_name:
            return [
                m for m in self.history
                if m.from_agent == agent_name or m.to_agent == agent_name
            ]
        return self.history


# ══════════════════════════════════════════════════════════════
# 第四部分：协作协议
# ══════════════════════════════════════════════════════════════

class CollaborationProtocol:
    """
    协作协议：定义代理团队的交互规则。

    提供两种常用协作模式：
      1. request_response() - 请求-响应模式
      2. review_cycle()     - 审查循环模式

    这些模式在 Claude Code 中的对应：
      - 请求-响应 → 主代理发送任务给子代理，等待结果
      - 审查循环  → 编码-审查-修改的迭代过程
    """

    def __init__(self, bus: MessageBus):
        """
        初始化协作协议。

        Args:
            bus: 消息总线实例
        """
        self.bus = bus

    def request_response(
        self,
        requester: str,
        responder: str,
        task: str,
        timeout: float = 30,
    ) -> Optional[str]:
        """
        请求-响应模式。

        流程：
          1. requester 发送 TASK 消息给 responder
          2. 等待 responder 的响应（RESULT 消息）
          3. 通过 in_reply_to 关联请求和响应
          4. 超时返回 None

        Args:
            requester: 请求方代理名称
            responder: 响应方代理名称
            task:      任务描述
            timeout:   超时时间（秒）

        Returns:
            Optional[str]: 响应内容，超时返回 None
        """
        # 创建请求消息
        request = Message(
            from_agent=requester,
            to_agent=responder,
            msg_type=MessageType.TASK,
            content=task,
        )
        self.bus.send(request)

        # 等待响应
        start = time.time()
        while time.time() - start < timeout:
            response = self.bus.receive(requester)
            # 检查是否是对应请求的响应
            if response and response.in_reply_to == request.message_id:
                return response.content
            time.sleep(0.1)

        return None

    def review_cycle(
        self,
        author: str,
        reviewer: str,
        work: str,
        max_rounds: int = 3,
    ) -> tuple[str, list[str]]:
        """
        审查循环模式。

        流程：
          1. 作者提交工作给审查者
          2. 审查者审查并给出反馈
          3. 如果有严重问题，作者修改后重新提交
          4. 重复直到审查通过或达到最大轮次

        Args:
            author:     作者代理名称
            reviewer:   审查者代理名称
            work:       要审查的工作内容
            max_rounds: 最大审查轮次

        Returns:
            tuple[str, list[str]]: (最终工作内容, 反馈历史)
        """
        feedback_history = []
        current_work = work

        for round_num in range(max_rounds):
            # 发送审查请求
            review_request = Message(
                from_agent=author,
                to_agent=reviewer,
                msg_type=MessageType.TASK,
                content=f"审查以下工作:\n{current_work}",
            )
            self.bus.send(review_request)

            # 模拟审查反馈（实际中由审查者代理生成）
            feedback = f"审查反馈 (第 {round_num + 1} 轮): 无重大问题"
            feedback_history.append(feedback)

            # 如果没有严重问题，结束审查
            if "无重大问题" in feedback or "CRITICAL" not in feedback:
                break

            # 根据反馈修改工作
            current_work = f"{current_work}\n\n[根据反馈修改: {feedback}]"

        return current_work, feedback_history


# ══════════════════════════════════════════════════════════════
# 第五部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U16 - Team Protocols 团队协议演示")
    print("=" * 60)

    # 创建消息总线
    bus = MessageBus()

    # ── 注册代理 ──────────────────────────────────────────
    print("\n── 注册代理 ──")
    agents = ["orchestrator", "coder", "reviewer", "tester"]
    for agent in agents:
        bus.register_agent(agent)
        print(f"  已注册: {agent}")

    # ── 消息发送演示 ──────────────────────────────────────
    print("\n── 消息发送演示 ──")

    # 1. 任务分配：orchestrator → coder
    bus.send(Message(
        from_agent="orchestrator",
        to_agent="coder",
        msg_type=MessageType.TASK,
        content="实现用户登录功能",
    ))
    print("  orchestrator → coder: 任务分配 (TASK)")

    # 2. 请求审查：coder → reviewer
    review_msg = Message(
        from_agent="coder",
        to_agent="reviewer",
        msg_type=MessageType.TASK,
        content="请审查登录功能代码",
    )
    bus.send(review_msg)
    print("  coder → reviewer: 请求审查 (TASK)")

    # 3. 审查反馈：reviewer → coder
    bus.send(Message(
        from_agent="reviewer",
        to_agent="coder",
        msg_type=MessageType.FEEDBACK,
        content="代码质量良好，建议添加输入验证",
        in_reply_to=review_msg.message_id,  # 关联到审查请求
    ))
    print("  reviewer → coder: 审查反馈 (FEEDBACK)")

    # 4. 广播状态
    bus.broadcast("orchestrator", MessageType.STATUS, "项目进度: 60%")
    print("  orchestrator → all: 广播状态 (STATUS)")

    # ── 接收消息演示 ──────────────────────────────────────
    print("\n── 接收消息演示 ──")

    # coder 接收消息
    msg = bus.receive("coder")
    if msg:
        print(f"  coder 收到: [{msg.msg_type.value}] {msg.content}")

    # reviewer 接收消息
    msg = bus.receive("reviewer")
    if msg:
        print(f"  reviewer 收到: [{msg.msg_type.value}] {msg.content}")

    # ── 消息历史 ──────────────────────────────────────────
    print("\n── 消息历史 ──")
    for msg in bus.history:
        content_preview = str(msg.content)[:40]
        print(f"  [{msg.msg_type.value:8s}] {msg.from_agent} → {msg.to_agent}: {content_preview}...")

    # ── 订阅机制演示 ──────────────────────────────────────
    print("\n── 订阅机制演示 ──")

    received_messages = []

    def on_message(msg: Message):
        received_messages.append(msg)
        print(f"  [订阅回调] tester 收到消息: {msg.content}")

    bus.subscribe("tester", on_message)

    # 发送消息给 tester，会触发回调
    bus.send(Message(
        from_agent="orchestrator",
        to_agent="tester",
        msg_type=MessageType.TASK,
        content="运行测试套件",
    ))

    # ── 协作协议演示 ──────────────────────────────────────
    print("\n── 协作协议演示 ──")

    protocol = CollaborationProtocol(bus)

    print("  请求-响应模式:")
    print("    result = protocol.request_response(")
    print('        requester="orchestrator",')
    print('        responder="coder",')
    print('        task="实现用户注册功能",')
    print("        timeout=30")
    print("    )")

    print("\n  审查循环模式:")
    print("    work, feedback = protocol.review_cycle(")
    print('        author="coder",')
    print('        reviewer="reviewer",')
    print('        work="def login(): ...",')
    print("        max_rounds=3")
    print("    )")

    # ── Claude Code 通信机制说明 ──────────────────────────
    print("\n── Claude Code 代理通信机制说明 ──")
    print("""
    Claude Code 使用 SendMessage 工具实现代理间通信：

    1. 发送消息：
       SendMessage(
           to="agent-id",
           summary="任务分配",
           message="请实现用户认证功能"
       )

    2. 消息路由：
       - 主代理 → 子代理: 任务分配
       - 子代理 → 主代理: 结果返回（通过通知）
       - 代理 → 代理: 直接通信（通过 SendMessage）

    3. 异步通信：
       - 子代理在后台运行
       - 完成时通过通知机制告知主代理
       - 主代理可以继续其他工作

    4. 消息过滤：
       - 使用 agentId 或 name 标识目标
       - 支持向特定类型的代理发送消息
    """)
