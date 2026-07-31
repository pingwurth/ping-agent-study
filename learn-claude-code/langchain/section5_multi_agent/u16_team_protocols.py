"""
U16 - Team Protocols（团队协议）
=================================
本文件演示代理团队的 **协作协议**：代理之间如何通信和协调。
使用 LangGraph 消息传递和 Send API 实现。

核心概念：
  1. 协议定义了代理之间的交互规则
  2. 消息格式：代理间通信的标准格式
  3. 冲突解决：当多个代理意见不一致时如何处理
  4. 状态同步：如何保持团队成员的共享状态

LangGraph 实现方式：
  ┌──────────────────────────────────────────────────────────┐
  │  LangGraph 的消息传递机制：                               │
  │                                                          │
  │  - State 中的 messages 字段 = 共享消息队列                │
  │  - 每个节点可以读取和追加消息                             │
  │  - Send API = 定向消息传递                               │
  │  - 条件边 = 基于消息内容的路由                            │
  └──────────────────────────────────────────────────────────┘

协作模式：
  - 主从模式：主代理分配任务，从代理执行
  - 对等模式：代理之间平等协商
  - 审查模式：一个代理审查另一个代理的工作
  - 管道模式：代理按顺序处理
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import deque


# ── 消息类型 ──────────────────────────────────────────────
class MessageType(str, Enum):
    """代理间通信的消息类型"""
    TASK = "task"
    RESULT = "result"
    FEEDBACK = "feedback"
    QUERY = "query"
    STATUS = "status"
    ERROR = "error"


class Priority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# ── 消息结构 ──────────────────────────────────────────────
@dataclass
class Message:
    """
    代理间通信的消息。

    在 LangGraph 中，消息通常使用 LangChain 的消息类型：
      - HumanMessage: 用户输入
      - AIMessage: AI 回复
      - ToolMessage: 工具结果
      - SystemMessage: 系统指令

    这里定义的 Message 类用于更细粒度的代理间通信。
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_agent: str = ""
    to_agent: str = ""
    msg_type: MessageType = MessageType.TASK
    content: Any = None
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    in_reply_to: Optional[str] = None


# ── 消息总线 ──────────────────────────────────────────────
class MessageBus:
    """
    消息总线：代理间通信的基础设施。

    在 LangGraph 中，消息总线的功能由 State 的共享字段实现：
      class AgentState(TypedDict):
          messages: list[Message]
          tasks: list[dict]
          results: dict[str, str]
    """

    def __init__(self):
        self.queues: dict[str, deque] = {}
        self.history: list[Message] = []
        self.subscribers: dict[str, list[callable]] = {}

    def register_agent(self, agent_name: str):
        """注册一个代理到消息总线。"""
        if agent_name not in self.queues:
            self.queues[agent_name] = deque()

    def send(self, message: Message) -> bool:
        """发送消息到目标代理。"""
        if message.to_agent not in self.queues:
            return False

        self.queues[message.to_agent].append(message)
        self.history.append(message)

        for callback in self.subscribers.get(message.to_agent, []):
            callback(message)

        return True

    def broadcast(self, from_agent: str, msg_type: MessageType, content: Any):
        """广播消息给所有代理。"""
        for agent_name in self.queues:
            if agent_name != from_agent:
                self.send(Message(
                    from_agent=from_agent,
                    to_agent=agent_name,
                    msg_type=msg_type,
                    content=content,
                ))

    def receive(self, agent_name: str) -> Optional[Message]:
        """从代理的队列中接收消息。"""
        if agent_name in self.queues and self.queues[agent_name]:
            return self.queues[agent_name].popleft()
        return None

    def subscribe(self, agent_name: str, callback: callable):
        """订阅代理的消息。"""
        self.subscribers.setdefault(agent_name, []).append(callback)

    def get_history(self, agent_name: str = None) -> list[Message]:
        """获取消息历史。"""
        if agent_name:
            return [m for m in self.history
                    if m.from_agent == agent_name or m.to_agent == agent_name]
        return self.history


# ── 协作协议 ──────────────────────────────────────────────
class CollaborationProtocol:
    """
    协作协议：定义代理团队的交互规则。

    在 LangGraph 中，这些协议通过图结构实现：
      - 请求-响应 = 节点调用子图
      - 审查循环 = 条件边
      - 广播 = Send API fan-out
    """

    def __init__(self, bus: MessageBus):
        self.bus = bus

    def request_response(
        self,
        requester: str,
        responder: str,
        task: str,
        timeout: float = 30,
    ) -> Optional[str]:
        """请求-响应模式。"""
        request = Message(
            from_agent=requester,
            to_agent=responder,
            msg_type=MessageType.TASK,
            content=task,
        )
        self.bus.send(request)

        start = time.time()
        while time.time() - start < timeout:
            response = self.bus.receive(requester)
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
        """审查循环模式。"""
        feedback_history = []
        current_work = work

        for round_num in range(max_rounds):
            review_request = Message(
                from_agent=author,
                to_agent=reviewer,
                msg_type=MessageType.TASK,
                content=f"审查以下工作:\n{current_work}",
            )
            self.bus.send(review_request)

            feedback = f"审查反馈 (第 {round_num + 1} 轮): 无重大问题"
            feedback_history.append(feedback)

            if "无重大问题" in feedback or "CRITICAL" not in feedback:
                break

            current_work = f"{current_work}\n\n[根据反馈修改: {feedback}]"

        return current_work, feedback_history


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Team Protocols 团队协议演示\n")

    bus = MessageBus()

    agents = ["orchestrator", "coder", "reviewer", "tester"]
    for agent in agents:
        bus.register_agent(agent)

    print("── 注册的代理 ──")
    for agent in agents:
        print(f"  - {agent}")

    print("\n── 消息发送演示 ──")

    bus.send(Message(
        from_agent="orchestrator",
        to_agent="coder",
        msg_type=MessageType.TASK,
        content="实现用户登录功能",
    ))
    print("  orchestrator → coder: 任务分配")

    bus.send(Message(
        from_agent="coder",
        to_agent="reviewer",
        msg_type=MessageType.TASK,
        content="请审查登录功能代码",
    ))
    print("  coder → reviewer: 请求审查")

    bus.send(Message(
        from_agent="reviewer",
        to_agent="coder",
        msg_type=MessageType.FEEDBACK,
        content="代码质量良好，建议添加输入验证",
        in_reply_to=bus.history[-1].message_id,
    ))
    print("  reviewer → coder: 审查反馈")

    bus.broadcast("orchestrator", MessageType.STATUS, "项目进度: 60%")
    print("  orchestrator → all: 广播状态")

    print("\n── 消息历史 ──")
    for msg in bus.history:
        print(f"  [{msg.msg_type.value:8s}] {msg.from_agent} → {msg.to_agent}: {msg.content[:40]}...")

    print("\n── LangGraph 实现方式 ──")
    print("""
    # 使用 State 共享消息
    class AgentState(TypedDict):
        messages: list[Message]
        current_agent: str

    # 使用 Send API 定向传递
    from langgraph.types import Send

    def route_message(state):
        return Send(state["current_agent"], state)

    # 使用条件边实现审查循环
    def should_continue_review(state):
        if "CRITICAL" in state["review"]:
            return "coder"
        return END
    """)
