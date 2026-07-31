"""
U19 - MCP Tools（MCP 工具）
============================
本文件演示 **MCP (Model Context Protocol)** 工具集成。
使用 LangChain MCP 适配器实现。

核心概念：
  1. MCP 是 Anthropic 定义的开放协议，用于连接 LLM 和外部工具
  2. MCP 服务器提供工具（tools）和资源（resources）
  3. Claude Code 可以连接多个 MCP 服务器
  4. MCP 工具像内置工具一样被模型调用

LangChain MCP 集成：
  ┌──────────────────────────────────────────────────────────┐
  │  使用 langchain-mcp-adapters 连接 MCP 服务器：            │
  │                                                          │
  │  from langchain_mcp_adapters.client import               │
  │      MultiServerMCPClient                                │
  │                                                          │
  │  async with MultiServerMCPClient({                       │
  │      "codegraph": {"command": "codegraph", ...},         │
  │  }) as client:                                           │
  │      tools = client.get_tools()                          │
  │      agent = create_react_agent(model, tools)            │
  └──────────────────────────────────────────────────────────┘
"""

import os
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── MCP 工具定义 ──────────────────────────────────────────
@dataclass
class MCPTool:
    """
    MCP 工具的定义。

    MCP 工具与内置工具的 JSON Schema 格式相同：
    {
        "name": "tool_name",
        "description": "What this tool does",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    """
    name: str
    description: str
    input_schema: dict
    server_name: str = ""


@dataclass
class MCPResource:
    """MCP 资源的定义。"""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


# ── MCP 服务器模拟 ────────────────────────────────────────
class MCPServer:
    """
    MCP 服务器的模拟实现。

    真实的 MCP 服务器是一个独立进程，通过 JSON-RPC 协议通信。
    这里简化为 Python 类来演示概念。
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tools: dict[str, MCPTool] = {}
        self.resources: dict[str, MCPResource] = {}

    def register_tool(self, tool: MCPTool):
        """注册一个工具。"""
        tool.server_name = self.name
        self.tools[tool.name] = tool

    def register_resource(self, resource: MCPResource):
        """注册一个资源。"""
        self.resources[resource.uri] = resource

    def list_tools(self) -> list[MCPTool]:
        """列出所有工具。"""
        return list(self.tools.values())

    def list_resources(self) -> list[MCPResource]:
        """列出所有资源。"""
        return list(self.resources.values())

    def call_tool(self, name: str, arguments: dict) -> Any:
        """调用工具。"""
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        return {"result": f"Executed {name} with {arguments}"}

    def read_resource(self, uri: str) -> str:
        """读取资源。"""
        if uri not in self.resources:
            return f"Error: resource '{uri}' not found"
        return f"Content of {uri}"


# ── MCP 客户端 ────────────────────────────────────────────
class MCPClient:
    """
    MCP 客户端：管理多个 MCP 服务器。

    在 LangChain 中，使用 langchain-mcp-adapters 的
    MultiServerMCPClient 替代自定义实现。
    """

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}

    def add_server(self, server: MCPServer):
        """添加一个 MCP 服务器。"""
        self.servers[server.name] = server

    def get_all_tools(self) -> list[dict]:
        """获取所有 MCP 服务器的工具列表。"""
        tools = []
        for server in self.servers.values():
            for tool in server.list_tools():
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                })
        return tools

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用指定的工具。"""
        for server in self.servers.values():
            if tool_name in server.tools:
                return server.call_tool(tool_name, arguments)
        return {"error": f"Tool '{tool_name}' not found in any server"}

    def list_servers(self) -> list[dict]:
        """列出所有已连接的服务器。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "tools_count": len(s.tools),
                "resources_count": len(s.resources),
            }
            for s in self.servers.values()
        ]


# ── 示例 MCP 服务器 ──────────────────────────────────────
def create_codegraph_server() -> MCPServer:
    """创建一个模拟的 codegraph MCP 服务器。"""
    server = MCPServer(
        name="codegraph",
        description="代码知识图谱 - 提供代码符号搜索和依赖分析",
    )

    server.register_tool(MCPTool(
        name="codegraph_explore",
        description="Explore code symbols, call paths, and dependencies",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol names, file names, or question",
                },
                "projectPath": {
                    "type": "string",
                    "description": "Path to the project",
                },
            },
            "required": ["query", "projectPath"],
        },
    ))

    server.register_resource(MCPResource(
        uri="codegraph://schema",
        name="Database Schema",
        description="The codegraph database schema",
    ))

    return server


def create_search_server() -> MCPServer:
    """创建一个模拟的搜索 MCP 服务器。"""
    server = MCPServer(
        name="web-search",
        description="网页搜索工具",
    )

    server.register_tool(MCPTool(
        name="search",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        },
    ))

    return server


# ── 程序入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    print("MCP Tools 演示\n")

    client = MCPClient()
    client.add_server(create_codegraph_server())
    client.add_server(create_search_server())

    print("── 已连接的 MCP 服务器 ──")
    for server in client.list_servers():
        print(f"  [{server['name']}] {server['description']}")
        print(f"    工具数: {server['tools_count']}, 资源数: {server['resources_count']}")

    print("\n── 所有 MCP 工具 ──")
    tools = client.get_all_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")

    print("\n── 调用 MCP 工具 ──")
    result = client.call_tool("codegraph_explore", {
        "query": "agent_loop",
        "projectPath": "/home/user/project",
    })
    print(f"  codegraph_explore 结果: {result}")

    result = client.call_tool("search", {"query": "Claude API tool use"})
    print(f"  search 结果: {result}")

    print("\n── LangChain MCP 集成 ──")
    print("""
    # 使用 langchain-mcp-adapters
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async with MultiServerMCPClient({
        "codegraph": {
            "command": "codegraph",
            "args": ["serve"],
        },
        "firecrawl": {
            "command": "npx",
            "args": ["firecrawl-mcp"],
            "env": {"FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"},
        },
    }) as mcp_client:
        tools = mcp_client.get_tools()
        agent = create_react_agent(model, tools)
        result = agent.invoke({"messages": [...]})
    """)

    print("\n── MCP 配置示例 (settings.json) ──")
    config = {
        "mcpServers": {
            "codegraph": {"command": "codegraph", "args": ["serve"]},
            "firecrawl": {
                "command": "npx",
                "args": ["firecrawl-mcp"],
                "env": {"FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"},
            },
        }
    }
    print(json.dumps(config, indent=2))
