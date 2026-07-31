"""
U19 - MCP Tools（MCP 工具）
============================
本文件演示 Claude Code 的 **MCP (Model Context Protocol)** 工具集成。

核心概念：
  1. MCP 是 Anthropic 定义的开放协议，用于连接 LLM 和外部工具
  2. MCP 服务器提供工具（tools）和资源（resources）
  3. Claude Code 可以连接多个 MCP 服务器
  4. MCP 工具像内置工具一样被模型调用

Claude Code 原始实现：
  ┌──────────────────────────────────────────────────────────┐
  │  Claude Code 通过 settings.json 配置 MCP 服务器：         │
  │                                                          │
  │  {                                                       │
  │    "mcpServers": {                                       │
  │      "codegraph": {                                      │
  │        "command": "codegraph",                           │
  │        "args": ["serve"]                                 │
  │      },                                                  │
  │      "firecrawl": {                                      │
  │        "command": "npx",                                 │
  │        "args": ["firecrawl-mcp"],                        │
  │        "env": {"FIRECRAWL_API_KEY": "..."}               │
  │      }                                                   │
  │    }                                                     │
  │  }                                                       │
  └──────────────────────────────────────────────────────────┘

MCP 协议架构：
  ┌─────────────┐     JSON-RPC      ┌─────────────┐
  │  Claude     │ ◄──────────────► │  MCP        │
  │  Code       │                   │  Server     │
  │  (Client)   │                   │             │
  └─────────────┘                   └─────────────┘
       │                                 │
       │ tools/list                      │ 注册工具
       │ tools/call                      │ 注册资源
       │ resources/list                  │
       │ resources/read                  │
       ▼                                 ▼

MCP 工具定义格式（JSON Schema）：
  {
      "name": "tool_name",
      "description": "What this tool does",
      "input_schema": {
          "type": "object",
          "properties": {
              "param1": {"type": "string", "description": "..."}
          },
          "required": ["param1"]
      }
  }

本文件是纯 Python 实现，不依赖 anthropic SDK。
使用 Python 类模拟 MCP 服务器和客户端。
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════
# 第一部分：MCP 工具和资源定义
# ══════════════════════════════════════════════════════════════

@dataclass
class MCPTool:
    """
    MCP 工具的定义。

    对应 MCP 协议中的 Tool 类型：
    {
        "name": "tool_name",
        "description": "What this tool does",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

    字段说明：
      - name:         工具名称（必须唯一）
      - description:  工具描述（模型用它来决定何时调用）
      - input_schema: 输入参数的 JSON Schema
      - server_name:  所属的 MCP 服务器名称
    """
    name: str
    description: str
    input_schema: dict
    server_name: str = ""


@dataclass
class MCPResource:
    """
    MCP 资源的定义。

    资源是 MCP 服务器提供的可读取内容：
      - 文件内容
      - 数据库记录
      - API 响应
      - 配置信息

    字段说明：
      - uri:         资源的唯一标识符（类似 URL）
      - name:        资源名称
      - description: 资源描述
      - mime_type:   MIME 类型（默认 text/plain）
    """
    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


# ══════════════════════════════════════════════════════════════
# 第二部分：MCP 服务器
# ══════════════════════════════════════════════════════════════

class MCPServer:
    """
    MCP 服务器的模拟实现。

    真实的 MCP 服务器：
      - 是一个独立进程
      - 通过 JSON-RPC 协议通信（stdin/stdout）
      - 实现 tools/list, tools/call, resources/list, resources/read 等方法

    这里简化为 Python 类来演示概念。

    Claude Code 支持的 MCP 服务器示例：
      - codegraph: 代码知识图谱（符号搜索、依赖分析）
      - firecrawl: 网页爬取和内容提取
      - context7:  库文档查询
    """

    def __init__(self, name: str, description: str = ""):
        """
        初始化 MCP 服务器。

        Args:
            name:        服务器名称（唯一标识）
            description: 服务器描述
        """
        self.name = name
        self.description = description
        # 已注册的工具
        self.tools: dict[str, MCPTool] = {}
        # 已注册的资源
        self.resources: dict[str, MCPResource] = {}

    def register_tool(self, tool: MCPTool):
        """
        注册一个工具。

        MCP 服务器启动时注册所有可用工具。
        客户端通过 tools/list 获取工具列表。

        Args:
            tool: 工具定义
        """
        tool.server_name = self.name
        self.tools[tool.name] = tool

    def register_resource(self, resource: MCPResource):
        """
        注册一个资源。

        Args:
            resource: 资源定义
        """
        self.resources[resource.uri] = resource

    def list_tools(self) -> list[MCPTool]:
        """
        列出所有工具。

        对应 MCP 协议的 tools/list 方法。

        Returns:
            list[MCPTool]: 工具列表
        """
        return list(self.tools.values())

    def list_resources(self) -> list[MCPResource]:
        """
        列出所有资源。

        对应 MCP 协议的 resources/list 方法。

        Returns:
            list[MCPResource]: 资源列表
        """
        return list(self.resources.values())

    def call_tool(self, name: str, arguments: dict) -> Any:
        """
        调用工具。

        对应 MCP 协议的 tools/call 方法。

        Args:
            name:      工具名称
            arguments: 工具参数

        Returns:
            Any: 工具执行结果
        """
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        # 简化实现：返回模拟结果
        return {"result": f"Executed {name} with {arguments}"}

    def read_resource(self, uri: str) -> str:
        """
        读取资源。

        对应 MCP 协议的 resources/read 方法。

        Args:
            uri: 资源 URI

        Returns:
            str: 资源内容
        """
        if uri not in self.resources:
            return f"Error: resource '{uri}' not found"
        return f"Content of {uri}"


# ══════════════════════════════════════════════════════════════
# 第三部分：MCP 客户端
# ══════════════════════════════════════════════════════════════

class MCPClient:
    """
    MCP 客户端：管理多个 MCP 服务器。

    Claude Code 作为 MCP 客户端：
      1. 从 settings.json 读取 MCP 服务器配置
      2. 启动每个 MCP 服务器进程
      3. 通过 JSON-RPC 与服务器通信
      4. 将 MCP 工具与内置工具一起提供给模型

    提供以下功能：
      1. add_server()    - 添加 MCP 服务器
      2. get_all_tools() - 获取所有服务器的工具列表
      3. call_tool()     - 调用指定工具（自动路由到正确的服务器）
      4. list_servers()  - 列出所有已连接的服务器
    """

    def __init__(self):
        # 已连接的服务器：server_name → MCPServer
        self.servers: dict[str, MCPServer] = {}

    def add_server(self, server: MCPServer):
        """
        添加一个 MCP 服务器。

        Args:
            server: MCP 服务器实例
        """
        self.servers[server.name] = server

    def get_all_tools(self) -> list[dict]:
        """
        获取所有 MCP 服务器的工具列表。

        合并所有服务器的工具，返回统一的工具列表。
        模型可以从这个列表中选择要调用的工具。

        Returns:
            list[dict]: 工具列表（JSON Schema 格式）
        """
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
        """
        调用指定的工具。

        自动在所有服务器中查找工具并调用。
        如果多个服务器有同名工具，调用第一个找到的。

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            Any: 工具执行结果
        """
        for server in self.servers.values():
            if tool_name in server.tools:
                return server.call_tool(tool_name, arguments)
        return {"error": f"Tool '{tool_name}' not found in any server"}

    def list_servers(self) -> list[dict]:
        """
        列出所有已连接的服务器。

        Returns:
            list[dict]: 服务器信息列表
        """
        return [
            {
                "name": s.name,
                "description": s.description,
                "tools_count": len(s.tools),
                "resources_count": len(s.resources),
            }
            for s in self.servers.values()
        ]


# ══════════════════════════════════════════════════════════════
# 第四部分：示例 MCP 服务器
# ══════════════════════════════════════════════════════════════

def create_codegraph_server() -> MCPServer:
    """
    创建一个模拟的 codegraph MCP 服务器。

    CodeGraph 是 Claude Code 支持的 MCP 服务器：
      - 提供代码符号搜索
      - 提供依赖分析
      - 提供调用路径追踪

    Claude Code 中的配置：
    {
        "mcpServers": {
            "codegraph": {
                "command": "codegraph",
                "args": ["serve"]
            }
        }
    }
    """
    server = MCPServer(
        name="codegraph",
        description="代码知识图谱 - 提供代码符号搜索和依赖分析",
    )

    # 注册 codegraph_explore 工具
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

    # 注册数据库 schema 资源
    server.register_resource(MCPResource(
        uri="codegraph://schema",
        name="Database Schema",
        description="The codegraph database schema",
    ))

    return server


def create_search_server() -> MCPServer:
    """
    创建一个模拟的搜索 MCP 服务器。

    提供网页搜索功能，类似 Claude Code 中的 WebSearch 工具。
    """
    server = MCPServer(
        name="web-search",
        description="网页搜索工具",
    )

    # 注册 search 工具
    server.register_tool(MCPTool(
        name="search",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ))

    return server


# ══════════════════════════════════════════════════════════════
# 第五部分：程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("U19 - MCP Tools MCP 工具演示")
    print("=" * 60)

    # 创建 MCP 客户端
    client = MCPClient()

    # 添加 MCP 服务器
    client.add_server(create_codegraph_server())
    client.add_server(create_search_server())

    # ── 已连接的服务器 ────────────────────────────────────
    print("\n── 已连接的 MCP 服务器 ──")
    for server in client.list_servers():
        print(f"  [{server['name']}] {server['description']}")
        print(f"    工具数: {server['tools_count']}, 资源数: {server['resources_count']}")

    # ── 所有 MCP 工具 ─────────────────────────────────────
    print("\n── 所有 MCP 工具 ──")
    tools = client.get_all_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")

    # ── 调用 MCP 工具 ─────────────────────────────────────
    print("\n── 调用 MCP 工具 ──")

    result = client.call_tool("codegraph_explore", {
        "query": "agent_loop",
        "projectPath": "/home/user/project",
    })
    print(f"  codegraph_explore 结果: {result}")

    result = client.call_tool("search", {"query": "Claude API tool use"})
    print(f"  search 结果: {result}")

    # 调用不存在的工具
    result = client.call_tool("nonexistent", {})
    print(f"  nonexistent 结果: {result}")

    # ── MCP 配置示例 ──────────────────────────────────────
    print("\n── Claude Code MCP 配置示例 (settings.json) ──")
    config = {
        "mcpServers": {
            "codegraph": {
                "command": "codegraph",
                "args": ["serve"],
            },
            "firecrawl": {
                "command": "npx",
                "args": ["firecrawl-mcp"],
                "env": {"FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"},
            },
        }
    }
    print(json.dumps(config, indent=2))

    # ── Claude Code MCP 机制说明 ──────────────────────────
    print("\n── Claude Code MCP 机制说明 ──")
    print("""
    MCP (Model Context Protocol) 是 Anthropic 定义的开放协议：

    1. MCP 服务器配置：
       在 ~/.claude/settings.json 中配置：
       {
         "mcpServers": {
           "server-name": {
             "command": "executable",
             "args": ["arg1", "arg2"],
             "env": {"KEY": "value"}
           }
         }
       }

    2. MCP 工具调用流程：
       - Claude Code 启动时连接所有 MCP 服务器
       - 获取每个服务器的工具列表（tools/list）
       - 将 MCP 工具与内置工具一起提供给模型
       - 模型决定调用哪个工具
       - Claude Code 将调用路由到正确的 MCP 服务器

    3. MCP 资源：
       - 资源是可读取的内容（文件、数据库、API）
       - 通过 resources/list 获取资源列表
       - 通过 resources/read 读取资源内容

    4. 常用 MCP 服务器：
       - codegraph: 代码知识图谱
       - firecrawl: 网页爬取
       - context7:  库文档查询
       - github:    GitHub API 集成
    """)
