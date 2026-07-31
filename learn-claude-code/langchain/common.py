"""
Common - 共享模块
==================
本文件提供所有单元共用的 LangChain/LangGraph 初始化代码。

核心概念：
  1. 统一的模型初始化（ChatAnthropic）
  2. 通用的工具执行辅助函数
  3. 环境变量加载

所有单元 import 此模块，避免重复的初始化代码。
"""

import os
from dotenv import load_dotenv

# 加载 .env 配置（override=True 确保 .env 值覆盖系统环境变量）
load_dotenv(override=True)

# 如果配置了自定义代理地址（如国内中转），则移除官方认证令牌
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


def get_model(model_id: str = None, max_tokens: int = 8000):
    """
    获取 LangChain ChatAnthropic 模型实例。

    Args:
        model_id: 模型 ID，默认从环境变量 MODEL_ID 读取
        max_tokens: 最大输出 token 数

    Returns:
        ChatAnthropic: 配置好的模型实例
    """
    from langchain_anthropic import ChatAnthropic

    if model_id is None:
        model_id = os.environ["MODEL_ID"]

    kwargs = {
        "model": model_id,
        "max_tokens": max_tokens,
    }

    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return ChatAnthropic(**kwargs)
