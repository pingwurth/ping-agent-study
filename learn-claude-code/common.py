#!/usr/bin/env python3
"""
common.py - 公共工具模块
========================
提供所有脚本共用的初始化逻辑：客户端创建、代理配置等。
"""

import os
import httpx
from anthropic import Anthropic
from dotenv import load_dotenv


def create_client() -> tuple[Anthropic, str]:
    """
    创建 Anthropic 客户端，自动处理代理配置。

    优先级：
      1. HTTPS_PROXY / HTTP_PROXY 环境变量 → 通过 httpx 代理
      2. 无代理环境变量 → 直连

    Returns:
        (client, model_id) 元组
    """
    load_dotenv(override=True)

    # 检查代理环境变量
    proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") \
                or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")

    http_client = None
    if proxy_url:
        # 通过 mitmproxy 等 HTTPS 代理时，代理用自签 CA 做中间人解密，
        # 需要关闭证书验证，否则会报 SSL:CERTIFICATE_VERIFY_FAILED
        http_client = httpx.Client(proxy=proxy_url, verify=False)

    client = Anthropic(
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        http_client=http_client,
    )
    model = os.environ["MODEL_ID"]
    return client, model
