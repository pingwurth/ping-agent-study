#!/usr/bin/env bash
# run_with_mitm.sh - 通过 mitmweb 代理运行 Python 脚本
# 用法: ./run_with_mitm.sh <python_script> [args...]
# 示例: ./run_with_mitm.sh section1_tool_execution/u01_agent_loop.py

set -euo pipefail

MITMWEB="/home/ping/devops/mitmproxy/mitmweb"
MITM_PORT="${MITM_PORT:-5417}"
MITM_WEB_PORT="${MITM_WEB_PORT:-8081}"

if [ $# -eq 0 ]; then
    echo "用法: $0 <python_script> [args...]"
    echo "示例: $0 section1_tool_execution/u01_agent_loop.py"
    echo ""
    echo "环境变量:"
    echo "  MITM_PORT     代理端口 (默认 5417)"
    echo "  MITM_WEB_PORT  Web UI 端口 (默认 8081)"
    exit 1
fi

SCRIPT="$1"
shift

# 检查 mitmweb 是否已在运行
if ! lsof -i ":${MITM_PORT}" -sTCP:LISTEN &>/dev/null; then
    echo ">>> 启动 mitmweb (代理端口: ${MITM_PORT}, Web UI: http://127.0.0.1:${MITM_WEB_PORT})"
    # script 伪造 TTY，强制 mitmweb (PyInstaller 二进制) 实时写入日志
    # stdbuf 对 PyInstaller 打包的二进制无效，因为 Python logging 有独立缓冲层
    script -qfc "$MITMWEB --set listen_port=${MITM_PORT} --set web_port=${MITM_WEB_PORT}" \
        /tmp/mitmweb.log >/dev/null 2>&1 &
    MITM_PID=$!
    trap 'kill ${MITM_PID} 2>/dev/null; wait ${MITM_PID} 2>/dev/null' EXIT
    # 等待代理就绪
    for i in $(seq 1 10); do
        if lsof -i ":${MITM_PORT}" -sTCP:LISTEN &>/dev/null; then
            break
        fi
        sleep 0.5
    done
    echo ">>> mitmweb PID: ${MITM_PID}"
else
    echo ">>> mitmweb 已在端口 ${MITM_PORT} 运行"
fi

echo ">>> 运行: python ${SCRIPT} $*"
if [ -f /tmp/mitmweb.log ]; then
  MITM_WEB_URL=$(sed -n -E "/Web server listening.*\?token=/p" /tmp/mitmweb.log | tail -n1)
  echo ">>> ${MITM_WEB_URL}"
fi
echo ""

HTTPS_PROXY="http://127.0.0.1:${MITM_PORT}" \
HTTP_PROXY="http://127.0.0.1:${MITM_PORT}" \
python "$SCRIPT" "$@"
