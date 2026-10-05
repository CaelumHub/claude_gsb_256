#!/usr/bin/env bash
# =============================================================================
# 一键启动脚本：前端（静态页面）+ 后端（Flask API）同源部署，均由本脚本启动。
#
# 用法：
#   ./run.sh              # 默认端口 8000，前台运行，Ctrl+C 退出
#   ./run.sh --port 9000  # 指定端口
#   ./run.sh --no-browser # 不自动打开浏览器
#   ./run.sh --install    # 强制重新安装依赖后启动
#   ./run.sh --help       # 查看帮助
# =============================================================================
set -euo pipefail

# 切到脚本所在目录（保证无论从何处调用都能找到项目文件）
cd "$(dirname "$0")"

PORT=8000
OPEN_BROWSER=1
DO_INSTALL=""
PYTHON=python3
VENV_DIR=".venv"

# -----------------------------------------------------------------------------
# 参数解析
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port|-p)
      PORT="${2:-8000}"; shift 2 ;;
    --no-browser)
      OPEN_BROWSER=0; shift ;;
    --install|-i)
      DO_INSTALL=1; shift ;;
    --help|-h)
      cat <<'EOF'
一键启动 NLP 流水线平台（前端页面 + 后端 API 同源部署）

用法:
  ./run.sh                # 默认端口 8000，前台运行，Ctrl+C 退出
  ./run.sh --port 9000    # 指定端口
  ./run.sh --no-browser   # 不自动打开浏览器
  ./run.sh --install      # 强制重新安装依赖后启动
  ./run.sh --help         # 查看帮助
EOF
      exit 0 ;;
    *)
      echo "未知参数: $1（用 --help 查看帮助）" >&2; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# 依赖：优先复用已有虚拟环境，否则创建并安装
# -----------------------------------------------------------------------------
if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON="$VENV_DIR/bin/python"
  echo "[run] 使用已存在的虚拟环境 $VENV_DIR"
elif [[ -n "$DO_INSTALL" ]] || ! "$PYTHON" -c "import flask" >/dev/null 2>&1; then
  echo "[run] 创建虚拟环境并安装依赖..."
  "$PYTHON" -m venv "$VENV_DIR"
  PYTHON="$VENV_DIR/bin/python"
  "$PYTHON" -m pip install --upgrade pip >/dev/null
  "$PYTHON" -m pip install -r requirements.txt
fi

# -----------------------------------------------------------------------------
# 端口占用检查
# -----------------------------------------------------------------------------
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  echo "[run] 端口 $PORT 已被占用，尝试其它端口（如 ./run.sh --port 9000）" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# 启动
# -----------------------------------------------------------------------------
echo "[run] 启动 NLP 流水线平台..."
echo "[run]   后端 API + 前端页面: http://127.0.0.1:$PORT"
echo "[run]   停止: Ctrl+C"

if [[ "$OPEN_BROWSER" == "1" ]]; then
  # 延迟 1.5s 打开浏览器，确保服务已就绪
  ( sleep 1.5
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "http://127.0.0.1:$PORT"
    elif command -v open >/dev/null 2>&1; then open "http://127.0.0.1:$PORT"
    fi ) >/dev/null 2>&1 &
fi

exec "$PYTHON" app.py --port "$PORT"
