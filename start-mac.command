#!/bin/bash
# SpriteMaster - 启动脚本 (前后端分离模式)

# 获取脚本所在目录
cd "$(dirname "$0")"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python3"
    read -p "按任意键退出..."
    exit 1
fi

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "虚拟环境不存在，正在创建..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "检查并安装依赖..."
pip install -q -r requirements.txt

echo ""
echo "========================================="
echo "  SpriteMaster - 前后端分离模式"
echo "========================================="
echo ""

# 查找可用端口
find_free_port() {
    local port=$1
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

# 启动后端 API 服务
BACKEND_PORT=8000
BACKEND_PORT=$(find_free_port $BACKEND_PORT)
echo "启动后端 API 服务 (端口 $BACKEND_PORT)..."
cd backend
python3 app.py --port $BACKEND_PORT &
BACKEND_PID=$!
cd ..

# 等待后端启动
sleep 2

# 启动前端静态服务器
FRONTEND_PORT=8080
FRONTEND_PORT=$(find_free_port $FRONTEND_PORT)
echo "启动前端服务器 (端口 $FRONTEND_PORT)..."
cd frontend
python3 -m http.server $FRONTEND_PORT &
FRONTEND_PID=$!
cd ..

# 等待前端启动
sleep 1

# 写入端口配置文件
echo "BACKEND_PORT=$BACKEND_PORT" > frontend/port_config.txt

echo ""
echo "========================================="
echo "  服务已启动!"
echo "========================================="
echo ""
echo "  后端 API: http://localhost:$BACKEND_PORT"
echo "  前端页面: http://localhost:$FRONTEND_PORT"
echo ""
echo "  请在浏览器中打开: http://localhost:$FRONTEND_PORT"
echo ""
echo "========================================="
echo "  按 Ctrl+C 停止所有服务"
echo "========================================="
echo ""

# 打开浏览器
if command -v open &> /dev/null; then
    open "http://localhost:$FRONTEND_PORT"
fi

# 捕获 Ctrl+C 信号，停止所有服务
cleanup() {
    echo ""
    echo "正在停止服务..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    rm -f frontend/port_config.txt
    echo "服务已停止"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 等待任一进程结束
wait $BACKEND_PID $FRONTEND_PID
