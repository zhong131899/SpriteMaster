@echo off
REM SpriteMaster - 启动脚本 (前后端分离模式) - Windows

cd /d "%~dp0"

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python
    pause
    exit /b 1
)

REM 检查虚拟环境是否存在
if not exist "venv\" (
    echo 虚拟环境不存在，正在创建...
    python -m venv venv
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 安装依赖
echo 检查并安装依赖...
pip install -q -r requirements.txt

echo.
echo =========================================
echo   SpriteMaster - 前后端分离模式
echo =========================================
echo.

REM 查找可用端口并启动后端
set BACKEND_PORT=8000
:check_backend_port
netstat -an | findstr ":%BACKEND_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    set /a BACKEND_PORT+=1
    goto check_backend_port
)

echo 启动后端 API 服务 (端口 %BACKEND_PORT%)...
cd backend
start /B python app.py --port %BACKEND_PORT%
cd ..

REM 等待后端启动
timeout /t 2 /nobreak >nul

REM 查找可用端口并启动前端
set FRONTEND_PORT=8080
:check_frontend_port
netstat -an | findstr ":%FRONTEND_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    set /a FRONTEND_PORT+=1
    goto check_frontend_port
)

echo 启动前端服务器 (端口 %FRONTEND_PORT%)...
cd frontend
start /B python -m http.server %FRONTEND_PORT%
cd ..

REM 等待前端启动
timeout /t 1 /nobreak >nul

REM 写入端口配置文件
echo BACKEND_PORT=%BACKEND_PORT% > frontend\port_config.txt

echo.
echo =========================================
echo   服务已启动!
echo =========================================
echo.
echo   后端 API: http://localhost:%BACKEND_PORT%
echo   前端页面: http://localhost:%FRONTEND_PORT%
echo.
echo   请在浏览器中打开: http://localhost:%FRONTEND_PORT%
echo.
echo =========================================
echo   关闭此窗口将停止所有服务
echo =========================================
echo.

REM 打开浏览器
start "" "http://localhost:%FRONTEND_PORT%"

REM 等待用户按键
pause

REM 停止服务并清理
taskkill /F /IM python.exe >nul 2>&1
del frontend\port_config.txt 2>nul
echo 服务已停止
