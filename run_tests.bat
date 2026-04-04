@echo off
REM 运行测试脚本（Windows）
echo ======================================
echo   LlamaIndex Hello World - 运行测试
echo ======================================
echo.

REM 检查 uv
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] uv 未安装
    echo.
    echo 请安装 uv：
    echo   访问 https://docs.astral.sh/uv/getting-started/installation
    echo.
    echo 或使用项目中的 uv：
    echo   uv/uv/uv.exe
    pause
    exit /b 1
)

echo [√] uv 已安装
echo.
echo.

REM 同步依赖
echo [INFO] 同步依赖...
uv sync

if %errorlevel% neq 0 (
    echo [X] 依赖同步失败
    pause
    exit /b 1
)

echo [√] 依赖同步完成
echo.
echo.

REM 运行测试
echo ======================================
echo   运行测试
echo ======================================
echo.

uv run pytest -v

if %errorlevel% neq 0 (
    echo [X] 测试失败
    pause
    exit /b 1
)

echo.
echo ======================================
echo   测试完成！
echo ======================================
echo.

REM 等待用户输入
pause
