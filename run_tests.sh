#!/bin/bash
# 运行测试脚本（macOS/Linux）
echo "========================================="
echo "  LlamaIndex Hello World - 运行测试"
echo "========================================="
echo ""

# 检查 uv
if ! command -v uv &> /dev/null; then
    echo "错误：uv 未安装"
    echo "请运行：pip install uv"
    echo "或访问：https://docs.astral.sh/uv/getting-started/installation"
    exit 1
fi

echo "✓ uv 版本: $(uv --version)"
echo ""

# 同步依赖
echo "同步依赖..."
uv sync

if [ $? -ne 0 ]; then
    echo "✗ 依赖同步失败"
    exit 1
fi

echo "✓ 依赖同步完成"
echo ""

# 运行测试
echo "运行测试..."
uv run pytest -v

if [ $? -ne 0 ]; then
    echo "✗ 测试失败"
    exit 1
fi

echo ""
echo "========================================="
echo "  测试完成！"
echo "========================================="
