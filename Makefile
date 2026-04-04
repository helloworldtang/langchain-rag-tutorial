# Makefile
# LlamaIndex Hello World 项目构建和测试工具

.PHONY: help install dev test lint format clean check-coverage

# 默认目标
.DEFAULT_GOAL := help

# 项目信息
PROJECT_NAME := llamaindex-hello-world
PYTHON := python3
UV := uv

# 项目目录
BACKEND_DIR := ./backend
FRONTEND_DIR := ./frontend
TESTS_DIR := ./tests
DATA_DIR := ./data

# 颜色和格式化
BLACK := $(UV) run black
RUFF := $(UV) run ruff
MYPY := $(UV) run mypy

# 测试
PYTEST := $(UV) run pytest
PYTEST_COV := $(UV) run pytest --cov=app --cov-report=html --cov-report=term

# 依赖
DEV_DEPENDENCIES := pytest pytest-asyncio pytest-mock black ruff mypy

# 帮助信息
help:
	@echo "========================================"
	@echo "  $(PROJECT_NAME) - Makefile"
	@echo "========================================"
	@echo ""
	@echo "使用方法:"
	@echo "  make install      - 安装项目依赖"
	@echo "  make dev         - 安装开发依赖（测试、格式化等）"
	@echo "  make test        - 运行所有测试"
	@echo "  make test-cov    - 运行测试并生成覆盖率报告"
	@echo "  make test-unit   - 运行单元测试"
	@echo "  make test-int    - 运行集成测试"
	@echo "  make lint        - 检查代码风格和质量"
	@echo "  make format       - 格式化代码"
	@echo "  make clean       - 清理缓存和生成的文件"
	@echo "  make check-cov   - 检查测试覆盖率"
	@echo ""
	@echo "测试目标:"
	@echo "  make test-load       - 测试文档加载"
	@echo "  make test-index       - 测试索引创建"
	@echo "  make test-query       - 测试文档查询"
	@echo ""

# 安装依赖
install:
	@echo "========================================"
	@echo "  安装项目依赖"
	@echo "========================================"
	@echo ""
	$(UV) sync
	@echo ""
	@echo "✓ 依赖安装完成"
	@echo ""

# 安装开发依赖
dev:
	@echo "========================================"
	@echo "  安装开发依赖"
	@echo "========================================"
	@echo ""
	$(UV) add --dev $(DEV_DEPENDENCIES)
	@echo ""
	@echo "✓ 开发依赖安装完成"
	@echo ""

# 运行所有测试
test:
	@echo "========================================"
	@echo "  运行所有测试"
	@echo "========================================"
	@echo ""
	$(PYTEST) -v
	@echo ""

# 运行测试并生成覆盖率报告
test-cov:
	@echo "========================================"
	@echo "  运行测试并生成覆盖率报告"
	@echo "========================================"
	@echo ""
	$(PYTEST_COV)
	@echo ""
	@echo "✓ 测试完成，覆盖率报告已生成"
	@echo "  - HTML报告: htmlcov/index.html"
	@echo "  - 终端报告: 见上"
	@echo ""

# 运行单元测试
test-unit:
	@echo "========================================"
	@echo "  运行单元测试"
	@echo "========================================"
	@echo ""
	$(PYTEST) tests/test_app.py -v -m "unit"
	@echo ""

# 运行集成测试
test-int:
	@echo "========================================"
	@echo "  运行集成测试"
	@echo "========================================"
	@echo ""
	$(PYTEST) tests/test_app_integration.py -v -m "integration"
	@echo ""

# 代码检查
lint:
	@echo "========================================"
	@echo "  代码检查"
	@echo "========================================"
	@echo ""
	@echo "使用 Ruff 检查代码..."
	$(RUFF) check . --fix
	@echo ""
	@echo "使用 Mypy 检查类型..."
	$(MYPY) app.py --ignore-missing-imports
	@echo ""
	@echo "✓ 代码检查完成"
	@echo ""

# 代码格式化
format:
	@echo "========================================"
	@echo "  代码格式化"
	@echo "========================================"
	@echo ""
	@echo "使用 Black 格式化代码..."
	$(BLACK) app.py tests/
	@echo ""
	@echo "使用 Ruff 排序导入..."
	$(RUFF) check app.py --select I --fix
	@echo ""
	@echo "✓ 代码格式化完成"
	@echo ""

# 清理缓存和生成的文件
clean:
	@echo "========================================"
	@echo "  清理缓存和生成的文件"
	@echo "========================================"
	@echo ""
	@echo "清理 Python 缓存..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "清理 .pyc 文件..."
	find . -type f -name "*.pyc" -exec rm -f {} +
	@echo "清理 .pytest_cache..."
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "清理 .coverage 和 htmlcov..."
	rm -rf .coverage htmlcov
	@echo "清理测试报告..."
	rm -rf reports/
	@echo ""
	@echo "✓ 清理完成"
	@echo ""

# 检查测试覆盖率
check-coverage:
	@echo "========================================"
	@echo "  检查测试覆盖率"
	@echo "========================================"
	@echo ""
	$(PYTEST) --cov=app --cov-fail-under=80 --cov-report=term tests/
	@echo ""
	@echo "✓ 测试覆盖率检查完成"
	@echo ""

# 快速测试
test-quick:
	@echo "========================================"
	@echo "  快速测试"
	@echo "========================================"
	@echo ""
	$(PYTEST) -v --tb=short -x tests/
	@echo ""

# 测试文档加载
test-load:
	@echo "========================================"
	@echo "  测试文档加载"
	@echo "========================================"
	@echo ""
	$(PYTEST) tests/test_app.py::TestLoadDocuments -v -x
	@echo ""

# 测试索引创建
test-index:
	@echo "========================================"
	@echo "  测试索引创建"
	@echo "========================================"
	@echo ""
	$(PYTEST) tests/test_app.py::TestCreateIndex -v -x
	@echo ""

# 测试文档查询
test-query:
	@echo "========================================"
	@echo "  测试文档查询"
	@echo "========================================"
	@echo ""
	$(PYTEST) tests/test_app.py::TestQueryDocuments -v -x
	@echo ""

# 测试集成
test-integration-all:
	@echo "========================================"
	@echo "  测试集成"
	@echo "========================================"
	@echo ""
	$(PYTEST) tests/test_app_integration.py::TestIntegration -v -x
	@echo ""
