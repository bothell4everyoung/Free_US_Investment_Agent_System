# 声明伪目标，这些目标不会创建实际文件
.PHONY: help install setup run test backtest clean

# 定义颜色代码
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m
BOLD := \033[1m

# 默认目标 - 当直接运行 'make' 时显示帮助信息
# Default target when just running 'make'
help:
	@echo "$(BOLD)=== 美股投资代理系统 US Stock Investment Agent System ===$(RESET)\n"
	@echo "$(BOLD)可用命令 Available commands:$(RESET)"
	@echo "  $(GREEN)make install$(RESET)    - 安装 Poetry 和项目依赖"
	@echo "                    Install Poetry and project dependencies"
	@echo "  $(GREEN)make setup$(RESET)      - 从模板创建环境变量文件"
	@echo "                    Setup environment variables from template"
	@echo "  $(GREEN)make run$(RESET)        - 运行主程序(默认分析TSLA)"
	@echo "                    Run main program with TSLA as default"
	@echo "  $(GREEN)make backtest$(RESET)   - 运行回测程序"
	@echo "                    Run backtesting with default parameters"
	@echo "  $(GREEN)make clean$(RESET)      - 清理缓存和日志文件"
	@echo "                    Clean up cache and log files"
	@echo "\n$(BOLD)使用示例 Usage examples:$(RESET)"
	@echo "  $(BLUE)make run TICKER=AAPL$(RESET)                                        # 分析苹果股票"
	@echo "  $(BLUE)make run TICKER=TSLA SHOW_REASONING=1$(RESET)                      # 分析特斯拉并显示决策过程"
	@echo "  $(BLUE)make backtest TICKER=TSLA START=2024-12-10 END=2024-12-17$(RESET)  # 特斯拉股票回测"

# 安装 Poetry 和项目依赖
# Install Poetry and project dependencies
install:
	@echo "$(BOLD)Installing Poetry and project dependencies...$(RESET)"
	@curl -sSL https://install.python-poetry.org | python3 -
	@echo "$(GREEN)Poetry installed successfully!$(RESET)"
	@poetry install
	@echo "$(GREEN)Dependencies installed successfully!$(RESET)"

# 从模板创建环境配置文件
# Setup environment from template
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)已从模板创建 .env 文件，请编辑并填入您的 API 密钥"; \
		echo "Created .env file from template. Please edit it with your API keys.$(RESET)"; \
	else \
		echo "$(YELLOW).env 文件已存在 (.env file already exists)$(RESET)"; \
	fi

# 运行主程序
# Usage: make run TICKER=TSLA SHOW_REASONING=1
run:
	$(eval TICKER ?= TSLA)
	$(eval SHOW_REASONING ?= 0)
	$(eval NUM_NEWS ?= 5)
	$(eval END_DATE ?= $(shell date +%Y-%m-%d))
	@echo "$(BOLD)Running analysis for $(BLUE)$(TICKER)$(RESET)..."
	@if [ "$(SHOW_REASONING)" = "1" ]; then \
		poetry run python src/main.py --ticker $(TICKER) --show-reasoning --end-date $(END_DATE) --num-of-news $(NUM_NEWS); \
	else \
		poetry run python src/main.py --ticker $(TICKER) --end-date $(END_DATE) --num-of-news $(NUM_NEWS); \
	fi

# 运行回测分析
# Usage: make backtest TICKER=TSLA START=2024-12-10 END=2024-12-17
backtest:
	$(eval TICKER ?= TSLA)
	$(eval START ?= 2024-12-10)
	$(eval END ?= 2024-12-17)
	$(eval NUM_NEWS ?= 5)
	@echo "$(BOLD)Running backtest for $(BLUE)$(TICKER)$(RESET) from $(YELLOW)$(START)$(RESET) to $(YELLOW)$(END)$(RESET)..."
	@poetry run python src/backtester.py --ticker $(TICKER) \
		--start-date $(START) \
		--end-date $(END) \
		--num-of-news $(NUM_NEWS)

# 清理项目缓存和日志文件
# Clean up cache and logs
clean:
	@echo "$(BOLD)Cleaning project files...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true    # 删除 Python 字节码缓存
	@find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true  # 删除 pytest 缓存
	@find . -type f -name "*.pyc" -delete                   # 删除编译的 Python 文件
	@find . -type f -name "*.pyo" -delete                   # 删除优化的 Python 文件
	@find . -type f -name "*.pyd" -delete                   # 删除 Python DLL 文件
	@find . -type f -name ".coverage" -delete               # 删除测试覆盖率文件
	@find . -type f -name "*.log" -delete                   # 删除日志文件
	@rm -rf logs/* 2>/dev/null || true                      # 清空日志目录
	@echo "$(GREEN)Cleanup completed!$(RESET)" 