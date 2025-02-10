from datetime import datetime, timedelta
import json
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import matplotlib
import pandas_market_calendars as mcal
import warnings

from main import run_hedge_fund
from tools.api import get_price_data

# 根据操作系统配置中文字体
if sys.platform.startswith('win'):
    matplotlib.rc('font', family='Microsoft YaHei')
elif sys.platform.startswith('linux'):
    matplotlib.rc('font', family='WenQuanYi Micro Hei')
else:
    matplotlib.rc('font', family='PingFang SC')

# 启用负号显示
matplotlib.rcParams['axes.unicode_minus'] = True

# Disable matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning,
                        module='pandas.plotting')
# 禁用所有与plotting相关的警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)


class Backtester:
    def __init__(self, agent, ticker, start_date, end_date, initial_capital, num_of_news=5):
        self.agent = agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.portfolio = {"cash": initial_capital, "stock": 0}
        self.portfolio_values = []
        self.num_of_news = num_of_news

        # 设置日志系统
        self.setup_backtest_logging()
        self.logger = self.setup_logging()

        # 初始化API调用管理
        self._api_call_count = 0
        self._api_window_start = time.time()
        self._last_api_call = 0

        # 初始化市场日历
        self.nyse = mcal.get_calendar('NYSE')

        # 验证输入参数
        self.validate_inputs()

    def setup_logging(self):
        """设置日志系统"""
        logger = logging.getLogger('backtester')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def validate_inputs(self):
        """验证输入参数"""
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if start >= end:
                raise ValueError("Start date must be earlier than end date")
            if self.initial_capital <= 0:
                raise ValueError("Initial capital must be greater than 0")
            if not isinstance(self.ticker, str) or len(self.ticker) == 0:
                raise ValueError("Invalid stock code format")
            # 支持美股代码（如AAPL）和A股代码（如600519）
            if not (self.ticker.isalpha() or (len(self.ticker) == 6 and self.ticker.isdigit())):
                self.backtest_logger.warning(
                    f"Stock code {self.ticker} might be in an unusual format")
            self.backtest_logger.info("输入参数验证通过")
        except Exception as e:
            self.backtest_logger.error(
                f"输入参数验证失败: {str(e)}")
            raise

    def setup_backtest_logging(self):
        """设置回测日志"""
        log_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        self.backtest_logger = logging.getLogger('backtest')
        self.backtest_logger.setLevel(logging.INFO)

        if self.backtest_logger.handlers:
            self.backtest_logger.handlers.clear()

        current_date = datetime.now().strftime('%Y%m%d')
        backtest_period = f"{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}"
        log_file = os.path.join(
            log_dir, f"backtest_{self.ticker}_{current_date}_{backtest_period}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.backtest_logger.addHandler(file_handler)

        self.backtest_logger.info(
            f"回测开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.backtest_logger.info(f"股票代码: {self.ticker}")
        self.backtest_logger.info(
            f"回测周期: {self.start_date} 到 {self.end_date}")
        self.backtest_logger.info(
            f"初始资本: {self.initial_capital:,.2f}\n")
        self.backtest_logger.info("-" * 100)

    def is_market_open(self, date_str):
        """检查给定日期市场是否开放"""
        schedule = self.nyse.schedule(start_date=date_str, end_date=date_str)
        return not schedule.empty

    def get_previous_trading_day(self, date_str):
        """获取给定日期的前一个交易日"""
        date = pd.Timestamp(date_str)
        schedule = self.nyse.schedule(
            start_date=date - pd.Timedelta(days=10),
            end_date=date
        )
        if schedule.empty:
            return None
        return schedule.index[-2].strftime('%Y-%m-%d')

    def get_agent_decision(self, current_date, lookback_start, portfolio, num_of_news):
        """获取代理决策并进行API速率限制"""
        max_retries = 3
        current_time = time.time()

        if current_time - self._api_window_start >= 60:
            self._api_call_count = 0
            self._api_window_start = current_time

        if self._api_call_count >= 8:
            wait_time = 60 - (current_time - self._api_window_start)
            if wait_time > 0:
                self.backtest_logger.info(
                    f"API limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self._api_call_count = 0
                self._api_window_start = time.time()

        for attempt in range(max_retries):
            try:
                if self._last_api_call:
                    time_since_last_call = time.time() - self._last_api_call
                    if time_since_last_call < 6:
                        time.sleep(6 - time_since_last_call)

                self._last_api_call = time.time()
                self._api_call_count += 1

                result = self.agent(
                    ticker=self.ticker,
                    start_date=lookback_start,
                    end_date=current_date,
                    portfolio=portfolio,
                    num_of_news=num_of_news
                )

                try:
                    if isinstance(result, str):
                        result = result.replace(
                            '```json\n', '').replace('\n```', '').strip()
                        parsed_result = json.loads(result)

                        formatted_result = {
                            "decision": parsed_result,
                            "analyst_signals": {}
                        }

                        if "agent_signals" in parsed_result:
                            formatted_result["analyst_signals"] = {
                                signal["agent"]: {
                                    "signal": signal.get("signal", "unknown"),
                                    "confidence": signal.get("confidence", 0)
                                }
                                for signal in parsed_result["agent_signals"]
                            }

                        return formatted_result
                    return result
                except json.JSONDecodeError as e:
                    self.backtest_logger.warning(
                        f"JSON parsing error: {str(e)}")
                    self.backtest_logger.warning(f"Raw result: {result}")
                    return {"decision": {"action": "hold", "quantity": 0}, "analyst_signals": {}}

            except Exception as e:
                if "AFC is enabled" in str(e):
                    self.backtest_logger.warning(
                        f"AFC limit triggered, waiting 60 seconds...")
                    time.sleep(60)
                    self._api_call_count = 0
                    self._api_window_start = time.time()
                    continue

                self.backtest_logger.warning(
                    f"Failed to get agent decision (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {"decision": {"action": "hold", "quantity": 0}, "analyst_signals": {}}
                time.sleep(2 ** attempt)

    def execute_trade(self, action, quantity, current_price):
        """根据投资组合约束执行交易"""
        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                self.portfolio["stock"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    self.portfolio["stock"] += max_quantity
                    self.portfolio["cash"] -= max_quantity * current_price
                    return max_quantity
                return 0
        elif action == "sell" and quantity > 0:
            quantity = min(quantity, self.portfolio["stock"])
            if quantity > 0:
                self.portfolio["cash"] += quantity * current_price
                self.portfolio["stock"] -= quantity
                return quantity
            return 0
        return 0

    def run_backtest(self):
        """运行回测模拟"""
        # 从市场日历中获取有效交易日期
        schedule = self.nyse.schedule(
            start_date=self.start_date, end_date=self.end_date)
        dates = pd.DatetimeIndex([dt.strftime('%Y-%m-%d')
                                 for dt in schedule.index])

        self.backtest_logger.info("\n开始回测...")
        print(f"{'Date':<12} {'Code':<6} {'Action':<6} {'Quantity':>8} {'Price':>8} {'Cash':>12} {'Stock':>8} {'Total':>12} {'Bull':>8} {'Bear':>8} {'Neutral':>8}")
        print("-" * 110)

        for current_date in dates:
            current_date_str = current_date.strftime("%Y-%m-%d")

            # 检查市场是否开放
            if not self.is_market_open(current_date_str):
                self.backtest_logger.info(
                    f"{current_date_str} 市场关闭（假期），跳过...")
                continue

            # 获取前一个交易日
            decision_date = self.get_previous_trading_day(current_date_str)
            if decision_date is None:
                self.backtest_logger.warning(
                    f"无法找到 {current_date_str} 的前一个交易日，跳过...")
                continue

            # 使用365天的回溯窗口
            lookback_start = (pd.Timestamp(current_date_str) -
                              pd.Timedelta(days=365)).strftime("%Y-%m-%d")

            self.backtest_logger.info(
                f"\n处理交易日: {current_date_str}")
            self.backtest_logger.info(
                f"使用数据截至: {decision_date}（前一个交易日）")
            self.backtest_logger.info(
                f"历史数据范围: {lookback_start} 到 {decision_date}")

            # 获取当前日的价格数据以执行交易
            try:
                df = get_price_data(
                    self.ticker, current_date_str, current_date_str)
                if df is None or df.empty:
                    self.backtest_logger.warning(
                        f"{current_date_str} 没有可用的价格数据，跳过...")
                    continue

                # 使用开盘价进行交易执行
                current_price = df.iloc[0]['open']
            except Exception as e:
                self.backtest_logger.error(
                    f"获取 {current_date_str} 的价格数据时出错: {str(e)}")
                continue

            # 获取基于历史数据的代理决策
            output = self.get_agent_decision(
                decision_date,
                lookback_start,
                self.portfolio,
                self.num_of_news
            )

            self.backtest_logger.info(f"\n交易日期: {current_date_str}")
            self.backtest_logger.info(
                f"基于数据的决策截至: {decision_date}")

            if "analyst_signals" in output:
                self.backtest_logger.info("\n代理分析结果:")
                for agent_name, signal in output["analyst_signals"].items():
                    self.backtest_logger.info(f"\n{agent_name}:")

                    signal_str = f"- 信号: {signal.get('signal', '未知')}"
                    if 'confidence' in signal:
                        signal_str += f", 置信度: {signal.get('confidence', 0)*100:.0f}%"
                    self.backtest_logger.info(signal_str)

                    if 'analysis' in signal:
                        self.backtest_logger.info("- 分析:")
                        analysis = signal['analysis']
                        if isinstance(analysis, dict):
                            for key, value in analysis.items():
                                self.backtest_logger.info(f"  {key}: {value}")
                        elif isinstance(analysis, list):
                            for item in analysis:
                                self.backtest_logger.info(f"  • {item}")
                        else:
                            self.backtest_logger.info(f"  {analysis}")

                    if 'reason' in signal:
                        self.backtest_logger.info("- 决策理由:")
                        reason = signal['reason']
                        if isinstance(reason, list):
                            for item in reason:
                                self.backtest_logger.info(f"  • {item}")
                        else:
                            self.backtest_logger.info(f"  • {reason}")

            agent_decision = output.get(
                "decision", {"action": "hold", "quantity": 0})
            action, quantity = agent_decision.get(
                "action", "hold"), agent_decision.get("quantity", 0)

            self.backtest_logger.info("\n最终决策:")
            self.backtest_logger.info(f"动作: {action.upper()}")
            self.backtest_logger.info(f"数量: {quantity}")
            if "reason" in agent_decision:
                self.backtest_logger.info(
                    f"理由: {agent_decision['reason']}")

            # 执行交易
            executed_quantity = self.execute_trade(
                action, quantity, current_price)

            # 更新投资组合价值
            total_value = self.portfolio["cash"] + \
                self.portfolio["stock"] * current_price
            self.portfolio["portfolio_value"] = total_value

            # 记录投资组合价值
            self.portfolio_values.append({
                "Date": current_date_str,
                "Portfolio Value": total_value,
                "Daily Return": (total_value / self.portfolio_values[-1]["Portfolio Value"] - 1) * 100 if self.portfolio_values else 0
            })

            # 计数信号
            bull_count = sum(1 for signal in output.get(
                "analyst_signals", {}).values() if signal.get("signal") == "buy")
            bear_count = sum(1 for signal in output.get(
                "analyst_signals", {}).values() if signal.get("signal") == "sell")
            neutral_count = sum(1 for signal in output.get(
                "analyst_signals", {}).values() if signal.get("signal") == "hold")

            # 打印交易记录
            print(
                f"{current_date_str:<12} {self.ticker:<6} {action:<6} {executed_quantity:>8} "
                f"{current_price:>8.2f} {self.portfolio['cash']:>12.2f} {self.portfolio['stock']:>8} "
                f"{total_value:>12.2f} {bull_count:>8} {bear_count:>8} {neutral_count:>8}"
            )

        # 分析回测结果
        self.analyze_performance()

    def analyze_performance(self):
        """分析回测性能"""
        if not self.portfolio_values:
            self.backtest_logger.warning("没有可分析的投资组合价值")
            return

        try:
            performance_df = pd.DataFrame(self.portfolio_values)
            # 将日期字符串转换为datetime类型
            performance_df['Date'] = pd.to_datetime(performance_df['Date'])
            performance_df = performance_df.set_index('Date')

            # 计算累计收益率
            performance_df["Cumulative Return"] = (
                performance_df["Portfolio Value"] / self.initial_capital - 1) * 100
            performance_df["Portfolio Value (K)"] = performance_df["Portfolio Value"] / 1000

            # 创建子图
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(12, 10), height_ratios=[1, 1])
            fig.suptitle("回测分析", fontsize=12)

            # 绘制投资组合价值
            line1 = ax1.plot(performance_df.index, performance_df["Portfolio Value (K)"],
                             label="投资组合价值", marker='o')
            ax1.set_ylabel("投资组合价值 (K)")
            ax1.set_title("投资组合价值变化")

            # 添加数据标注
            for x, y in zip(performance_df.index, performance_df["Portfolio Value (K)"]):
                ax1.annotate(f'{y:.1f}K',
                             (x, y),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center')

            # 绘制累计收益率
            line2 = ax2.plot(performance_df.index, performance_df["Cumulative Return"],
                             label="累计收益率", color='green', marker='o')
            ax2.set_ylabel("累计收益率 (%)")
            ax2.set_title("累计收益率变化")

            # 添加数据标注
            for x, y in zip(performance_df.index, performance_df["Cumulative Return"]):
                ax2.annotate(f'{y:.1f}%',
                             (x, y),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center')

            plt.xlabel("日期")
            plt.tight_layout()

            # 保存图片
            plt.savefig("backtest_results.png", bbox_inches='tight', dpi=300)
            # 显示图片
            plt.show(block=True)
            # 关闭图形
            plt.close('all')

            # 计算性能指标
            total_return = (
                self.portfolio["portfolio_value"] - self.initial_capital) / self.initial_capital

            # 输出回测总结
            self.backtest_logger.info("\n" + "=" * 50)
            self.backtest_logger.info("回测总结")
            self.backtest_logger.info("=" * 50)
            self.backtest_logger.info(
                f"初始资本: {self.initial_capital:,.2f}")
            self.backtest_logger.info(
                f"最终价值: {self.portfolio['portfolio_value']:,.2f}")
            self.backtest_logger.info(
                f"总回报: {total_return * 100:.2f}%")

            # 计算夏普比率
            daily_returns = performance_df["Daily Return"] / 100
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe_ratio = (mean_daily_return / std_daily_return) * \
                (252 ** 0.5) if std_daily_return != 0 else 0
            self.backtest_logger.info(f"夏普比率: {sharpe_ratio:.2f}")

            # 计算最大回撤
            rolling_max = performance_df["Portfolio Value"].cummax()
            drawdown = (
                performance_df["Portfolio Value"] / rolling_max - 1) * 100
            max_drawdown = drawdown.min()
            self.backtest_logger.info(f"最大回撤: {max_drawdown:.2f}%")

            return performance_df
        except Exception as e:
            self.backtest_logger.error(
                f"性能分析出错: {str(e)}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run backtest simulation')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock code (e.g., 600519)')
    parser.add_argument('--end-date', type=str,
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str,
                        default=(datetime.now() - timedelta(days=90)
                                 ).strftime('%Y-%m-%d'),
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float,
                        default=100000,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--num-of-news', type=int,
                        default=5,
                        help='Number of news articles to analyze (default: 5)')

    args = parser.parse_args()

    backtester = Backtester(
        agent=run_hedge_fund,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        num_of_news=args.num_of_news
    )

    backtester.run_backtest()
    
