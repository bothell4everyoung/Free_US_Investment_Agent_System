import math

from langchain_core.messages import HumanMessage

from agents.state import AgentState, show_agent_reasoning
from tools.api import prices_to_df

import json
import ast

##### 风险管理代理 #####


def risk_management_agent(state: AgentState):
    """评估投资组合风险并基于综合风险分析设置仓位限制"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]

    prices_df = prices_to_df(data["prices"])

    # Fetch messages from other agents
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")

    try:
        fundamental_signals = json.loads(fundamentals_message.content)
        technical_signals = json.loads(technical_message.content)
        sentiment_signals = json.loads(sentiment_message.content)
        valuation_signals = json.loads(valuation_message.content)
    except Exception as e:
        fundamental_signals = ast.literal_eval(fundamentals_message.content)
        technical_signals = ast.literal_eval(technical_message.content)
        sentiment_signals = ast.literal_eval(sentiment_message.content)
        valuation_signals = ast.literal_eval(valuation_message.content)

    agent_signals = {
        "fundamental": fundamental_signals,
        "technical": technical_signals,
        "sentiment": sentiment_signals,
        "valuation": valuation_signals
    }

    # 1. 计算风险指标
    returns = prices_df['close'].pct_change().dropna()
    daily_vol = returns.std()
    # 年化波动率近似值
    volatility = daily_vol * (252 ** 0.5)

    # 计算波动率分布
    rolling_std = returns.rolling(window=120).std() * (252 ** 0.5)
    volatility_mean = rolling_std.mean()
    volatility_std = rolling_std.std()
    volatility_percentile = (volatility - volatility_mean) / volatility_std

    # 简单的95%置信度历史VaR
    var_95 = returns.quantile(0.05)
    # 使用60天窗口计算最大回撤
    max_drawdown = (prices_df['close'] / prices_df['close'].rolling(window=60).max() - 1).min()

    # 2. 市场风险评估
    market_risk_score = 0

    # 基于百分位的波动率评分
    if volatility_percentile > 1.5:     # 高于1.5个标准差
        market_risk_score += 2
    elif volatility_percentile > 1.0:   # 高于1个标准差
        market_risk_score += 1

    # VaR评分
    if var_95 < -0.03:
        market_risk_score += 2
    elif var_95 < -0.02:
        market_risk_score += 1

    # 最大回撤评分
    if max_drawdown < -0.20:  # 严重回撤
        market_risk_score += 2
    elif max_drawdown < -0.10:
        market_risk_score += 1

    # 3. 仓位大小限制
    current_stock_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_stock_value

    # 总投资组合的25%作为基础仓位
    base_position_size = total_portfolio_value * 0.25

    if market_risk_score >= 4:
        max_position_size = base_position_size * 0.5  # 高风险时减少仓位
    elif market_risk_score >= 2:
        max_position_size = base_position_size * 0.75  # 中等风险时略微减少仓位
    else:
        max_position_size = base_position_size  # 低风险时保持基础仓位

    # 4. 风险调整后的信号分析
    def parse_confidence(conf_str):
        try:
            if isinstance(conf_str, str):
                return float(conf_str.replace('%', '')) / 100.0
            return float(conf_str)
        except:
            return 0.0

    # 检查低置信度信号
    low_confidence = any(parse_confidence(
        signal['confidence']) < 0.30 for signal in agent_signals.values())

    # 检查信号分歧
    unique_signals = set(signal['signal'] for signal in agent_signals.values())
    signal_divergence = (2 if len(unique_signals) == 3 else 0)

    # 计算最终风险分数
    risk_score = market_risk_score + \
        (2 if low_confidence else 0) + signal_divergence
    risk_score = min(round(risk_score), 10)

    # 5. 确定交易行为
    # 更灵活的方法，考虑技术信号
    technical_confidence = parse_confidence(
        agent_signals['technical']['confidence'])
    fundamental_confidence = parse_confidence(
        agent_signals['fundamental']['confidence'])

    if risk_score >= 9:
        trading_action = "hold"  # 极端风险，强制持仓
    elif risk_score >= 7:
        # 高风险但考虑强劲的技术信号
        if (technical_signals['signal'] == 'bullish' and technical_confidence > 0.7 and
                fundamental_signals['signal'] == 'bullish'):
            trading_action = "buy"
        else:
            trading_action = "reduce"
    else:
        # 正常风险环境
        if technical_signals['signal'] == 'bullish' and technical_confidence > 0.5:
            trading_action = "buy"
        elif technical_signals['signal'] == 'bearish' and technical_confidence > 0.5:
            trading_action = "sell"
        else:
            trading_action = "hold"

    message_content = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "risk_metrics": {
            "volatility": float(volatility),
            "value_at_risk_95": float(var_95),
            "max_drawdown": float(max_drawdown),
            "market_risk_score": market_risk_score
        },
        "reasoning": f"风险评分 {risk_score}/10: 市场风险={market_risk_score}, "
                     f"波动率={volatility:.2%}, VaR={var_95:.2%}, "
                     f"最大回撤={max_drawdown:.2%}"
    }

    # 创建风险管理消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "风险管理代理")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
