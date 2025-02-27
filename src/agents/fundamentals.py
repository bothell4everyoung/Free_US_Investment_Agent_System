from langchain_core.messages import HumanMessage

from agents.state import AgentState, show_agent_reasoning

import json

##### 基本面分析代理 #####
def fundamentals_agent(state: AgentState):
    """分析基本面数据并生成交易信号"""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]

    # 初始化不同基本面方面的信号列表
    signals = []
    reasoning = {}
    
    # 1. 盈利能力分析
    return_on_equity = metrics.get("return_on_equity")
    net_margin = metrics.get("net_margin")
    operating_margin = metrics.get("operating_margin")

    thresholds = [
        (return_on_equity, 0.15),  # ROE强劲超过15%
        (net_margin, 0.20),  # 健康的利润率
        (operating_margin, 0.15)  # 强劲的运营效率
    ]
    profitability_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds
    )
        
    signals.append('bullish' if profitability_score >= 2 else 'bearish' if profitability_score == 0 else 'neutral')
    reasoning["profitability_signal"] = {
        "signal": signals[0],
        "details": (
            f"ROE: {metrics['return_on_equity']:.2%}" if metrics["return_on_equity"] else "ROE: N/A"
        ) + ", " + (
            f"净利润率: {metrics['net_margin']:.2%}" if metrics["net_margin"] else "净利润率: N/A"
        ) + ", " + (
            f"运营利润率: {metrics['operating_margin']:.2%}" if metrics["operating_margin"] else "运营利润率: N/A"
        )
    }
    
    # 2. 增长分析
    revenue_growth = metrics.get("revenue_growth")
    earnings_growth = metrics.get("earnings_growth")
    book_value_growth = metrics.get("book_value_growth")

    thresholds = [
        (revenue_growth, 0.10),  # 10% 收入增长
        (earnings_growth, 0.10),  # 10% 盈利增长
        (book_value_growth, 0.10)  # 10% 账面价值增长
    ]
    growth_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds
    )
        
    signals.append('bullish' if growth_score >= 2 else 'bearish' if growth_score == 0 else 'neutral')
    reasoning["growth_signal"] = {
        "signal": signals[1],
        "details": (
            f"收入增长: {metrics['revenue_growth']:.2%}" if metrics["revenue_growth"] else "收入增长: N/A"
        ) + ", " + (
            f"盈利增长: {metrics['earnings_growth']:.2%}" if metrics["earnings_growth"] else "盈利增长: N/A"
        )
    }
    
    # 3. 财务健康度
    current_ratio = metrics.get("current_ratio")
    debt_to_equity = metrics.get("debt_to_equity")
    free_cash_flow_per_share = metrics.get("free_cash_flow_per_share")
    earnings_per_share = metrics.get("earnings_per_share")

    health_score = 0
    if current_ratio and current_ratio > 1.5:  # 强劲的流动性
        health_score += 1
    if debt_to_equity and debt_to_equity < 0.5:  # 保守的债务水平
        health_score += 1
    if (free_cash_flow_per_share and earnings_per_share and
            free_cash_flow_per_share > earnings_per_share * 0.8):  # 强劲的自由现金流转化
        health_score += 1
        
    signals.append('bullish' if health_score >= 2 else 'bearish' if health_score == 0 else 'neutral')
    reasoning["financial_health_signal"] = {
        "signal": signals[2],
        "details": (
            f"流动比率: {metrics['current_ratio']:.2f}" if metrics["current_ratio"] else "流动比率: N/A"
        ) + ", " + (
            f"债务股本比: {metrics['debt_to_equity']:.2f}" if metrics["debt_to_equity"] else "债务股本比: N/A"
        )
    }
    
    # 4. 市价比率
    pe_ratio = metrics.get("price_to_earnings_ratio")
    pb_ratio = metrics.get("price_to_book_ratio")
    ps_ratio = metrics.get("price_to_sales_ratio")

    thresholds = [
        (pe_ratio, 25),  # 合理的市盈率
        (pb_ratio, 3),  # 合理的市净率
        (ps_ratio, 5)  # 合理的市销率
    ]
    price_ratio_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds
    )
        
    signals.append('bullish' if price_ratio_score >= 2 else 'bearish' if price_ratio_score == 0 else 'neutral')
    reasoning["price_ratios_signal"] = {
        "signal": signals[3],
        "details": (
            f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A"
        ) + ", " + (
            f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A"
        ) + ", " + (
            f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"
        )
    }
    
    # 确定整体信号
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # 计算置信水平
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning
    }
    
    # 创建基本面分析消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="fundamentals_agent",
    )
    
    # 如果设置了标志，则打印推理
    if show_reasoning:
        show_agent_reasoning(message_content, "基本面分析代理")
    
    return {
        "messages": [message],
        "data": data,
    }
