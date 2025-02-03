from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
import json

def valuation_agent(state: AgentState):
    """使用多种方法执行详细的估值分析"""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]
    current_financial_line_item = data["financial_line_items"][0]
    previous_financial_line_item = data["financial_line_items"][1]
    market_cap = data["market_cap"]

    reasoning = {}

    # 计算营运资金变动
    working_capital_change = (current_financial_line_item.get('working_capital') or 0) - (previous_financial_line_item.get('working_capital') or 0)
    
    # 所有者收益估值（巴菲特方法）
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get('net_income'),
        depreciation=current_financial_line_item.get('depreciation_and_amortization'),
        capex=current_financial_line_item.get('capital_expenditure'),
        working_capital_change=working_capital_change,
        growth_rate=metrics["earnings_growth"],
        required_return=0.15,
        margin_of_safety=0.25
    )
    
    # DCF估值
    dcf_value = calculate_intrinsic_value(
        free_cash_flow=current_financial_line_item.get('free_cash_flow'),
        growth_rate=metrics["earnings_growth"],
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        num_years=5,
    )
    
    # 计算综合估值差距（两种方法的平均值）
    dcf_gap = (dcf_value - market_cap) / market_cap
    owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
    valuation_gap = (dcf_gap + owner_earnings_gap) / 2

    if valuation_gap > 0.15:  # 低估超过15%
        signal = 'bullish'
    elif valuation_gap < -0.15:  # 高估超过15%
        signal = 'bearish'
    else:
        signal = 'neutral'

    reasoning["dcf_analysis"] = {
        "signal": "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral",
        "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}"
    }

    reasoning["owner_earnings_analysis"] = {
        "signal": "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral",
        "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}"
    }

    message_content = {
        "signal": signal,
        "confidence": f"{abs(valuation_gap):.0%}",
        "reasoning": reasoning
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Valuation Analysis Agent")

    return {
        "messages": [message],
        "data": data,
    }

def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5
) -> float:
    """
    使用巴菲特的所有者收益法计算内在价值
    
    所有者收益 = 净利润 
                + 折旧/摊销
                - 资本支出
                - 营运资金变动
    
    参数:
        net_income: 年度净利润
        depreciation: 年度折旧和摊销
        capex: 年度资本支出
        working_capital_change: 年度营运资金变动
        growth_rate: 预期增长率
        required_return: 要求回报率（巴菲特通常使用15%）
        margin_of_safety: 安全边际
        num_years: 预测年数
    
    返回:
        float: 考虑安全边际的内在价值
    """
    if not all([isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]]):
        return 0
    
    # 计算初始所有者收益
    owner_earnings = (
        net_income +
        depreciation -
        capex -
        working_capital_change
    )
    
    if owner_earnings <= 0:
        return 0

    # 预测未来所有者收益
    future_values = []
    for year in range(1, num_years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        discounted_value = future_value / (1 + required_return) ** year
        future_values.append(discounted_value)
    
    # 计算终值（使用永续增长公式）
    terminal_growth = min(growth_rate, 0.03)  # 终值增长率上限为3%
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / (1 + required_return) ** num_years
    
    # 求和所有价值并应用安全边际
    intrinsic_value = (sum(future_values) + terminal_value_discounted)
    value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)
    
    return value_with_safety_margin


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    基于当前自由现金流计算公司的贴现现金流(DCF)。
    使用此函数计算股票的内在价值。
    """
    # Estimate the future cash flows based on the growth rate
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]

    # Calculate the present value of projected cash flows
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)

    # Calculate the terminal value
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    # Sum up the present values and terminal value
    dcf_value = sum(present_values) + terminal_present_value

    return dcf_value

def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    计算两个期间之间的营运资金绝对变化。
    正变化意味着更多资金被占用在营运资金中（现金流出）。
    负变化意味着较少资金被占用（现金流入）。
    
    参数:
        current_working_capital: 当期营运资金
        previous_working_capital: 上期营运资金
    
    返回:
        float: 营运资金变动（当期 - 上期）
    """
    return current_working_capital - previous_working_capital