from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tools.openrouter_config import get_chat_completion

from agents.state import AgentState, show_agent_reasoning


##### 投资组合管理代理 #####
def portfolio_management_agent(state: AgentState):
    """做出最终交易决策并生成订单"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # 获取技术分析、基本面、风险管理等代理的消息
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")
    risk_message = next(
        msg for msg in state["messages"] if msg.name == "risk_management_agent")

    # 创建系统消息
    system_message = {
        "role": "system",
        "content": """您是一个做出最终交易决策的投资组合经理。
            您的工作是基于团队的分析做出交易决策，同时考虑
            风险管理指导原则。

            风险管理指导原则:
            - 尽可能保持在风险管理的最大仓位限制内
            - 将风险管理的交易行为视为强烈建议
            - 权衡风险信号与潜在机会

            在权衡不同信号的方向和时机时:
            1. 技术分析 (35%权重)
               - 短期交易决策的主要驱动因素
               - 进出场时机的关键
               - 更活跃交易的较高权重
            
            2. 基本面分析 (30%权重)
               - 业务质量和增长评估
               - 决定持仓的信念
            
            3. 估值分析 (25%权重)
               - 长期价值评估
               - 进出场点的二次确认
            
            4. 情绪分析 (10%权重)
               - 最终考虑因素
               - 可影响仓位大小和时机
            
            决策过程应该是:
            1. 评估技术信号的时机
            2. 检查基本面支持
            3. 通过估值确认
            4. 考虑市场情绪
            5. 将风险管理作为指导原则
            
            在输出中提供以下内容:
            - "action": "buy" | "sell" | "hold",
            - "quantity": <正整数>
            - "confidence": <0到1之间的浮点数>
            - "agent_signals": <代理信号列表，包括代理名称、信号(bullish | bearish | neutral)和它们的置信度>
            - "reasoning": <决策的简明解释，包括如何权衡各个信号>

            交易规则:
            - 尽可能保持在风险管理的仓位限制内
            - 只有在有可用现金时才能买入
            - 只有在有股票时才能卖出
            - 卖出数量必须≤当前持仓
            - 将风险管理的最大仓位作为指导原则"""
    }

    # 创建用户消息
    user_message = {
        "role": "user",
        "content": f"""基于以下团队分析，做出您的交易决策。

            技术分析交易信号: {technical_message.content}
            基本面分析交易信号: {fundamentals_message.content}
            情绪分析交易信号: {sentiment_message.content}
            估值分析交易信号: {valuation_message.content}
            风险管理交易信号: {risk_message.content}

            当前投资组合:
            投资组合:
            现金: {portfolio['cash']:.2f}
            当前持仓: {portfolio['stock']} 股

            在输出中仅包含作为JSON的action、quantity、reasoning、confidence和agent_signals。不要包含任何JSON标记。

            记住，action必须是buy、sell或hold之一。
            只有在有可用现金时才能买入。
            只有在投资组合中有股票时才能卖出。"""
    }

    # 从OpenRouter获取完成结果
    result = get_chat_completion([system_message, user_message])

    # 如果API调用失败，返回默认的hold决策
    if result is None:
        result = '''
        {
            "action": "hold",
            "quantity": 0,
            "confidence": 0.5,
            "agent_signals": [
                {"name": "technical_analyst", "signal": "neutral", "confidence": 0.5},
                {"name": "fundamentals", "signal": "neutral", "confidence": 0.5},
                {"name": "sentiment", "signal": "neutral", "confidence": 0.5},
                {"name": "valuation", "signal": "neutral", "confidence": 0.5},
                {"name": "risk_management", "signal": "hold", "confidence": 1.0}
            ],
            "reasoning": "API调用失败，为安全起见默认保持当前持仓"
        }
        '''

    # 创建投资组合管理消息
    message = HumanMessage(
        content=result,
        name="portfolio_management",
    )

    # 如果设置了标志，打印决策
    if show_reasoning:
        show_agent_reasoning(message.content, "投资组合管理代理")

    return {"messages": state["messages"] + [message]}
