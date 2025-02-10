from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
from tools.news_crawler import get_stock_tweets, get_news_sentiment
from tools.openrouter_config import get_chat_completion
import json
from datetime import datetime, timedelta


def sentiment_agent(state: AgentState):
    """分析市场情绪并生成交易信号"""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    symbol = data["ticker"]
    current_date = data["end_date"]  # 使用回测的当前日期

    # 从命令行参数获取推文日期，默认为5天内
    num_of_news = data.get("num_of_news", 5)

    # 获取推文数据并使用历史日期分析情绪
    news_list = get_stock_tweets(symbol, date=current_date, max_articles=num_of_news)

    sentiment_score = get_news_sentiment(
        news_list, date=current_date, num_of_news=num_of_news)

    # 根据情绪分数生成交易信号和置信度
    if sentiment_score >= 0.5:
        signal = "看涨"
        confidence = str(round(abs(sentiment_score) * 100)) + "%"
    elif sentiment_score <= -0.5:
        signal = "看跌"
        confidence = str(round(abs(sentiment_score) * 100)) + "%"
    else:
        signal = "中性"
        confidence = str(round((1 - abs(sentiment_score)) * 100)) + "%"

    # 生成分析结果
    message_content = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": f"基于截至{current_date}的{len(news_list)}篇最近新闻文章，情绪分数: {sentiment_score:.2f}"
    }

    # 如果设置了标志，则显示推理
    if show_reasoning:
        show_agent_reasoning(message_content, "情绪分析代理")

    # 创建消息
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_agent",
    )

    return {
        "messages": [message],
        "data": data,
    }
