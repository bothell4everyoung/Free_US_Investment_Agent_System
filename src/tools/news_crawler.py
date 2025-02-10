import os
import sys
import json
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from tools.openrouter_config import get_chat_completion, logger as api_logger
import logging
import time
import pandas as pd
from tools.api import get_tweets_about_ticker
# 设置日志记录
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     handlers=[
#                         logging.FileHandler('logs/news_crawler.log'),
#                         logging.StreamHandler()
#                     ])
logger = logging.getLogger(__name__)


def fetch_article_content(url: str) -> str:
    """从 URL 获取文章内容，使用 BeautifulSoup

    参数:
        url (str): 文章 URL

    返回:
        str: 文章内容，如果失败则返回空字符串
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.decompose()
            # 获取文本内容
            text = soup.get_text()
            # 按行分割并去除前后空格
            lines = (line.strip() for line in text.splitlines())
            # 将多头条分割成每行一个
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
            # 删除空行
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text[:5000]  # 限制内容长度
        return ""
    except Exception as e:
        logger.error(f"获取文章内容失败: {e}")
        return ""


def get_stock_tweets(symbol: str, date: str = None, max_articles: int = 5) -> list:
    """从数据库获取和处理股票推文

    参数:
        symbol (str): 股票代码，例如 "AAPL"
        date (str, 可选): 获取新闻的截止日期 (YYYY-MM-DD)。如果为 None，则使用当前日期。
        max_articles (int, 可选): 最大新闻条数。默认为 5。

    返回:
        list: 新闻文章列表，每篇文章包含标题、内容、发布时间等。
    """
    # 如果未提供日期，则获取当前日期
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # 计算开始时间为 date 的 max_articles 天之前
    start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=max_articles)).strftime("%Y-%m-%d")
    logger.info(f"开始时间: {start_date}")
    logger.info(f'开始获取 {symbol} 的推文，截止到 {date}...')

    try:
        # 从数据库获取新闻
        # 这里需要添加数据库查询的逻辑
        news_list = get_tweets_about_ticker(symbol, start_date, date)

        if not news_list:
            logger.warning(f"未找到 {symbol} 的推文")
            return []

        # 记录获取的新闻数量
        logger.info(f"获取到的新闻数量: {len(news_list)}")

        return news_list

    except Exception as e:
        logger.error(f"获取新闻数据失败: {e}")
        return []


def get_news_sentiment(news_list: list, date: str = None, num_of_news: int = 5) -> float:
    """使用 LLM 分析新闻情感

    参数:
        news_list (list): 新闻文章列表
        date (str, 可选): 情感分析的日期 (YYYY-MM-DD)。如果为 None，则使用当前日期。
        num_of_news (int, 可选): 要分析的新闻文章数量。默认为 5。

    返回:
        float: 情感分数，范围在 -1 到 1 之间
    """
    if not news_list:
        return 0.0

    # 如果未提供日期，则获取当前日期
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # 检查缓存
    cache_file = "src/data/sentiment_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # 仅使用日期作为键检查缓存
    if os.path.exists(cache_file):
        logger.info("找到情感分析缓存文件")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if date in cache:
                    logger.info("使用缓存的情感分析结果")
                    return cache[date]
                logger.info("未找到匹配的情感分析缓存")
        except Exception as e:
            logger.error(f"读取情感缓存失败: {e}")
            cache = {}
    else:
        logger.info(
            "未找到情感分析缓存文件，将创建新文件")
        cache = {}

    # 准备系统消息
    system_message = {
        "role": "system",
        "content": """你是一位专业的美国股市分析师，专注于新闻情感分析。你需要分析一组新闻文章并提供一个情感分数，范围在 -1 到 1 之间：
        - 1 代表极其积极（例如，重大积极新闻、突破性收益、强大的行业支持）
        - 0.5 到 0.9 代表积极（例如，收益增长、新项目启动、合同赢得）
        - 0.1 到 0.4 代表稍微积极（例如，小合同签署、正常运营）
        - 0 代表中立（例如，例行公告、人事变动、无影响新闻）
        - -0.1 到 -0.4 代表稍微消极（例如，小型诉讼、非核心业务损失）
        - -0.5 到 -0.9 代表消极（例如，业绩下滑、重大客户流失、行业监管收紧）
        - -1 代表极其消极（例如，重大违规、核心业务严重损失、监管处罚）

        关注点：
        1. 业绩相关：财务报告、收益预测、收入/利润
        2. 政策影响：行业政策、监管政策、地方政策
        3. 市场表现：市场份额、竞争地位、商业模式
        4. 资本运作：并购、股权激励、增发
        5. 风险事件：诉讼、仲裁、处罚
        6. 行业地位：技术创新、专利、市场份额
        7. 公众舆论：媒体评价、社会影响

        请确保分析：
        1. 新闻的真实性和可靠性
        2. 新闻的时效性和影响范围
        3. 对公司基本面的实际影响
        4. 美国股市的特定反应模式"""
    }

    # 准备新闻内容
    news_content = "\n\n".join([
        f"作者: {news['Author']}\n"
        f"推文连接: {news['Link']}\n"
        f"时间: {news['PublishedDate']}\n"
        f"内容: {news['Content']}"
        for news in news_list[:num_of_news]
    ])

    user_message = {
        "role": "user",
        "content": f"请分析以下与美国股票相关的新闻情感:\n\n{news_content}\n\n请仅返回一个介于 -1 和 1 之间的数字，不需要解释。"
    }

    try:
        # 获取 LLM 分析结果
        result = get_chat_completion([system_message, user_message])
        if result is None:
            logger.error("错误: LLM 返回 None")
            return 0.0

        # 提取数字结果
        try:
            sentiment_score = float(result.strip())
        except ValueError as e:
            logger.error(f"解析情感分数时出错: {e}")
            logger.error(f"原始结果: {result}")
            return 0.0

        # 确保分数在 -1 和 1 之间
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # 使用日期作为键缓存结果
        cache[date] = sentiment_score
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.info(
                f"成功缓存日期 {date} 的情感分数 {sentiment_score}")
        except Exception as e:
            logger.error(f"写入缓存时出错: {e}")

        return sentiment_score

    except Exception as e:
        logger.error(f"分析新闻情感时出错: {e}")
        return 0.0  # 在出错时返回中性分数
