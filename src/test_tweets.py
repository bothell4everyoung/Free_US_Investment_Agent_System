from tools.api import get_tweets_about_ticker
def test_get_tweets_about_ticker():
    """测试获取特定股票相关推文的功能"""

    # 设置要查询的股票代码和日期范围
    symbol = 'BTC'  # 替换为您想要查询的股票代码
    # start_date = '2023-01-01'  # 可选，开始日期
    # end_date = '2023-01-31'  # 可选，结束日期

    # 调用函数获取推文
    tweets = get_tweets_about_ticker(symbol)
    # 打印结果
    for tweet in tweets:
        # print(f"推文内容: {tweet['Content']}, 发布时间: {tweet['PublishedDate']}")
        print(tweet['Link'])

if __name__ == "__main__":
    test_get_tweets_about_ticker()
