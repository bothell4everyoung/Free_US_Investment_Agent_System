from datetime import datetime, timedelta
import argparse
from agents.valuation import valuation_agent
from agents.state import AgentState
from agents.sentiment import sentiment_agent
from agents.risk_manager import risk_management_agent
from agents.technicals import technical_analyst_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.market_data import market_data_agent
from agents.fundamentals import fundamentals_agent
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from args_processor import parse_and_validate_args

load_dotenv()  # 加载 .env 文件中的环境变量


##### Run the Hedge Fund #####
def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False, num_of_news: int = 5):
    # 定义新的工作流程
    graph_builder = StateGraph(AgentState)

    # 添加节点
    graph_builder.add_node("market_data_agent", market_data_agent)
    graph_builder.add_node("technical_analyst_agent", technical_analyst_agent)
    graph_builder.add_node("fundamentals_agent", fundamentals_agent)
    graph_builder.add_node("sentiment_agent", sentiment_agent)
    graph_builder.add_node("risk_management_agent", risk_management_agent)
    graph_builder.add_node("portfolio_management_agent", portfolio_management_agent)
    graph_builder.add_node("valuation_agent", valuation_agent)

    # 定义边
    graph_builder.set_entry_point("market_data_agent")
    graph_builder.add_edge("market_data_agent", "technical_analyst_agent")
    graph_builder.add_edge("market_data_agent", "fundamentals_agent")
    graph_builder.add_edge("market_data_agent", "sentiment_agent")
    graph_builder.add_edge("market_data_agent", "valuation_agent")
    graph_builder.add_edge("technical_analyst_agent", "risk_management_agent")
    graph_builder.add_edge("fundamentals_agent", "risk_management_agent")
    graph_builder.add_edge("sentiment_agent", "risk_management_agent")
    graph_builder.add_edge("valuation_agent", "risk_management_agent")
    graph_builder.add_edge("risk_management_agent", "portfolio_management_agent")
    graph_builder.add_edge("portfolio_management_agent", END)

    graph = graph_builder.compile()
    final_state = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="根据提供的数据做出交易决策。",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "num_of_news": num_of_news,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    return final_state["messages"][-1].content


if __name__ == "__main__":
    args = parse_and_validate_args()

    # 使用初始资本配置投资组合
    portfolio = {
        "cash": args.initial_capital,
        "stock": 0  # 没有初始股票头寸
    }

    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        num_of_news=args.num_of_news
    )
    print("\n最终结果:")
    print(result)
