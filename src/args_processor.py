from datetime import datetime, timedelta
import argparse

def parse_and_validate_args():
    parser = argparse.ArgumentParser(
        description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD). Defaults to 3 months before end date')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD). Defaults to today')
    parser.add_argument('--show-reasoning', action='store_true',
                       help='Show reasoning from each agent')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial cash amount (default: 100,000)')
    parser.add_argument('--num-of-news', type=int, default=5,
                       help='Number of news articles to analyze for sentiment (default: 5)')

    args = parser.parse_args()
    
    # Process dates
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    if not args.start_date:
        end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d')
        start_date_obj = end_date_obj - timedelta(days=90)
        args.start_date = start_date_obj.strftime('%Y-%m-%d')

    # Validate inputs
    validate_dates(args.start_date, args.end_date)
    validate_news_count(args.num_of_news)

    return args

def validate_dates(start_date: str, end_date: str):
    """Validate the date format."""
    for date_str in [start_date, end_date]:
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Date {date_str} must be in YYYY-MM-DD format")

def validate_news_count(num_of_news: int):
    """Validate the number of news articles."""
    if num_of_news < 1:
        raise ValueError("Number of news articles must be at least 1")
    if num_of_news > 100:
        raise ValueError("Number of news articles cannot exceed 100") 