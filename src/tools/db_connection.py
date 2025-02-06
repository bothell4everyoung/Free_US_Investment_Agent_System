from sqlalchemy import create_engine
import os  

def get_stocks_db_connection():
    """获取 MySQL 数据库连接"""
    engine = create_engine(f'mysql+mysqlconnector://root:{os.getenv("DB_PASSWORD")}@localhost/stocks')
    return engine.connect() 

def get_tweets_db_connection():
    """获取 MySQL 数据库连接"""
    engine = create_engine(f'mysql+mysqlconnector://root:{os.getenv("DB_PASSWORD")}@localhost/tweets')
    return engine.connect() 