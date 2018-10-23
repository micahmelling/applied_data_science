import pymysql.cursors
from sqlalchemy import create_engine


db_conn = pymysql.connect(host={},
                          user='{}',
                          password='{}',
                          db='{}',
                          charset='utf8mb4',
                          cursorclass=pymysql.cursors.DictCursor)

db_engine = create_engine('mysql+pymysql://{}:{}@{}:3306/baseball_data_science', echo=False)
