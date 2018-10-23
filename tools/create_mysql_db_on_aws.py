import pandas as pd
import zipfile
import shutil
import os.path

# from data.db_config import db_conn, db_engine

# Lahman database csv files: http://www.seanlahman.com/baseball-archive/statistics/


def unzip_files():
    zip_ref = zipfile.ZipFile('baseballdatabank-master_2018-03-28.zip', 'r')
    zip_ref.extractall('lahman')
    zip_ref.close()
    shutil.move('lahman', os.path.abspath(os.path.join(os.pardir, 'data/lahman')))
    return


def read_in_csv_files():
    directory = os.path.abspath(os.path.join(os.pardir, 'data/lahman/baseballdatabank-master/core'))
    all_star_df = pd.read_csv((os.path.join((directory), 'AllstarFull.csv')))
    print(all_star_df.head())
    batting_df = pd.read_csv((os.path.join((directory), 'Batting.csv')))
    salaries_df = pd.read_csv((os.path.join((directory), 'Salaries.csv')))
    people_df = pd.read_csv((os.path.join((directory), 'People.csv')))
    awards_df = pd.read_csv((os.path.join((directory), 'Awards.csv')))
    hall_of_fame_df = pd.read_csv((os.path.join((directory), 'HallOfFame.csv')))
    return all_star_df, batting_df, salaries_df, people_df, awards_df, hall_of_fame_df


def create_table(ddl):
    cursor = db_conn.cursor()
    cursor.execute(ddl)
    return


def write_data_to_table(df, table):
    df.to_sql(name=table, con=db_engine, if_exists='append', index=False)
    return


if __name__ == "__main__":
    # unzip_files()
    all_star_df, batting_df, salaries_df, people_df, awards_df, hall_of_fame_df = read_in_csv_files()

