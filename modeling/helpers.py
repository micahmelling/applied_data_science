import pandas as pd
import numpy as np


def calculate_average_vote_percentage(df):
    df['vote_percentage'] = df['votes'] / df['ballots']
    df['vote_percentage'].fillna(value=0, inplace=True)

    grouped = pd.DataFrame(df.groupby('player_id')['vote_percentage'].mean())
    grouped.columns = ['average_vote_percentage']
    grouped.reset_index(inplace=True)
    return grouped


def drop_pitchers(df):
    df = pd.DataFrame(df.groupby('player_id').agg({'GS': 'sum', 'SV': 'sum'}))
    df.reset_index(inplace=True)
    df = df.loc[(df.loc[df['GS'] < 4]) | (df.loc[df['SV'] < 4]) & (df.loc[df['playerID'] != 'ruthba01'])]
    return df


def find_debut_year_and_decade(df):
    df = df[['player_id', 'year_id']]
    df = df.sort_values(by=['player_id', 'year_id'], ascending=[True, True])
    df.drop_duplicates(subset='player_id', keep='first', inplace=True)
    df['debut_decade'] = df['year_id'].str[0:3]
    df.rename(columns={'year_id': 'debut_decade'}, inplace=True)
    return df


def find_most_frequent_team(df):
    df = pd.DataFrame(df.groupby('playerID')['teamID'].agg(lambda x: x.mode() if len(x) > 2 else np.array(x)))
    df.columns = ['most_frequent_team']
    df.reset_index(inplace=True)
    return df


def aggregate_offensive_data(df):
    df['IBB'] = pd.to_numeric(df['IBB'], errors='coerce')
    df['HBP'] = pd.to_numeric(df['HBP'], errors='coerce')
    df['SH'] = pd.to_numeric(df['SH'], errors='coerce')
    df['SF'] = pd.to_numeric(df['SF'], errors='coerce')
    df['GIDP'] = pd.to_numeric(df['GIDP'], errors='coerce')

    df.fillna(value=0, inplace=True)
    df = df.groupby('playerID').agg({'G': 'sum', 'AB': 'sum', 'R': 'sum', 'H': 'sum', '2B': 'sum', '3B': 'sum',
                                     'HR': 'sum', 'RBI': 'sum', 'SB': 'sum', 'CS': 'sum', 'BB': 'sum', 'SO': 'sum',
                                     'IBB': 'sum', 'HBP': 'sum', 'SH': 'sum', 'SF': 'sum', 'GIDP': 'sum'})
    return df


def calculate_offensive_statistics(df):
    df['stolen_base_percentage'] = df['SB'] / (df['SB'] + df['CS'])
    df['obp'] = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF'])
    df['slg'] = (df['H'] - (df['2B'] + df['3B'] + df['HR']) + df['2B'] * 2  + df['3B'] * 3 + df['HR'] * 4) / df['AB']
    df['avg'] = df['H'] / df['AB']
    df['ops'] = df['obp'] + df['slg']
    df['extra_base_hits'] = df['HR'] + df['2B'] + df['3B']
    return df


def aggregate_pitching_data(df):
    df = df.groupby('playerID').agg({'put_outs': 'sum', 'assists': 'sum', 'errors': 'sum'})
    # TODO: add fielding percentage
    return df


def find_most_common_position(df):
    df = df.groupby('player_id')['position'].agg(lambda x: x.mode() if len(x) > 2 else np.array(x))
    df.columns = ['most_frequent_position']
    df.reset_index(inplace=True)
    return df


def count_number_of_mvps_and_gold_gloves(df):
    df = df.groupby(['playerID', 'awardID'])['awardID'].agg('count').unstack()
    df.reset_index(inplace=True)

    df['Gold Glove'].fillna(value=0, inplace=True)
    df['Most Valuable Player'].fillna(value=0, inplace=True)

    df['Gold Glove'] = df['Gold Glove'].astype('int')
    df['Most Valuable Player'] = df['Most Valuable Player'].astype('int')

    df.rename(columns={'Gold Glove': 'Gold_Gloves', 'Most Valuable Player': 'MVPs'}, inplace=True)
    return df


def find_world_series_wins_and_losses(df):
    return df




