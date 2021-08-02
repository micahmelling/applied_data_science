import pytest
import pandas as pd

from helpers.model_helpers import clip_feature_bounds, find_non_dummied_columns, create_uid, CombineCategoryLevels, \
    TakeLog


@pytest.fixture
def simple_num_df():
    return pd.DataFrame({
        'col_a': [10, 15, 15, 20],
        'dummied_col': [1, 0, 1, 0]
    })


@pytest.fixture
def simple_cat_df():
    return pd.DataFrame({
        'col_a': ['a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
    })


def test_clip_feature_bounds(simple_num_df):
    df = clip_feature_bounds(df=simple_num_df, feature='col_a', cutoff=17, new_amount=17, clip_type='upper')
    assert df['col_a'].max() == 17
    df = clip_feature_bounds(df=simple_num_df, feature='col_a', cutoff=12, new_amount=12, clip_type='lower')
    assert df['col_a'].min() == 12


def test_find_non_dummied_columns(simple_num_df):
    assert find_non_dummied_columns(simple_num_df) == ['col_a']


def test_create_uid():
    base_string = 'random_forest'
    uid = create_uid(base_string)
    assert len(uid) > len(base_string)


def test_CombineCategoryLevels(simple_cat_df):
    combiner = CombineCategoryLevels(sparsity_cutoff=0.10)
    df = combiner.fit_transform(simple_cat_df)
    assert set(df['col_a'].tolist()) == {'b', 'sparse_combined'}


def test_TakeLog(simple_num_df):
    log_taker = TakeLog()
    df = log_taker.fit_transform(simple_num_df)
    assert round(df['col_a'][0], 3) == 2.303
