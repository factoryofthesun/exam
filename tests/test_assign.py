# Test random assignment
import sys
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from pathlib import Path
import random

from exam import compute_probs, assign

dropbox_path = "/Users/rl874/Dropbox/aneesha/input"

@pytest.fixture
def data_numpy():
    df = pd.read_csv("test_data/test_input_types.csv", index_col = 0)
    subject_ids = np.array(df.index)
    treatment_labels = np.array(df.columns)
    array = np.array(df)
    return (array, subject_ids, treatment_labels)

@pytest.fixture
def data_df():
    df = pd.read_csv("test_data/test_input_types.csv", index_col = 0)
    return df

@pytest.fixture
def data_df_n():
    df_n = pd.read_csv("test_data/test_n_treatments.csv", index_col = 0)
    return df_n

def test_label_params(data_numpy, data_df, data_df_n):
    probs, subject_ids, treatment_labels = data_numpy
    ret_np = assign(probs, treatment_labels, subject_ids)
    assert np.array_equal(ret_np.index, subject_ids)
    assert np.isin(ret_np, treatment_labels).all()

    # Assignment realization shouldn't differ from expected counts that much
    np_counts = pd.DataFrame(ret_np.assignment.value_counts())
    np_counts.columns = ['assigned_counts']
    expected_counts = pd.DataFrame(np.sum(probs, axis=0), index = treatment_labels, columns = ['expected_counts'])
    check_counts = np_counts.merge(expected_counts, left_index = True, right_index = True)
    check_counts['perc_diff'] = (check_counts.assigned_counts - check_counts.expected_counts).abs()/check_counts.assigned_counts
    assert (check_counts.perc_diff <= 0.1).all()

    ret_df = assign(data_df, data_df.columns, data_df.index)
    assert np.array_equal(ret_df.index, data_df.index)
    assert np.isin(ret_df, data_df.columns).all()

    # Assignment realization shouldn't differ from expected counts that much
    np_counts = pd.DataFrame(ret_df.assignment.value_counts(), columns = ['assigned_counts'])
    expected_counts = pd.DataFrame(np.sum(data_df, axis=0), index = data_df.columns, columns = ['expected_counts'])
    check_counts = np_counts.merge(expected_counts, left_index = True, right_index = True)
    check_counts['perc_diff'] = (check_counts.assigned_counts - check_counts.expected_counts).abs()/check_counts.assigned_counts
    assert (check_counts.perc_diff <= 0.2).all()

    ret_df_n = assign(data_df_n, data_df_n.columns, data_df_n.index)
    assert np.array_equal(ret_df_n.index, data_df_n.index)
    assert np.isin(ret_df_n, data_df_n.columns).all()

    # Assignment realization shouldn't differ from expected counts that much
    np_counts = pd.DataFrame(ret_df_n.assignment.value_counts(), columns = ['assigned_counts'])
    expected_counts = pd.DataFrame(np.sum(data_df_n, axis=0), index = data_df_n.columns, columns = ['expected_counts'])
    check_counts = np_counts.merge(expected_counts, left_index = True, right_index = True)
    check_counts['perc_diff'] = (check_counts.assigned_counts - check_counts.expected_counts).abs()/check_counts.assigned_counts
    assert (check_counts.perc_diff <= 0.2).all()

def test_seed(data_numpy, data_df):
    probs, subject_ids, treatment_labels = data_numpy
    seed = np.random.choice(100)
    ret_np0 = assign(probs, treatment_labels, subject_ids, seed = seed)
    ret_np1 = assign(probs, treatment_labels, subject_ids, seed = seed)
    pd.testing.assert_frame_equal(ret_np0, ret_np1)

    ret_df1 = assign(data_df, data_df.columns, data_df.index, seed = seed)
    np.testing.assert_array_equal(ret_np0.values, ret_df1.values)

def test_save(data_df_n):
    ret_df_n = assign(data_df_n, data_df_n.columns, data_df_n.index, save_path = "test_data/test_n_assign.csv")
    load_df_n = pd.read_csv("test_data/test_n_assign.csv", index_col = 0)
    pd.testing.assert_frame_equal(ret_df_n, load_df_n)
