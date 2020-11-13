# Test optimization algorithm
import sys
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from pathlib import Path
import random

from exam import compute_probs

dropbox_path = "/Users/rl874/Dropbox/aneesha/input"

@pytest.fixture
def data_numpy():
    d = np.random.choice(range(1,1001))
    pte_df = pd.read_csv(f"{dropbox_path}/WTP_HTE_forPythonEXaMalgorithm/PTE_dia_all_{d}_COARSE.csv")
    wtp_df = pd.read_csv(f"{dropbox_path}/WTP_HTE_forPythonEXaMalgorithm/WTP_wdays_all_{d}_COARSE.csv")

    pte_matrix = np.array([[0, i] for i in pte_df['PTE'].values.tolist()])
    wtp_matrix = np.array([[0, i] for i in wtp_df['WTP'].values.tolist()])
    subject_ids = np.array(pte_df['subject'])

    return (pte_matrix, wtp_matrix, subject_ids)

@pytest.fixture
def data_df():
    d = np.random.choice(range(1,1001))
    pte_df = pd.read_csv(f"{dropbox_path}/WTP_HTE_forPythonEXaMalgorithm/PTE_dia_all_{d}_COARSE.csv")
    wtp_df = pd.read_csv(f"{dropbox_path}/WTP_HTE_forPythonEXaMalgorithm/WTP_wdays_all_{d}_COARSE.csv")

    pte_matrix = pd.DataFrame([[0, i] for i in pte_df['PTE'].values.tolist()], columns = ["t1", "t2"])
    wtp_matrix = pd.DataFrame([[0, i] for i in wtp_df['WTP'].values.tolist()], columns = ["t1", "t2"])
    subject_ids = pte_df['subject']

    return (pte_matrix, wtp_matrix, subject_ids)

@pytest.fixture
def data_n_treatments():
    # Generate random WTP, treatment effects, and number of treatments
    n_treatments = np.random.choice(range(2, 5))
    print(f"data_n_treatments: generated {n_treatments} treatments")
    wtp = np.random.uniform(0, 100, size = (1000, n_treatments))
    treatment = np.random.uniform(0, 100, size = (1000, n_treatments))

    return (treatment, wtp)

def test_inp_errors():
    n_subjects = 100
    wtp = np.random.uniform(size = (n_subjects, 2))
    treatment_effects = np.random.uniform(size = (n_subjects, 2))
    rct_prob = 0.5
    error_threshold = 0.1

    # Probability bound should never go above minimum RCT prob
    bad_probability = rct_prob + 1e-4
    with pytest.raises(ValueError):
        ret = compute_probs(wtp, treatment_effects, probability_bound = bad_probability)

    bad_capacity = [20, 80]
    bad_probability = 0.2 + 1e-4
    with pytest.raises(ValueError):
        ret = compute_probs(wtp, treatment_effects, capacity = bad_capacity, probability_bound = bad_probability)

    # Total capacity needs to cover total subjects
    bad_capacity = [50,49]
    with pytest.raises(ValueError):
        ret = compute_probs(wtp, treatment_effects, capacity = bad_capacity)

    # Budget must be positive
    with pytest.raises(ValueError):
        ret = compute_probs(wtp, treatment_effects, budget = 0)

def test_threshold():
    # Test that output probabilities given extreme bound of 0.5 are within error threshold
    n_subjects = 100
    wtp = np.random.uniform(size = (n_subjects, 2))
    treatment_effects = np.random.uniform(size = (n_subjects, 2))
    rct_prob = 0.5
    error_threshold = 0.1

    ret_good = compute_probs(wtp, treatment_effects, probability_bound = rct_prob, error_threshold = error_threshold)
    good_probs = np.array([ret_good['p_star'].loc[:,0], ret_good['p_star'].loc[:,1]]).flatten()
    good_err = np.sqrt(np.sum((good_probs - 0.5)**2))/n_subjects

    # Deviation from 0.5 must be within clearing error
    print(good_probs)
    assert good_err == ret_good['error']
    assert good_err < error_threshold

def test_trivial_allocation():
    # Each participant has positive WTP/treatment effect for a different treatment
    wtp_null = np.array([0] * 9).reshape(3,3)
    treatment_effects_null = np.array([0] * 9).reshape(3,3)
    treatment_effects = np.array([[1,0,0],[0,1,0], [0,0,1]])
    wtp = np.array([[1,0,0],[0,1,0], [0,0,1]])

    # ret_wtp_null = compute_probs(wtp_null, treatment_effects)
    # ret_treatment_null = compute_probs(wtp, treatment_effects_null)
    ret_full = compute_probs(wtp, treatment_effects)

    # print("P-star for WTP 0s")
    # print(ret_wtp_null['p_star'])
    # print("P-star for Treatment Effect 0s")
    # print(ret_treatment_null['p_star'])
    print("P-star for Fully Differentiated Preferences")
    print(ret_full['p_star'])
    # assert ret_wtp_null['p_star'].shape == ret_treatment_null['p_star'].shape == ret_full['p_star'].shape
    full_pstar = ret_full['p_star']
    for i in range(full_pstar.shape[0]):
        for j in range(full_pstar.shape[1]):
            if i == j:
                pass
            else:
                assert full_pstar.iloc[i, j] < 1e-4

    # Probabilities for each subject must sum to 1
    # assert np.all(ret_wtp_null['p_star'].sum(axis=1) == 1)
    # assert np.all(ret_treatment_null['p_star'].sum(axis=1) == 1)
    np.testing.assert_allclose(ret_full['p_star'].sum(axis=1), np.ones(3))

def test_trivial_capacity():
    n_subjects = 100
    wtp = np.random.uniform(size = (n_subjects, 2))
    treatment_effects = np.random.uniform(size = (n_subjects, 2))
    capacity = [n_subjects] * 2

    ret = compute_probs(wtp, treatment_effects, capacity)

    assert ret['error'] == 0

def test_input_types(data_numpy, data_df):
    pte_numpy, wtp_numpy, subject_numpy = data_numpy
    pte_df, wtp_df, subject_df = data_df

    # Budget, thresholds, and capacity
    ret_np = compute_probs(wtp_numpy, pte_numpy)
    ret_df = compute_probs(wtp_df, pte_df)
    ret_np1 = compute_probs(wtp_numpy, pte_numpy, budget = 10000, error_threshold = 0.001)
    ret_df1 = compute_probs(wtp_df, pte_df, budget = 0.01, error_threshold = 0.05)
    ret_np2 = compute_probs(wtp_numpy, pte_df, capacity = [663, 877], probability_bound = 0.43)
    ret_df2 = compute_probs(wtp_df, pte_numpy, iterations_threshold = 100)

    # Probabilities for each subject must sum to 1
    np.testing.assert_allclose(ret_np['p_star'].sum(axis=1), np.ones(pte_numpy.shape[0]))
    np.testing.assert_allclose(ret_df['p_star'].sum(axis=1), np.ones(pte_numpy.shape[0]))
    np.testing.assert_allclose(ret_np1['p_star'].sum(axis=1), np.ones(pte_numpy.shape[0]))
    np.testing.assert_allclose(ret_df1['p_star'].sum(axis=1), np.ones(pte_numpy.shape[0]))
    np.testing.assert_allclose(ret_np2['p_star'].sum(axis=1), np.ones(pte_numpy.shape[0]))
    np.testing.assert_allclose(ret_df2['p_star'].sum(axis=1), np.ones(pte_numpy.shape[0]))

    # Budget determines price parameter sizes
    assert np.all(np.abs(ret_np1['beta_star']) > np.abs(ret_df1['beta_star']))

    # Labels and save
    ret_np = compute_probs(wtp_numpy, pte_numpy, subject_ids = subject_numpy, treatment_labels = ['0', '1'])
    ret_df = compute_probs(wtp_numpy, pte_numpy, subject_ids = subject_df, treatment_labels = ['t0', 't1'],
                            save_path = "test_data/test_input_types.csv")

    assert np.array_equal(ret_np['p_star'].index, subject_numpy)
    assert np.array_equal(ret_np['p_star'].index, subject_df)
    assert np.array_equal(ret_df['p_star'].index, subject_numpy)
    assert np.array_equal(ret_df['p_star'].index, subject_df)
    assert np.array_equal(ret_np['p_star'].columns, ['0','1'])
    assert np.array_equal(ret_df['p_star'].columns, ['t0','t1'])

    ret_df_load = pd.read_csv("test_data/test_input_types.csv", index_col = 0)

    pd.testing.assert_frame_equal(ret_df['p_star'], ret_df_load)
    assert np.array_equal(ret_df['p_star'].index, ret_df_load.index)

def test_n_treatments(data_n_treatments):
    # Budgets, thresholds, labels, save path
    pte, wtp = data_n_treatments
    n_treatments = pte.shape[1]
    n_subjects = pte.shape[0]
    base = n_subjects // n_treatments
    mod = n_subjects % n_treatments
    capacity = np.array([base + (1 if i < mod else 0) for i in range(n_treatments)])
    rct_prob = capacity/n_subjects
    subject_budgets = np.random.choice(range(1,100), n_subjects)
    subject_ids = np.random.choice(n_subjects, n_subjects, replace = False)

    with pytest.raises(ValueError):
        ret = compute_probs(wtp, pte, probability_bound = np.min(rct_prob) + 1e-2)

    ret = compute_probs(wtp, pte, probability_bound = np.min(rct_prob)/2, error_threshold = 0.05, iterations_threshold = 10,
                        subject_budgets = subject_budgets, save_path = "test_data/test_n_treatments.csv",
                        subject_ids = subject_ids, treatment_labels = [f"col{i}" for i in range(n_treatments)])

    ret_df = ret['p_star']
    ret_df_load = pd.read_csv("test_data/test_n_treatments.csv", index_col=0)

    pd.testing.assert_frame_equal(ret_df, ret_df_load)
    np.testing.assert_allclose(ret_df.sum(axis=1), np.ones(pte.shape[0]))
    assert np.array_equal(ret_df.index, subject_ids)
    assert np.array_equal(ret_df.columns, ret_df_load.columns)
    assert ret['error'] < 0.05
