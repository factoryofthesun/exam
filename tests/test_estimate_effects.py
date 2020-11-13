# Test random assignment
import sys
import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from pathlib import Path
import random

from exam import estimate_effects

@pytest.fixture
def rct_data():
    D = np.random.choice([0,1], 1000)
    Y = [0 if d == 0 else 10 for d in D]
    probs = np.array([[0.5,0.5] for _ in range(1000)])
    return (Y, D, probs, 10)

@pytest.fixture
def sim_data():
    n_treatments = np.random.choice(range(1,5))
    treat_prob = np.random.choice(np.arange(0.2,0.8,0.05),1000)
    treat_prob = np.column_stack([treat_prob/n_treatments]*n_treatments)
    D = []
    probs = []
    for p in treat_prob:
        full_p = np.append(np.array([1 - np.sum(p)]), p).flatten()
        d = np.random.choice(n_treatments + 1, p = full_p)
        probs.append(full_p)
        D.append(d)
    effects = np.random.choice(range(1,50), n_treatments)
    Y = [0 if d == 0 else effects[d-1] for d in D]
    return (Y, D, probs, effects)

@pytest.fixture
def sim_data_controls():
    n_treatments = np.random.choice(range(1,5))
    treat_prob = np.random.choice(np.arange(0.2,0.8,0.05),1000)
    treat_prob = np.column_stack([treat_prob/n_treatments]*n_treatments)
    D = []
    probs = []
    for p in treat_prob:
        full_p = np.append(np.array([1 - np.sum(p)]), p).flatten()
        d = np.random.choice(n_treatments + 1, p = full_p)
        probs.append(full_p)
        D.append(d)
    effects = np.random.choice(range(1,50), n_treatments)
    controls = np.random.uniform(-5, 5, size = (1000, 3))
    control_effects = np.random.choice(range(-5,5), replace = False, size = 3)
    Y = np.sum(controls*control_effects, axis=1) + np.array([0 if d == 0 else effects[d-1] for d in D])
    return (Y, D, probs, controls, effects, control_effects)

def test_errors(rct_data):
    Y, D, probs, effect = rct_data
    # Method string
    with pytest.raises(ValueError):
        ret = estimate_effects(Y, D, probs, method = "fake")
    # Need at least two treatments
    with pytest.raises(ValueError):
        ret = estimate_effects(Y, [1]*len(Y), probs, method = "fake")

def test_RCT(rct_data):
    Y, D, probs, effect = rct_data
    single_ret = estimate_effects(Y, D, probs, method = "single")
    matched_ret = estimate_effects(Y, D, probs)

    np.testing.assert_allclose(np.array(single_ret['coefs']['1']), np.array(matched_ret['coefs']['1']))
    np.testing.assert_allclose(np.array(matched_ret['coefs']['1']), [10])

def test_single(sim_data):
    Y, D, probs, effects = sim_data
    print(f"n_treatments: {len(effects)}")
    single_ret = estimate_effects(Y, D, probs, control = '0', method = "single",
                    save_path = "test_data/test_estimate_single.csv")
    # No heterogeneous treatment effects conditional on probs
    np.testing.assert_allclose(np.array(list(single_ret['coefs'].values())), effects)

def test_matched(sim_data):
    Y, D, probs, effects = sim_data
    print(f"n_treatments: {len(effects)}")
    matched_ret = estimate_effects(Y, D, probs, control = '0')
    print(matched_ret['varcov'])
    np.testing.assert_allclose(np.array(list(matched_ret['coefs'].values())), effects)

def test_controls(sim_data_controls):
    Y, D, probs, controls, effects, control_effects = sim_data_controls
    print(f"n_treatments: {len(effects)}")
    single_ret = estimate_effects(Y, D, probs, controls, method = "single")
    matched_ret = estimate_effects(Y, D, probs, controls)

    for model in matched_ret['model'].values():
        print(model.params)

    np.testing.assert_allclose(np.array(list(matched_ret['coefs'].values())), np.array(list(single_ret['coefs'].values())), rtol=1e-2)
    np.testing.assert_allclose(np.array(list(matched_ret['coefs'].values())), effects)
