import numpy as np
import pandas as pd
from typing import Tuple, Dict, Set, Union, Sequence, Optional
from exam._utils import clear_market

def compute_probs(wtp: Sequence, predicted_effects: Sequence, capacity: Sequence = None, probability_bound: float = 0,
                  error_threshold: float = 0.01, iterations_threshold: int = 20, budget: int = 100,
                  subject_budgets: Sequence = None, save_path: str = None, treatment_labels: Sequence = None,
                  subject_ids: Sequence = None):
    """Given the market parameters, compute the equilibrium and return treatment assignment probabilities.

    Parameters
    ----------
    wtp: array-like
        2-D array of participant willingness-to-pay for each treatment
    predicted_effects: array-like
        2-D array of participant estimated treatment effects
    capacity: array-like, default: None
        Capacity constraints for treatments.
    probability_bound: float, default = 0
        The minimum bound for any treatment assignment probability. All computed probabilities will be
        constrained within [:math:`\\epsilon`, :math:`1 - \\epsilon`] where :math:`\\epsilon` is the probability bound.
    error_threshold: float, default = 0.01
        The minimum market clearing error to satisfy the market clearing algorithm.
    max_iterations: int, default = 10
        The maximum number of iterations to run the market clearing algorithm before resetting the search.
    budget: int, default = 100
        The budget to assign to each participant.
    subject_budgets: list-like, default = None
        An array of budgets if distributing budgets unequally among subjects.
    save_path: str, default = None
        String path to save output probabilities to CSV.
    treatment_labels: array-like, default: None
        Treatment labels for recording assignment. Default is 0,1,...
    subject_ids: array-like, default: None
        Ids to set as the index of the output dataframe. Default is the default pandas index.

    Returns
    -----------
    dict[str, any]
        Returns dict containing calculated treament probabilities for each participant ('p_star'), the optimized treatment-effect coefficient :math:`\\alpha*` ('alpha_star'), the vector of optimized treatment coefficients (:math:`\\beta_t*`) ('beta_star'), and the minimized market clearing error :math:`\\text{error}_{min}` ('error').

    """
    wtp = np.array(wtp)
    predicted_effects = np.array(predicted_effects)
    n_subjects = wtp.shape[0]
    n_treatments = wtp.shape[1]

    # Set subject budgets
    if subject_budgets is None:
        subject_budgets = np.array([budget] * n_subjects)
    else:
        subject_budgets = np.array(subject_budgets)

    # Set treatment capacities
    if capacity is None:
        base = n_subjects // n_treatments
        mod = n_subjects % n_treatments
        capacity = np.array([base + (1 if i < mod else 0) for i in range(n_treatments)])
    else:
        capacity = np.array(capacity)
    rct_treatment_probabilities = capacity/n_subjects

    ## Input checking
    # probability_bound <= min(p_RCT)
    print(f"Prob_bound: {probability_bound}, Min RCT prob: {np.min(rct_treatment_probabilities)}")
    if probability_bound > np.min(rct_treatment_probabilities) or probability_bound < 0:
        raise ValueError(f"compute_probs: probability bound must not be larger than the minimum RCT treatment probability {np.min(rct_treatment_probabilities)}")
    # total capacity >= n_subjects
    if np.sum(capacity) < n_subjects:
        raise ValueError(f"compute_probs: total capacity must be greater than or equal to total # subjects {n_subjects}")
    # budget must be positive
    if budget <= 0:
        raise ValueError(f"compute_probs: budget must be positive")

    # TODO: Figure out optimal beta scaling factor
    beta_scaling_factor = budget/50
    res = clear_market(wtp, predicted_effects, capacity, rct_treatment_probabilities, subject_budgets,
                        error_threshold, iterations_threshold, probability_bound, beta_scaling_factor)
    p_star = res['p_star']

    # Cast numpy output to pandas
    res['p_star'] = pd.DataFrame(p_star, index = subject_ids, columns = treatment_labels)

    # Save to csv
    if save_path is not None:
        res['p_star'].to_csv(save_path)

    return res

def assign(probs: Sequence, treatment_labels: Sequence = None, subject_ids: Sequence = None, save_path: str = None,
           seed: int = None):
    """Assign treatments given set of probabilities

    Parameters
    ----------
    probs: array-like
        Array of treatment assignment probabilites for each suvhect
    treatment_labels: array-like, default: None
        Treatment labels for recording assignment. Default is 0,1,...
    subject_ids: array-like, default: None
        Ids to set as the index of the output dataframe. Default is the default pandas index.
    save_path: str, default: None
        String path to save assignments file to CSV.
    seed: int, default: None
        Numpy seed

    Returns
    -----------
    pd.DataFrame
        Dataframe of assigned treatments

    """
    probs = np.array(probs)
    if treatment_labels is None:
        treatment_labels = range(probs.shape[1])

    if seed is not None:
        np.random.seed(seed)

    # Assign treatments
    assignments = np.apply_along_axis(lambda p: np.random.choice(treatment_labels, p = p), 1, probs)
    assignment_df = pd.DataFrame(assignments, index = subject_ids, columns = ['assignment'])

    if save_path is not None:
        assignment_df.to_csv(save_path)

    return assignment_df

if __name__ == "__main__":
    # RUN COMMAND LINE ARGUMENTS
    pass
