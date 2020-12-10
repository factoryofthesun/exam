"""Compute probabilities and assign treatment"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Set, Union, Sequence, Optional
from exam._utils import clear_market, norm_round
import warnings
warnings.filterwarnings("ignore")

def compute_probs(wtp: Sequence, predicted_effects: Sequence, capacity: Sequence = None, probability_bound: float = 0,
                  error_threshold: float = 0.01, iterations_threshold: int = 20, budget: int = 100,
                  subject_budgets: Sequence = None, round: int = None, save_path: str = None,
                  treatment_labels: Sequence = None, subject_ids: Sequence = None, max_search: int = 100):
    """Given the market parameters, compute the equilibrium and return treatment assignment probabilities.

    Parameters
    ----------
    wtp: array-like
        2-D array of participant willingness-to-pay for each treatment
    predicted_effects: array-like
        2-D array of participant estimated treatment effects
    capacity: array-like, default: None
        Capacity constraints for treatments.
    probability_bound: float, default: 0
        The minimum bound for any treatment assignment probability. All computed probabilities will be
        constrained within [:math:`\\epsilon`, :math:`1 - \\epsilon`] where :math:`\\epsilon` is the probability bound.
    error_threshold: float, default: 0.01
        The minimum market clearing error to satisfy the market clearing algorithm.
    max_iterations: int, default: 10
        The maximum number of iterations to run the market clearing algorithm before resetting the search.
    budget: int, default: 100
        The budget to assign to each participant.
    subject_budgets: list-like, default: None
        An array of budgets if distributing budgets unequally among subjects.
    round: int, default: None
        Decimal places to round output probabilities, to ensure coarseness. Must be > 0.
    save_path: str, default: None
        String path to save output probabilities to CSV.
    treatment_labels: array-like, default: None
        Treatment labels for recording assignment. Default is 0,1,...
    subject_ids: array-like, default: None
        Ids to set as the index of the output dataframe. Default is the default pandas index.

    Returns
    -----------
    dict[str, any]
        Returns dict containing calculated treament probabilities for each participant ('p_star'), the optimized treatment-effect coefficient :math:`\\alpha*` ('alpha_star'), the vector of optimized treatment coefficients (:math:`\\beta_t*`) ('beta_star'), and the minimized market clearing error :math:`\\text{error}_{min}` ('error').

    Notes
    ------
    The market equilibrium is computed iteratively by finding optimal \\alpha and \\beta_t such that prices :math:`\\pi_{ti} = \\alpha e_{ti} + \\beta_t` clear the market within a certain threshold. The iteration works by updating :math:`\\beta_t = \\beta_t(1 + \\lambda_t\\min(1, \\text{E}_t/\\text{C}_t))` where  :math:`\\text{E}_t` is excess demand for treatment t and :math:`\\text{C}_t` is the experimental capacity for treatment t. :math:`\\lambda_t` is a scaling factor computed as the absolute value of the mean of the ratio between the :math:`\\alpha` and :math:`\\beta` contributions to the price equation. In mathematical terms, :math:`\\lambda_t = |\\frac{1}{N}\\sum_{i = 1}^{N} \\alpha e_{ti}/\\beta_t |`.

    """
    wtp = np.array(wtp)
    predicted_effects = np.array(predicted_effects)
    n_subjects = wtp.shape[0]
    n_treatments = wtp.shape[1]

    # Set subject budgets
    if subject_budgets is None:
        subject_budgets = np.array([budget] * n_subjects)
        budget_setting = "constant"
    else:
        subject_budgets = np.array(subject_budgets)
        budget_setting = "heterogeneous"

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
    if probability_bound > np.min(rct_treatment_probabilities) or probability_bound < 0:
        raise ValueError(f"compute_probs: probability bound must not be larger than the minimum RCT treatment probability {np.min(rct_treatment_probabilities)}")
    # total capacity >= n_subjects
    if np.sum(capacity) < n_subjects:
        raise ValueError(f"compute_probs: total capacity must be greater than or equal to total # subjects {n_subjects}")
    # budget must be positive
    if budget <= 0:
        raise ValueError(f"compute_probs: budget must be positive")

    # TODO: Figure out optimal beta scaling factor
    beta_scaling_factor = budget/100
    import re
    param_string = f"""# treatments: {n_treatments}\n# subjects: {n_subjects}\ncapacity: {capacity}\nepsilon-bound: {probability_bound}\nerror clearing threshold: {error_threshold}\niterations threshold: {iterations_threshold}\nbudget type: {budget_setting}\n"""
    print("compute_probs: Parameters")
    print("-"*50)
    print(param_string)

    res = clear_market_new(wtp, predicted_effects, capacity, rct_treatment_probabilities, subject_budgets,
                        error_threshold, iterations_threshold, probability_bound, max_search)

    if res is None:
        return None

    p_star = res['p_star']

    # Round probabilities if necessary
    if round is not None:
        if round > 0:
            print(f"Rounding probabilities to {round} decimals...")
            p_star = np.apply_along_axis(lambda x: norm_round(x, round), 1, p_star)

    #np.testing.assert_allclose(np.sum(p_star, axis = 1), np.ones(p_star.shape[0]), rtol=1e-5)

    # Cast numpy output to pandas
    res['p_star'] = pd.DataFrame(p_star, index = subject_ids, columns = treatment_labels)

    # Save to csv
    if save_path is not None:
        res['p_star'].to_csv(save_path)
        print(f"Treatment probabilities saved to: {save_path}")

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
    # Normalize probabilities to remove precision nonsense
    probs = np.array(probs)/np.sum(probs, axis = 1)[:,np.newaxis]
    if treatment_labels is None:
        treatment_labels = range(probs.shape[1])

    if seed is not None:
        np.random.seed(seed)

    # Assign treatments
    assignments = np.apply_along_axis(lambda p: np.random.choice(treatment_labels, p = p), 1, probs)
    assignment_df = pd.Series(assignments, index = subject_ids, name = "assignment")

    if save_path is not None:
        assignment_df.to_csv(save_path, columns = ["assignment"])
        print(f"Treatment assignments saved to: {save_path}")

    return assignment_df

def _get_args():
    import argparse
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV containing WTP and PTE, in that order. If --index flag toggled, the first column will be assumed to be the index containing subject ids. The columns will be taken to be treatment labels, unless otherwise specified in --labels.")
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save CSV file with treatment probabilities."
        )
    parser.add_argument(
        "--assign_output",
        type=str,
        required=False,
        help="Path to save CSV file with treatment assignments."
        )
    parser.add_argument(
        "--capacity",
        nargs="*",
        type=int,
        required=False,
        help="List of treatment capacities (integer).")
    parser.add_argument(
        "--pbound",
        type=float,
        default=0,
        required=False,
        help="The minimum bound for any treatment assignment probability. All computed probabilities will be constrained within [e, 1-e] where e is the probability bound."
        )
    parser.add_argument(
        "--error",
        type=float,
        default=0.01,
        required=False,
        help="Minimum market clearing error."
        )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        required=False,
        help="Maximum number of algorithm iterations before a new search is conducted.")
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        required=False,
        help="Common budget to distribute to every subject."
        )
    parser.add_argument(
        "--subject_budgets",
        type=str,
        required=False,
        help="Path to CSV file containing subject-specific budgets."
        )
    parser.add_argument(
        "--round",
        type=int,
        required=False,
        help="Decimal places to round optimal probabilities. Use to ensure coarseness of propensity vectors."
        )
    parser.add_argument(
        "--labels",
        nargs="*",
        type=str,
        required=False,
        help="List of treatment labels."
        )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Flag to indicate data has index column to use "
        )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = _get_args()
    data = pd.read_csv(args.data, index_col=0)
    # N Treatments is n columns/2
    if len(data.columns) % 2 != 0:
        raise ValueError("compute_probs: WTP and PTE have uneven treatment columns.")
    wtp = data.iloc[:,:int(len(data.columns)/2)]
    pte = data.iloc[:,int(len(data.columns)/2):]
    subject_ids = data.index
    if args.labels is None:
        treatment_labels = data.columns[:int(len(data.columns)/2)]
    else:
        treatment_labels = args.labels

    prob_res = compute_probs(wtp, pte, args.capacity, args.pbound, args.error, args.iterations,
                             args.budget, args.subject_budgets, args.round, args.output,
                             treatment_labels, subject_ids)

    if args.assign_output is not None:
        assignments = assign(prob_res['p_star'], treatment_labels, subject_ids, args.assign_output)
