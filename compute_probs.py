import numpy as np
import pandas as pd
from typing import Tuple, Dict, Set, Union, Sequence, Optional
from _utils import *

def compute_probs(wtp: Sequence, predicted_effects: Sequence, capacity: Sequence = None, probabiity_bound: float = 0,
                  error_threshold: float = 0.01, max_iterations: int = 100, budget: int = None,
                  save_path: str = None, treatment_labels: Sequence = None, participant_ids: Sequence = None):
    """Given the market parameters, compute the equilibrium and return treatment assignment probabilities.

    Parameters
    ----------
    wtp: array-like
        Array of participant willingness-to-pay
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
    budget: int, default = None
        The budget to assign to each participant.
    save_path: str, default = None
        String path to save outputs to CSV.
    treatment_labels: array-like, default: None
        Treatment labels for recording assignment. Default is 0,1,...
    participant_ids: array-like, default: None
        Ids to set as the index of the output dataframe. Default is the default pandas index.

    Returns
    -----------
    tuple(pd.DataFrame, float, np.ndarray, float)
        Returns tuple containing calculated treament probabilities for each participant, the optimized treatment-effect
        coefficient :math:`\\alpha*`, the vector of optimized treatment coefficients (:math:`\\beta_t*`),
        and the minimized market clearing error :math:`\\text{error}_{min}`.

    """




    return
def assign(probs: Sequence, treatment_labels: Sequence = None, participant_ids: Sequence = None, save_path: str = None):
    """Assign treatments given set of probabilities

    Parameters
    ----------
    probs: array-like
        Array of treatment assignment probabilites for each participant
    treatment_labels: array-like, default: None
        Treatment labels for recording assignment. Default is 0,1,...
    participant_ids: array-like, default: None
        Ids to set as the index of the output dataframe. Default is the default pandas index.
    save_path: str
        String path to save assignments file to CSV.

    Returns
    -----------
    pd.DataFrame
        Array of assigned treatments

    """

if __name__ == "__main__":
    # RUN COMMAND LINE ARGUMENTS
