from scipy.optimize import linprog
from math import sqrt
from collections import defaultdict
import numpy as np
import random
import copy
import math
import timeit
import pandas as pd
from decimal import Decimal
from scipy import stats
import time

def init_alpha(budget, n_treatments):
    alpha = np.random.uniform(-budget, 0, size = None)
    return alpha

def init_beta(budget, n_treatments):
    beta = np.random.uniform(-budget, budget, size = n_treatments)
    return beta

# Output dimensions: (n_subjects, n_treatments)
def get_price_matrix(alpha, beta, pte_matrix):
    price_matrix = alpha * pte_matrix + beta
    return price_matrix

# Solve linear programming problem to get utility-maximizing demands. Dimensions: (n_subjects, n_treatments)
def get_demand_matrix(wtp_matrix, pte_matrix, price_matrix,  budget_matrix, rct_treatment_probabilities, epsilon):
    x0_bounds = (0,1)
    x1_bounds = (0,1)

    # Create empty matrix for filling
    demand_matrix = np.ndarray((wtp_matrix.shape[0],wtp_matrix.shape[1]), float)

    # Solve problem for each individual
    for i in range(wtp_matrix.shape[0]):
        # Constraints:
        # 1. <p*(i), pi(i)> <= b(i) for every subject i
        # 2. sum of all p*(t) = 1 for every subject i

        # When linprog can't solve coefficients become NA
        if any(np.isnan(price_matrix[i])):
            return None

        result = linprog(c=-wtp_matrix[i],
                         A_ub = price_matrix[i],
                         b_ub = budget_matrix[i],
                         A_eq = np.ones(wtp_matrix.shape[1]),
                         b_eq = 1,
                         bounds = (x0_bounds, x1_bounds),
                         options = {"presolve": True})

        # TODO: REMOVE THIS WHEN RELEASING PACKAGE
        if result.status != 0:
            print(result.message)

        demand_matrix[i] = result.x

    # Apply epsilon bounds
    # TODO: migrate this to the top-level function
    # Check if epsilon bounds are reasonable (eps < min(p_RCT))
    if epsilon > min(rct_treatment_probabilities) or epsilon < 0:
        return ValueError("Epsilon is bad.")

    # We only need to apply bounds if market-clearing probabilities fall outside
    min_prob = min(demand_matrix)
    max_prob = max(demand_matrix)

    if min_prob >= epsilon and max_prob <= 1-epsilon:
        return demand_matrix
    else:
        q_min = max((epsilon - min_prob)/(rct_treatment_probabilities - min_prob))
        q_max = max((1 - epsilon - max_prob)/(rct_treatment_probabilities - max_prob))
        q = max(q_min, q_max)
        demand_matrix = (1-q) * demand_matrix + q * rct_treatment_probabilities
        return demand_matrix

# Treatment_demand(t) = sum of demand(t) across all i. Dimensions 1 * num_treatments
def get_treatment_demand_matrix(demand_matrix):
    treatment_demand = np.zeros(num_treatments)
    for subject in range(demand_matrix.shape[0]):
        for treatment in range(demand_matrix.shape[1]):
            treatment_demand[treatment] += demand_matrix[subject, treatment]
    return treatment_demand

# Excess_demand(t) = treatment_demand(t) - capacity(t). Dimensions 1 * num_treatments
def get_excess_demand_matrix(treatment_demand, capacity):
    excess_demand = treatment_demand - capacity
    return excess_demand

# Clearing error in market = sqrt(sum of excess_demand(t)^2 for every treatment t)
def get_clearing_error(excess_demand):
    # If demand is satisfied everywhere and total capacity > number of subjects, no clearing error
    if all(excess <= 0 for excess in excess_demand):
        print ("get_clearing_error: Market clear, no clearing error!")
        return 0
    else:
        clearing_error = sqrt(sum([excess**2 for excess in excess_demand]))
        clearing_error = clearing_error / sum(capacity_matrix)
        print ("get_clearing_error: Clearing error:"), clearing_error
        return clearing_error

def get_beta_new(beta, excess_demand):
    beta_new = beta + excess_demand_matrix * beta_scaling_factor
    return beta_new

# Find market clearing price vector. The objective is to change alpha and beta values so that we reduce clearing error
def clear_market():

    # Initialize market prices and demand
    alpha = init_alpha()
    beta = init_beta()
    price_matrix = get_price_matrix(alpha, beta)
    demand_matrix = get_demand_matrix(price_matrix)
    excess_demand_matrix = get_excess_demand_matrix(get_treatment_demand_matrix(demand_matrix))
    clearing_error = get_clearing_error(excess_demand_matrix)

    # clearing error is percentage of total capacity so we want the market to clear at 1%
    clearing_error_threshold = 0.01
    threshold_iterations = 20
    iterations = 0
    minimum_clearing_error = clearing_error
    alpha_star = 0
    beta_star = 0

    # Set new prices to clear market
    while True:
        if iterations > threshold_iterations:
            # new search start
            alpha = init_alpha()
            beta = init_beta()
            iterations = 0
            print ("new search start")
        else:
            # continue down current search
            beta = get_beta_new(beta, excess_demand_matrix)

        price_matrix = get_price_matrix(alpha, beta)
        demand_matrix = get_demand_matrix(price_matrix)
        excess_demand_matrix = get_excess_demand_matrix(get_treatment_demand_matrix(demand_matrix))
        clearing_error = get_clearing_error(excess_demand_matrix)

        # Store parameter values for minimum clearing error
        if clearing_error < minimum_clearing_error:
            minimum_clearing_error = clearing_error
            alpha_star = alpha.copy()
            beta_star = beta.copy()
        # cleared the market!
        if minimum_clearing_error < clearing_error_threshold:
            break
        iterations += 1

    print ("Minimum clearing error:"), minimum_clearing_error
    print ("Alpha_star:"), alpha_star
    print ("Beta star:"), beta_star
    return (minimum_clearing_error, alpha_star, beta_star)
