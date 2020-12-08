from scipy.optimize import linprog
import numpy as np
import pandas as pd
from scipy import stats
import scipy.special as sc
sc.seterr(all = "ignore") # TODO: suppress ALL warnings

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
def get_demand_matrix(wtp_matrix, pte_matrix, price_matrix,  subject_budgets, rct_treatment_probabilities, epsilon):
    # Create empty matrix for filling
    demand_matrix = []

    # Solve problem for each individual
    for i in range(wtp_matrix.shape[0]):
        # Constraints:
        # 1. <p*(i), pi(i)> <= b(i) for every subject i
        # 2. sum of all p*(t) = 1 for every subject i

        # When linprog can't solve coefficients become NA
        if np.isnan(price_matrix[i]).any():
            return None

        result = linprog(c=-wtp_matrix[i,np.newaxis],
                         A_ub = price_matrix[i,np.newaxis],
                         b_ub = [subject_budgets[i]],
                         A_eq = np.ones((1, wtp_matrix.shape[1])),
                         b_eq = 1,
                         bounds = (0, 1))

        # TODO: REMOVE THIS WHEN RELEASING PACKAGE
        if result.status != 0:
            print(result.message)
        demand_matrix.append(result.x)
    demand_matrix = np.stack(demand_matrix)
    # We only need to apply epsilon bounds if market-clearing probabilities fall outside
    min_prob = np.min(demand_matrix)
    max_prob = np.max(demand_matrix)

    if min_prob >= epsilon and max_prob <= 1-epsilon:
        return demand_matrix
    else:
        q_min = max((epsilon - min_prob)/(rct_treatment_probabilities - min_prob))
        q_max = max((1 - epsilon - max_prob)/(rct_treatment_probabilities - max_prob))
        q = max(q_min, q_max)
        demand_matrix = (1-q) * demand_matrix + q * rct_treatment_probabilities
        return demand_matrix

# Clearing error in market = sqrt(sum of excess_demand(t)^2 for every treatment t)
def get_clearing_error(excess_demand, capacity):
    # If demand is satisfied everywhere and total capacity > number of subjects, no clearing error
    if (excess_demand <= 0).all():
        print("get_clearing_error: Market clear, no clearing error!")
        return 0
    else:
        clearing_error = np.sqrt(np.sum(excess_demand**2))
        clearing_error = clearing_error / np.sum(capacity)
        print(f"get_clearing_error: Clearing error: {clearing_error}")
        return clearing_error

# def get_beta_new(beta, excess_demand, beta_scaling_factor):
#     beta_new = beta + excess_demand * beta_scaling_factor
#     return beta_new
#
# # Find market clearing price vector. Iteratively adjust beta to minimize market clearing error.
# def clear_market(wtp_matrix: np.ndarray, pte_matrix: np.ndarray, capacity: np.ndarray,
#                 rct_treatment_probabilities: np.ndarray, subject_budgets: np.ndarray, clearing_error_threshold: int,
#                 iterations_threshold: int, epsilon: float, beta_scaling_factor: float, max_search: int):
#     # Initialize market prices and demand
#     init_budget = subject_budgets[0]
#     alpha = init_alpha(init_budget, wtp_matrix.shape[1])
#     beta = init_beta(init_budget, wtp_matrix.shape[1])
#     price_matrix = get_price_matrix(alpha, beta, pte_matrix)
#     demand_matrix = get_demand_matrix(wtp_matrix, pte_matrix, price_matrix, subject_budgets,
#                                     rct_treatment_probabilities, epsilon)
#
#     iterations = 0
#     tot_search = 1
#     alpha_star = alpha
#     beta_star = beta
#
#     # Linprog will return NA if unable to solve
#     if np.isnan(demand_matrix).any():
#         minimum_clearing_error = 100
#         iterations = iterations_threshold + 1
#     else:
#         treatment_demand = np.sum(demand_matrix, axis = 0)
#         excess_demand = treatment_demand - capacity
#         minimum_clearing_error = get_clearing_error(excess_demand, capacity)
#
#     # Set new prices to clear market
#     while True:
#         if iterations > iterations_threshold:
#             tot_search += 1
#             if tot_search > max_search:
#                 print("Max search limit reached. Market clearing search was unable to converge.")
#                 return None
#             # New search start
#             alpha = init_alpha(init_budget, wtp_matrix.shape[1])
#             beta = init_beta(init_budget, wtp_matrix.shape[1])
#             iterations = 0
#             print("new search start")
#         else:
#             # Continue down current search
#             beta = get_beta_new(beta, excess_demand, beta_scaling_factor)
#
#         price_matrix = get_price_matrix(alpha, beta, pte_matrix)
#         demand_matrix = get_demand_matrix(wtp_matrix, pte_matrix, price_matrix, subject_budgets,
#                                         rct_treatment_probabilities, epsilon)
#
#         # Linprog will return NA if unable to solve
#         if np.isnan(demand_matrix).any():
#             clearing_error = 100
#             iterations = iterations_threshold + 1
#         else:
#             treatment_demand = np.sum(demand_matrix, axis = 0)
#             excess_demand = treatment_demand - capacity
#             clearing_error = get_clearing_error(excess_demand, capacity)
#
#         # Store parameter values for minimum clearing error
#         if clearing_error < minimum_clearing_error:
#             minimum_clearing_error = clearing_error
#             alpha_star = alpha
#             beta_star = beta
#         # cleared the market!
#         if minimum_clearing_error < clearing_error_threshold:
#             break
#         iterations += 1
#
#     print(f"Minimum clearing error: {minimum_clearing_error}")
#     print(f"Alpha_star: {alpha_star}")
#     print(f"Beta star: {beta_star}")
#
#     results_dict = {"p_star": demand_matrix, "error": minimum_clearing_error, "alpha_star": alpha_star, "beta_star": beta_star}
#     return results_dict

def get_beta_new(beta, excess_demand, l, capacity):
    beta_new = beta * (1 + l * np.amin(np.column_stack([np.ones(len(excess_demand)), excess_demand/capacity]), axis=1))
    return beta_new

# Find market clearing price vector. Iteratively adjust beta to minimize market clearing error.
def clear_market(wtp_matrix: np.ndarray, pte_matrix: np.ndarray, capacity: np.ndarray,
                rct_treatment_probabilities: np.ndarray, subject_budgets: np.ndarray, clearing_error_threshold: int,
                iterations_threshold: int, epsilon: float, max_search: int):
    # Initialize market prices and demand
    init_budget = subject_budgets[0]
    alpha = init_alpha(init_budget, wtp_matrix.shape[1])
    beta = init_beta(init_budget, wtp_matrix.shape[1])
    price_matrix = get_price_matrix(alpha, beta, pte_matrix)
    demand_matrix = get_demand_matrix(wtp_matrix, pte_matrix, price_matrix, subject_budgets,
                                    rct_treatment_probabilities, epsilon)

    iterations = 0
    tot_search = 1
    alpha_star = alpha
    beta_star = beta

    # Linprog will return NA if unable to solve
    if np.isnan(demand_matrix).any():
        minimum_clearing_error = 100
        iterations = iterations_threshold + 1
    else:
        treatment_demand = np.sum(demand_matrix, axis = 0)
        excess_demand = treatment_demand - capacity
        minimum_clearing_error = get_clearing_error(excess_demand, capacity)

    # Set new prices to clear market
    while True:
        if iterations > iterations_threshold:
            tot_search += 1
            if tot_search > max_search:
                print("Max search limit reached. Market clearing search was unable to converge.")
                return None
            # New search start
            alpha = init_alpha(init_budget, wtp_matrix.shape[1])
            beta = init_beta(init_budget, wtp_matrix.shape[1])
            iterations = 0
            print("new search start")
        else:
            # Continue down current search
            l = np.abs(np.mean(alpha * wtp_matrix/beta, axis = 0))
            beta = get_beta_new(beta, excess_demand, l, capacity)

        price_matrix = get_price_matrix(alpha, beta, pte_matrix)
        demand_matrix = get_demand_matrix(wtp_matrix, pte_matrix, price_matrix, subject_budgets,
                                        rct_treatment_probabilities, epsilon)

        # Linprog will return NA if unable to solve
        if np.isnan(demand_matrix).any():
            clearing_error = 100
            iterations = iterations_threshold + 1
        else:
            treatment_demand = np.sum(demand_matrix, axis = 0)
            excess_demand = treatment_demand - capacity
            clearing_error = get_clearing_error(excess_demand, capacity)

        # Store parameter values for minimum clearing error
        if clearing_error < minimum_clearing_error:
            minimum_clearing_error = clearing_error
            alpha_star = alpha
            beta_star = beta
        # cleared the market!
        if minimum_clearing_error < clearing_error_threshold:
            break
        iterations += 1

    print(f"Minimum clearing error: {minimum_clearing_error}")
    print(f"Alpha_star: {alpha_star}")
    print(f"Beta star: {beta_star}")

    results_dict = {"p_star": demand_matrix, "error": minimum_clearing_error, "alpha_star": alpha_star, "beta_star": beta_star}
    return results_dict

def norm_round(p, round):
    p_res = p % 10**-round
    p = np.round(p, round)
    if np.sum(p) > 1:
        adj_ind = p_res.argmin()
        p[adj_ind] = p[adj_ind] - 10**-round
    elif np.sum(p) < 1:
        adj_ind = p_res.argmax()
        p[adj_ind] = p[adj_ind] + 10**-round
    return p
