import pandas as pd
import numpy as np
from typing import Tuple, Dict, Set, Union, Sequence, Optional
import statsmodels.api as sm
from scipy import stats

def estimate_effects(Y: Sequence, D: Sequence, probs: Sequence, X: Sequence = None, control: str = None,
                     method: str = "matched", save_path: str = None, return_model: bool = True, verbose=True):
    """Estimate treatment effects using either single regression with propensity controls or propensity-matched subpopulation regressions.

    Parameters
    ----------
    Y: array-like
        Array of outcome data
    D: array-like
        Array of treatment assignment data
    probs: array-like
        2D array of treatment probabilities, where the first column is the control probability.
    X: array-like, default: None
        Control variables to include in regression
    control: str, default: None
        Name of control treatment as given in `D`. Defaults to first column from pandas `get_dummies`.
    method: str, default: "matched"
        Estimation method. Supports "matched" for subpopulation regressions or "single" for single regression with propensity controls.
    save_path: str, default: None
        Path to save output
    return_model: bool, default: True
        Whether to return fitted models as part of output
    verbose: bool, default: True
        Whether to print estimation summary

    Returns
    -----------
    dict[str, any]

    """
    # Check inputs
    supported_methods = ["matched", "single"]
    if method not in supported_methods:
        raise ValueError(f"estimate_effects: `method` not found among support methods {supported_methods}")
    if not isinstance(D, pd.Series) and not isinstance(D, pd.DataFrame):
        D = pd.Series(D)
    if control is None:
        D_dums = pd.get_dummies(D.astype(str), drop_first = True) # Need control to be constant for ATE estimation
    else:
        D_dums = pd.get_dummies(D.astype(str)).drop(control, axis = 1) # Need control to be constant for ATE estimation
    # Need at least 2 treatments
    if D_dums.empty:
        raise ValueError(f"estimate_effects: `D` must contain at least 2 treatments (including control)")

    # Set up dataframe for estimation
    df = pd.DataFrame({"Y":Y})
    probs = np.array(probs)
    probs = probs[:,1:] # First probability column is control
    prob_names = [f"prob_{c}" for c in D_dums.columns]
    probs_df = pd.DataFrame(probs)
    probs_df.columns = prob_names
    if X is not None:
        df = pd.concat([df, D_dums, probs_df, pd.DataFrame(X)], axis=1)
        df = sm.add_constant(df, has_constant = 'add')
    else:
        df = pd.concat([df, D_dums, probs_df], axis=1)
        df = sm.add_constant(df, has_constant = 'add')

    # Estimation
    treatment_names = D_dums.columns
    delta_df = probs_df.value_counts() # Delta weights
    delta_df = delta_df/delta_df.sum()
    delta_df.name = "delta_weights"
    if method == "single":
        if verbose == True:
            print("-"*75)
            print("ATE estimation method: single regression with propensity controls")
            print("-"*75)
        final_out = dict()
        model = sm.OLS(df.Y, df.drop("Y", axis=1))
        ols_res = model.fit()

        # Dict of treatment coefficients
        coefs = ols_res.params
        treatment_coefs = dict(coefs.loc[treatment_names,])

        # Var-cov matrix
        varcov = ols_res.cov_params()

        # Lambda-weights on CATE (for each treatment)
        p_vectors = np.array([list(p) for p in delta_df.index])
        lambda_arr = np.array(delta_df)[:,np.newaxis] * (1 - p_vectors) * p_vectors
        lambda_df = pd.DataFrame(lambda_arr, index = delta_df.index, columns = treatment_names)

        final_out['coefs'] = treatment_coefs
        final_out['p'] = ols_res.pvalues.loc[treatment_names,]
        final_out['varcov'] = varcov
        final_out['lambda_weights'] = lambda_df
        final_out['delta_weights'] = delta_df
        if return_model == True:
            final_out['model'] = ols_res

        if verbose == True:
            print(ols_res.summary())

        if save_path is not None:
            #TODO: MORE COMPREHENSIVE SAVE OUTPUT FOR COMMAND LINE INTERFACE
            out_str = ols_res.summary().as_csv()
            with open("test_data/test_single.csv", 'w') as f:
                f.write(out_str)

        return final_out

    if method == "matched":
        if verbose == True:
            print("-"*75)
            print("ATE estimation method: propensity subpopulation regressions")
            print("-"*75)
        final_out = dict()
        model_dict = dict()
        cate_dict = dict()
        variances = []
        # Separate propensity subpopulation regressions
        df = df.set_index(prob_names)
        ate = pd.Series([0]*len(treatment_names), index = treatment_names, name = 'coef', dtype="float64")
        for p in delta_df.index:
            # Subset propensity sample
            delta = delta_df.loc[p,]
            tmp_df = df.loc[p,]

            # Fit model
            model = sm.OLS(tmp_df.Y, tmp_df.drop("Y", axis=1))
            ols_res = model.fit()

            # Get subpopulation model outputs and compute ATE
            cate_dict[p] = ols_res.params.loc[treatment_names,]
            model_dict[p] = ols_res
            ate = ate + delta * ols_res.params.loc[treatment_names,]

            varcov = ols_res.cov_params()
            variances.append(varcov)

        # Assume each subpopulation coefficient is independent random variable, then variance of weighted average
        # coefficient is sum of square weighted variances
        # TODO: DIFFERENT VARIANCE TYPES???
        delta_np = np.array(delta_df)**2
        ate_varcov = np.sum(delta_np[:,np.newaxis,np.newaxis] * np.array(variances), axis=0)

        # Treatment estimator SEs are sqrt of diagonal
        ate_se = np.sqrt(np.diagonal(ate_varcov[:len(ate),:len(ate)]))

        # Compute p-values
        t_stat = np.array(ate)/ate_se
        p = (1 - stats.t.cdf(abs(t_stat), len(df) - len(df.columns))) * 2 # Two-tailed t-test
        p = pd.Series(p, index = treatment_names, name='p-value')

        final_out['coefs'] = dict(ate)
        final_out['p'] = p
        final_out['varcov'] = ate_varcov
        final_out['subpop_coefs'] = cate_dict
        final_out['delta_weights'] = delta_df
        if return_model is True:
            final_out['model'] = model_dict

        if verbose == True:
            print(f"Estimated treatment effects: {ate}")
            print(f"P-values: {p}")

        if save_path is not None:
            #TODO: MORE COMPREHENSIVE SAVE OUTPUT FOR COMMAND LINE INTERFACE
            pass

        return final_out

if __name__ == "__main__":
    # RUN COMMAND LINE ARGUMENTS
    pass
