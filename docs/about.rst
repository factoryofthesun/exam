About
=====

RCTs and Welfare
--------------------------------

RCTs (randomized control trials) determine the fate of numerous people, giving rise to a long-standing ethical dilemma: “How can a physician committed to doing what he thinks is best for each patient tell a woman with breast cancer that he is choosing her treatment by something like a coin toss? How can he give up the option to make changes in treatment according to the patient’s responses?” (“Patients’ Preferences in RCTs” by physician Marcia Angell.) :cite:`Narita2020` proposes a potential solution in the form of an experimental design dubbed Experiment-as-Market, or EXAM for short. This experimental design has been proven to improve subjects' welfare while producing similar experimental information as a standard RCT. Please refer to :cite:`Narita2020` for a detailed exposition of the design.

EXAM Pipeline
----------------

The exam package is an implementation of the experimental design framework discussed above. Specifically, the user is expected to supply subject WTP (willingness-to-pay for each treatment) and predicted treatment effects,

The package serves every step of a randomized experiment framework:

- **Computing treatment probabilities**: The key insight of EXAM is in modeling treatment assignment as a market in which subjects are priced according to their preference for treatment (WTP) and predicted treatment outcomes. The treatment probabilities that clear such a market are welfare-optimal and Pareto efficient. The package's `compute_probs` function computes these probabilities.
- **Treatment assignment**: This package provides a simple assignment function which randomly assigns subjects to treatments given the treatment probabilities computed above.
- **Treatment effect estimation**: This package provides functions to estimate causal treatment effects following treatment that make use of the OLS estimator from the `statsmodels package <https://www.statsmodels.org/devel/index.html>`_.

The package offers both an API and a command line interface for the EXAM pipeline. Please refer to :doc:`quickstart` for short examples of each. 
