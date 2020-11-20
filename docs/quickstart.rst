Quickstart
==========

Please see below for minimal examples of the full EXAM pipeline using the API or command-line interface.

**API**

.. code-block:: python

  import pandas as pd
  import numpy as np
  from exam import compute_probs, assign, estimate_effects

  # Read in data containing WTP and predicted treatment effects
  data = pd.read_csv("path_to_your_data.csv")

  # Compute treatment probabilities with default parameters
  probs_dict = compute_probs(wtp = data['wtp0','wtp1',...], predicted_effects = data['predicted_effect0','predicted_effect1',...])
  probs = probs_dict['p_star']

  # Assign treatments
  treatment_assignments = assign(probs)

  # Get outcome data
  outcome_data = pd.read_csv("path_to_your_outcome_data.csv")

  # Estimate treatment effects
  effects_dict = estimate_effects(Y = outcome_data['Y'], D = treatment_assignments, probs = probs)

  # Print estimation results
  print(effects_dict['coefs']) # Estimated treatment coefficients
  print(effects_dict['p']) # P-values for treatment coefficients
  print(effects_dict['varcov']) # Variance-covariance matrix
  print(effects_dict['model'])

**Command Line**

.. code-block:: bash

  python compute_probs.py --data wtp_pte_data.csv --output computed_probs.csv --assign_output assignments.csv
  python estimate_effects.py --data outcome_data.csv --probs computed_probs.csv --output effects_output.csv

The following covers each interface in greater detail.

Python API
~~~~~~~~~~~~~

Compute Treatment Probabilities and Assignment
-----------------------------------------------

The treatment probability calculation and assignment functions are ``compute_probs`` and ``assign``, respectively. Below are some lines which demonstrate some of the key parameters for each function. Please refer to the API documentation :doc:exam for the full list of keyword parameters and output values.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from exam import compute_probs, assign, estimate_effects

  ### Compute treatment probabilities
  # Read in data containing WTP and predicted treatment effects
  data = pd.read_csv("path_to_your_data.csv")
  wtp = data['wtp0','wtp1',...]
  pte = data['predicted_effect0','predicted_effect1',...]

  # Can set seed to get the same compute probabilities for each subject
  probs_dict0 = compute_probs(wtp, pte, seed = 1)
  probs_dict1 = compute_probs(wtp, pte, seed = 1)

  assert np.array_equal(probs_dict0['p_star'], probs_dict1['p_star'])

  # Can set capacities for each treatment as well as the market-clearing error threshold
  treatment_capacities = [c0, c1, c2, etc...] # Total capacity must be at least as many subjects in the study
  probs_dict = compute_probs(wtp, pte, capacity = treatment_capacities, error_threshold = 0.005)

  # Can set either a common budget for every participant or distribute budgets unequally
  probs_dict = compute_probs(wtp, pte, budget = 100)
  unequal_budgets = np.random.choice(100, size=wtp.shape[0])
  probs_dict = compute_probs(wtp, pte, subject_budgets = unequal_budgets)

  # Can send in treatment labels as output columns, subject ids as output indices, and choose to output probabilities to a CSV file
  probs_dict = compute_probs(wtp, pte, treatment_labels = ['control', 'treatment1', 'treatment2', ...], subject_ids = ['id0', 'id1', 'id2', ...], save_path = "computed_probs.csv")
  probs = probs_dict['p_star']
  assert probs.columns == ['control', 'treatment1', 'treatment2', ...]
  assert probs.index == ['id0', 'id1', 'id2', ...]

  ### Assign treatment
  # The default output of the `assign` function will be a pandas DataFrame with default indices and treatment names (integer)
  default_assignments = assign(probs) # indices: 0,1,...; treatment labels: 0,1,...

  # We can pass in subject ids and treatment labels
  labelled_assignments = assign(probs, subject_ids = ['id0', 'id1', ...], treatment_labels = ['control', 'treatment1', 'treatment2', ...])

Estimate Treatment Effects
--------------------------------

After assigning treatments and running the experiment, the ``estimate_effects`` function computes an unbiased estimate of the treatment effects. The function offers two estimation methods as outlined in :cite:`Narita2020`, where "matched" computes the weighted average of estimated coefficients for each propensity-score (probability) matched subpopulation, and "single" estimates a single OLS regression controlling for treatment probabilities. The former produces an unbiased estimate of the ATE (average treatment effect) for each treatment, whereas the latter produces an unbiased estimate of a well-defined weighted average of the CATE (conditional average treatment effect) for each propensity-score subpopulation.

.. code-block:: python

  import pandas as pd
  import numpy as np
  from exam import compute_probs, assign, estimate_effects

  # Read in data
  outcome_data = pd.read_csv("path_to_your_outcome_data.csv")
  treatment_probabilities = pd.read_csv("treatment_probabilities.csv")

  # Let's assume outcome dataframe has outcome Y, treatment assignments D, and controls
  Y = outcome_data['Y']
  assignments = outcome_data['D']
  control_variables = outcome_data[['list', 'of', 'controls']]

  # Default method will be "matched" for an unbiased estimate of ATE
  # We can send in control variables into the regression through the `X` parameter
  matched_estimate = estimate_effects(Y = Y, D = assignments, probs = treatment_probabilities, X = control_variables)

  # By default the lowest factor in the `assignments` object will be excluded as the control treatment
  # To set another treatment as the control, you can pass the label as it appears in `assignments` into the `control` parameter
  matched_estimate = estimate_effects(Y = Y, D = assignments, probs = treatment_probabilities, X = control_variables, control = "another_treatment")

  # The 'model' key in the return dictionary will be either a dictionary of subpopulation models indexed by propensity vectors (if using the "matched" method) or a single fitted OLS results object (if using the "single" method)
  matched_estimate = estimate_effects(Y = Y, D = assignments, probs = treatment_probabilities, X = control_variables)
  matched_estimate['model'] # Dictionary of models

  single_estimate = estimate_effects(Y = Y, D = assignments, method = "single", probs = treatment_probabilities, X = control_variables)
  single_estimate['model'].summary() # OLS RegressionResults object

  # We can save estimation outputs to CSV
  matched_estimate = estimate_effects(Y = Y, D = assignments, probs = treatment_probabilities, X = control_variables, save_path = "estimated_effects.csv")

Command-Line Tool
~~~~~~~~~~~~~~~~~~

Compute Treatment Probabilities and Assignment
-----------------------------------------------

The script ``compute_probs.py`` can also be called from the command line to compute welfare-optimal treatment probabilities. The required input is the path to a saved CSV file with WTP and PTE data, in that order, passed to ``--data``. The computed probabilities will be saved to ``--output``, and treatment assignments to ``--assign_output``. Below is an example demonstrating all the possible parameters that can be set.

.. code-block:: bash

  # For command line documentation
  python compute_probs.py -h
  python compute_probs.py --data wtp_pte_data.csv --output computed_probs.csv --assign_output assignments.csv --capacity 100 50 50 --pbound 0.2 --error 0.1 --iterations 20 --budget 100 --subject_budgets budgets.csv --labels control t1 t2 --index

Estimate Treatment Effects
--------------------------------

Similarly, the script ``estimate_effects.py`` can be called from the command line to compute treatment effects. The required inputs are the path to a saved CSV file with the experiment outcome data with the outcome variable, treatment assignments, and controls, in that order, and the path to a saved CSV file with the previously computed treatment probabilities. Below is an example demonstrating all the possible parameters that can be set.

.. code-block:: bash

  # For command line documentation
  python estimate_effects.py -h
  python estimate_effects.py --data outcomes.csv --probs computed_probs.csv --output est_effects.csv --control control --method matched --dindex --pindex --noverb
