"""
Description
----------------------------
A Python implementation of dynamic Double Machine Learning (DML) as
developed in Bodory, Huber & Lafférs (2022)[^Bodory3] and Bradic, Ji & Zhang
(2024).[^Bradic3]
The `dynamicDML` package allows to flexibly estimate counterfactual outcomes
and treatment effects of sequential policies from observational data, where
treatment assignment may dynamically depend on time-varying characteristics.
For an overview of these methods, see Muny (2025).[^Muny3]

Installation
----------------------------
To install the `dynamicDML` package run
```
pip install dynamicDML
```
in the terminal. `dynamicDML` requires the following dependencies:

* flaml[automl]>=2.3.3
* matplotlib>=3.10.0
* mgzip>=0.2.1
* numpy>=2.2.3
* pandas>=2.2.3
* scikit-learn>=1.6.1
* scipy>=1.15.2
* seaborn>=0.13.2

The implementation relies on Python 3.

Basic Example
----------------------------

The following examples demonstrate the basic usage of the `dynamicDML`
package.
```
# load packages
import dynamicDML
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

# Seed
seed = 999
# Generate data
data = dynamicDML.dyn_data_example(n=2000, random_state=seed)

# Define counterfactual contrasts of interest
all_treat = np.ones_like(data['D1'])
all_control = np.zeros_like(data['D1'])

# Basic setting with Linear and Logistic regression
model = dynamicDML.dml2periods(dynamic_confounding=True, random_state=seed)

# APO treat-treat
model = model.init_sequence(
    d1treat='treat',
    d2treat='treat',
    MLmethod_p1=LogisticRegression(),
    MLmethod_p2=LogisticRegression(),
    MLmethod_mu=LinearRegression(),
    MLmethod_nu=LinearRegression()
    )
model = model.fit_sequence(
    'treat', 'treat', data['Y'], data['D1'], data['D2'], data['X0'],
    data['X1'], g1t=all_treat, g2t=all_treat)
model.sequence_summary()
model = model.compute_APO(d1treat='treat', d2treat='treat')

# APO control-control
model = model.init_sequence(
    d1treat='control',
    d2treat='control',
    MLmethod_p1=LogisticRegression(),
    MLmethod_p2=LogisticRegression(),
    MLmethod_mu=LinearRegression(),
    MLmethod_nu=LinearRegression()
    )
model = model.fit_sequence(
    'control', 'control', data['Y'], data['D1'], data['D2'], data['X0'],
    data['X1'], g1t=all_control, g2t=all_control)
model.sequence_summary()
model = model.compute_APO(d1treat='control', d2treat='control')

# ATE treat-treat vs. control-control
model = model.compute_ATE(
    d1treat='treat', d2treat='treat', d1control='control', d2control='control')

# GATE treat-treat vs. control-control for first covariate
model = model.compute_GATEmATE(
    d1treat='treat', d2treat='treat', d1control='control', d2control='control',
    groupvar=(data['X1'][:, 0] > 0), name_groupvar='X1')
```

Advanced Example
----------------------------
The following examples demonstrate the usage of the `dynamicDML`
package with tuning and trimming.
```
# %% Load packages

import numpy as np
from dynamicDML import (
    dyn_data_example, dml2periods, FlamlRegressor, FlamlClassifier)

# %% Generate data
seed = 999
data_dict = dyn_data_example(n=2000, random_state=seed)

x0 = data_dict['X0']
d1 = data_dict['D1']
x1 = data_dict['X1']
d2 = data_dict['D2']
y = data_dict['Y']
y00 = data_dict['Y00']
y10 = data_dict['Y10']
y01 = data_dict['Y01']
y11 = data_dict['Y11']

# %% Oracle estimator

# True ate = alpha_y11 + np.sum(gamma_y11_signal) - alpha_y00 = -3
ate = -3
print(f"ATE_true = {ate:.3f}")
ite = y11-y00

# Oracle ATE (that has access to potential outcomes)
ate_oracle = np.mean(ite)
print(f"ATE_oracle = {ate_oracle:.3}")

# Difference-in-means estimator
ate_DiM = np.mean(y[(d1 == 1) & (d2 == 1)]) - np.mean(y[(d1 == 0) & (d2 == 0)])
print(f"ATE_DiM = {ate_DiM:.3}")

# E[y^{11}]
print(f"E[y^11]={np.mean(y11):.3}")
print(f"E[y|d_1=1, d_2=1]={np.mean(y[(d1 == 1) & (d2 == 1)]):.3}")
# -> confounding because E[y^11] != E[y|d_1=1, d_2=1]

# E[y^{11}]
print(f"E[y^00]={np.mean(y00):.3}")
print(f"E[y|d_1=0, d_2=0]={np.mean(y[(d1 == 0) & (d2 == 0)]):.3}")
# -> confounding because E[y^00] != E[y|d_1=0, d_2=0]

# %% Estimate using dml2periods with tuned Random Forest

# Initialize model
model = dml2periods(dynamic_confounding=True, random_state=seed)
model.sequence_summary()

# APO treat-treat
# 1.) Define policy of interest
all_treat = np.ones_like(d1)
# 2.) Initialize sequence
model = model.init_sequence(
    d1treat='treat',
    d2treat='treat',
    MLmethod_p1=FlamlClassifier(
        time=10, estimator_list=['rf'], metric="log_loss", verbose=2,
        random_state=seed),
    MLmethod_p2=FlamlClassifier(
        time=10, estimator_list=['rf'], metric="log_loss", verbose=2,
        random_state=seed),
    MLmethod_mu=FlamlRegressor(
        time=10, estimator_list=['rf'], metric="mse", verbose=2,
        random_state=seed),
    MLmethod_nu=FlamlRegressor(
        time=10, estimator_list=['rf'], metric="mse", verbose=2,
        random_state=seed))
model.sequence_summary()
# 3.) Tune nuisance learners (takes 10 seconds per nuisance function)
model = model.tune_auto_sequence(
    'treat', 'treat', y, d1, d2, x0, x1, g1t=all_treat, g2t=all_treat)
model.sequence_summary()
# 4.) Fit nuisance learners
model = model.fit_sequence(
    'treat', 'treat', y, d1, d2, x0, x1, g1t=all_treat, g2t=all_treat)
model.sequence_summary()
# 5.) Compute APO
model = model.compute_APO(d1treat='treat', d2treat='treat')

# APO control-control
# 1.) Define policy of interest
all_control = np.zeros_like(d1)
# 2.) Initialize sequence
model = model.init_sequence(
    d1treat='control',
    d2treat='control',
    MLmethod_p1=FlamlClassifier(
        time=10, estimator_list=['rf'], metric="log_loss", verbose=2,
        random_state=seed),
    MLmethod_p2=FlamlClassifier(
        time=10, estimator_list=['rf'], metric="log_loss", verbose=2,
        random_state=seed),
    MLmethod_mu=FlamlRegressor(
        time=10, estimator_list=['rf'], metric="mse", verbose=2,
        random_state=seed),
    MLmethod_nu=FlamlRegressor(
        time=10, estimator_list=['rf'], metric="mse", verbose=2,
        random_state=seed))
model.sequence_summary()
# 3.) Tune nuisance learners (takes 10 seconds per nuisance function)
model = model.tune_auto_sequence(
    'control', 'control', y, d1, d2, x0, x1, g1t=all_control, g2t=all_control)
model.sequence_summary()
# 4.) Fit nuisance learners
model = model.fit_sequence(
    'control', 'control', y, d1, d2, x0, x1, g1t=all_control, g2t=all_control)
model.sequence_summary()
# 5.) Compute APO
model = model.compute_APO(d1treat='control', d2treat='control')

# Treatment effects treat-treat vs. control-control
# ATE
model = model.compute_ATE(
    d1treat='treat', d2treat='treat', d1control='control', d2control='control')

# GATE for X1 > 0
model = model.compute_ATE(
    d1treat='treat', d2treat='treat', d1control='control', d2control='control',
    groupvar=(x0[:, 0] > 0), name_groupvar='x01', trim=0)
# GATE minus ATE for X1 > 0
model = model.compute_GATEmATE(
    d1treat='treat', d2treat='treat', d1control='control', d2control='control',
    groupvar=(x0[:, 0] > 0), name_groupvar='x01', trim=0)

# Plot propensity scores
model.plot_pscores('treat', 'treat')
model.plot_pscores('control', 'control')

# Joint minmax trimming
trim = model.joint_trimming(['treat_treat', 'control_control'], trim=(0, 1))
# Re-fit
model = model.compute_ATE(
    d1treat='treat', d2treat='treat', d1control='control', d2control='control',
    trim=trim)
# compare propensity scores
model.plot_pscores('treat', 'treat')
model.plot_pscores('treat', 'treat', trim=trim)
```

Release Notes
----------------------------
Version 0.1.0: Initial (experimental) release of `dynamicDML` python package

Authors
----------------------------
Fabian Muny

References
----------------------------
[^Bodory3]:
    Bodory, H., Huber, M., & Lafférs, L. (2022). Evaluating (weighted) dynamic
    treatment effects by double machine learning. The Econometrics Journal,
    25(3), 648.
[^Bradic3]:
    Bradic, J., Ji, W., & Zhang, Y. (2024). High-dimensional inference for
    dynamic treatment effects. The Annals of Statistics, 52(2), 415–440.
[^Muny3]:
    Muny, F. (2025). Evaluating Program Sequences with Double Machine Learning:
    An Application to Labor Market Policies. Manuscript in preparation.
"""

from dynamicDML.dml2periods import dml2periods
from dynamicDML._example_data import dyn_data_example
from dynamicDML._flaml_estimators import FlamlRegressor, FlamlClassifier
__all__ = [
    "dml2periods", "dyn_data_example", "FlamlRegressor", "FlamlClassifier"]
__version__ = "0.1.0"
__module__ = 'dynamicDML'
__author__ = "Fabian Muny"
__copyright__ = "Copyright (c) 2025, Fabian Muny"
__license__ = "MIT License"
