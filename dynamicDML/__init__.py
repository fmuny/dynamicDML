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

Examples
----------------------------

The following examples demonstrate the basic usage of the `dynamicDML`
package with default settings.
```
# load dynamicDML package
import dynamicDML
import numpy as np

# Generate data
data = dyn_data_example(n=n, r_0=r_0, random_state=seed)

# Define seed
seed=999

# Define counterfactual contrasts of interest
all_treat = np.ones_like(data['D1'])
all_control = np.zeros_like(data['D1'])

# Basic setting without tuning
model = dml2periods(dynamic_confounding=True, random_state=seed)

# APO treat-treat
model = model.init_sequence(d1treat='treat', d2treat='treat')
model = model.fit_sequence(
    'treat', 'treat', data['Y'], data['D1'], data['D2'], data['X0'],
    data['X1'], g1t=all_treat, g2t=all_treat)
model.sequence_summary()
model = model.compute_APO(d1treat='treat', d2treat='treat')

# APO control-control
model = model.init_sequence(d1treat='control', d2treat='control')
model = model.fit_sequence(
    'control', 'control', data['Y'], data['D1'], data['D2'], data['X0'],
    data['X1'], g1t=all_control, g2t=all_control)
model.sequence_summary()
model = model.compute_APO(d1treat='control', d2treat='control')

# ATE treat-treat vs. control-control
model = model.compute_ATE(
    d1treat='treat', d2treat='treat', d1control='control', d2control='control')
```

Release Notes
----------------------------
Version 0.1.0: Initial release of `dynamicDML` python package

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
__version__ = "0.0.1"
__module__ = 'dynamicDML'
__author__ = "Fabian Muny"
__copyright__ = "Copyright (c) 2025, Fabian Muny"
__license__ = "MIT License"
