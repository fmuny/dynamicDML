# dynamicDML
A Python implementation of dynamic Double Machine Learning (DML) as
developed in [Bodory, Huber & Laffers (2022)](https://doi.org/10.1093/ectj/utac018) and [Bradic, Ji & Zhang
(2024)](https://doi.org/10.1214/24-AOS2352). The `dynamicDML` package allows to flexibly estimate counterfactual outcomes
and treatment effects of sequential policies from observational data, where
treatment assignment may dynamically depend on time-varying characteristics.
For a detailed overview of these methods, see [Muny (2025)](https://arxiv.org/abs/2506.11960).

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

The implementation has been developed and tested in Python version 3.12.

Examples
----------------------------

The following examples demonstrate the basic usage of the `dynamicDML`
package with default settings.
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

Release Notes
----------------------------
- Version 0.1.0: Unpublished
- Version 0.2.0: Initial release of `dynamicDML` python package

References
----------------------------
- Bodory, H., Huber, M., & Laffers, L. (2022). Evaluating (weighted) dynamic treatment effects by double machine learning. The Econometrics Journal, 25(3), 648. [[1]](https://doi.org/10.1093/ectj/utac018)
- Bradic, J., Ji, W., & Zhang, Y. (2024). High-dimensional inference for dynamic treatment effects. The Annals of Statistics, 52(2), 415-440. [[2]](https://doi.org/10.1214/24-AOS2352) 
- Muny, F. (2025). Evaluating Program Sequences with Double Machine Learning: An Application to Labor Market Policies. arXiv preprint arXiv:2506.11960. [[3]](https://arxiv.org/abs/2506.11960)
