# %%
import numpy as np
from optuna.distributions import FloatDistribution, IntDistribution
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from dynamicDML import (
    FlamlClassifier,
    FlamlRegressor,
    dml2periods,
    dyn_data_example,
)
#%%

def print_tuning_result(label, tune_result):
    print(f"\n{label}")
    for nuisance, result in tune_result.items():
        params = result["best_params"].get("treat")
        score = result["minimum_score"].get("treat")
        estimator = result["estimator"].get("treat")
        print(f"  {nuisance}: {type(estimator).__name__}")
        print(f"    minimum_score: {score}")
        print(f"    best_params: {params}")


def init_flaml_model(seed):
    model = dml2periods(dynamic_confounding=True, random_state=seed)
    return model.init_sequence(
        d1treat="treat",
        d2treat="treat",
        MLmethod_p1=FlamlClassifier(
            time=5, estimator_list=["rf"], metric="log_loss",
            random_state=seed),
        MLmethod_p2=FlamlClassifier(
            time=5, estimator_list=["rf"], metric="log_loss",
            random_state=seed),
        MLmethod_mu=FlamlRegressor(
            time=5, estimator_list=["rf"], metric="mse",
            random_state=seed),
        MLmethod_nu=FlamlRegressor(
            time=5, estimator_list=["rf"], metric="mse",
            random_state=seed),
    )


def init_optuna_model(seed):
    model = dml2periods(dynamic_confounding=True, random_state=seed)
    return model.init_sequence(
        d1treat="treat",
        d2treat="treat",
        MLmethod_p1=RandomForestClassifier(random_state=seed, n_jobs=1),
        MLmethod_p2=RandomForestClassifier(random_state=seed, n_jobs=1),
        MLmethod_mu=RandomForestRegressor(random_state=seed, n_jobs=1),
        MLmethod_nu=RandomForestRegressor(random_state=seed, n_jobs=1),
    )

# %%
if __name__ == "__main__":
    seed = 42
    data = dyn_data_example(n=500, random_state=seed)
    y = data["Y"]
    d1 = data["D1"]
    d2 = data["D2"]
    x0 = data["X0"]
    x1 = data["X1"]
    all_treat = np.ones_like(d1)

    flaml_model = init_flaml_model(seed)
    flaml_model = flaml_model.tune_auto_sequence(
        "treat", "treat", y, d1, d2, x0, x1,
        g1t=all_treat, g2t=all_treat)
    print_tuning_result(
        "FLAML tuning",
        flaml_model.sequences["treat_treat"].tune_result)

    rf_class_space = {
        "n_estimators": IntDistribution(50, 200),
        "max_depth": IntDistribution(2, 12),
        "min_samples_leaf": IntDistribution(1, 10),
        "max_features": FloatDistribution(0.4, 1.0),
    }
    rf_reg_space = {
        "n_estimators": IntDistribution(50, 200),
        "max_depth": IntDistribution(2, 12),
        "min_samples_leaf": IntDistribution(1, 10),
        "max_features": FloatDistribution(0.4, 1.0),
    }
    optuna_spaces = {
        "p1": rf_class_space,
        "p2": rf_class_space,
        "mu": rf_reg_space,
        "nu": rf_reg_space,
    }
    scoring = {
        "p1": "neg_log_loss",
        "p2": "neg_log_loss",
        "mu": "neg_mean_squared_error",
        "nu": "neg_mean_squared_error",
    }
    optuna_kwargs = {
        "n_trials": 10,
        "n_jobs": 1,
        "verbose": 0,
    }

    optuna_model = init_optuna_model(seed)
    optuna_model = optuna_model.tune_optuna_sequence(
        "treat", "treat", y, d1, d2, x0, x1,
        g1t=all_treat, g2t=all_treat,
        param_distributions=optuna_spaces,
        scoring=scoring,
        cv=3,
        optuna_kwargs=optuna_kwargs)
    print_tuning_result(
        "Optuna tuning",
        optuna_model.sequences["treat_treat"].tune_result)
#%%
optuna_model.sequence_summary()
optuna_model.fit_sequence(
        "treat", "treat", y, d1, d2, x0, x1,
        g1t=all_treat, g2t=all_treat,)
optuna_model.sequence_summary()
optuna_model.compute_APO(d1treat='treat', d2treat='treat')
flaml_model.fit_sequence(
        "treat", "treat", y, d1, d2, x0, x1,
        g1t=all_treat, g2t=all_treat,)
flaml_model.compute_APO(d1treat='treat', d2treat='treat')
print(np.mean(data['Y11']))

optuna_model.plot_pscores('treat', 'treat')
flaml_model.plot_pscores('treat', 'treat')

# %%

# Check import and export
export_dir = "D:/Test"

# Lightweight export: stores fitted score ingredients, not learner objects.
optuna_model.export_sequence(
    d1treat="treat",
    d2treat="treat",
    path=str(export_dir),
    fitted_only=True,
)

restored = dml2periods(
        dynamic_confounding=True,
        random_state=42,
        verbose=False,
    )

restored.import_sequence(
    d1treat="treat",
    d2treat="treat",
    path=str(export_dir),
)

restored.compute_APO(d1treat="treat", d2treat="treat")

print(restored.sequences["treat_treat"].is_lightweight)
print(restored.sequences["treat_treat"].is_tuned)
print(restored.sequences["treat_treat"].learner_info)

restored.plot_pscores('treat', 'treat')

# %%
