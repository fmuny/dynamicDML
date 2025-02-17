from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator as BaseEstimator
from flaml import AutoML


# The following two functions have been copied from
# https://github.com/DoubleML/DML-Hyperparameter-Tuning-Replication/blob/main/doubleml_flaml_api/doubleml_flaml_api.py
class FlamlRegressor(BaseEstimator):
    """
    Initialize instance of
    [``flaml.AutoML``](https://microsoft.github.io/FLAML/docs/reference/automl/automl){:target="_blank"}
     for regression to be used for
    tuning DML nuisance functions. See the
    [FLAML documentation](https://microsoft.github.io/FLAML/){:target="_blank"}
    for details.
    """
    _estimator_type = 'regressor'

    def __init__(self, time, estimator_list, metric, *args, **kwargs):
        """
        Parameters
        ----------
        time : float
            A float number of the time budget in seconds.
        estimator_list : List of strings
            Estimator names, e.g. `['lgbm', 'xgboost', 'xgb_limitdepth',
            'catboost', 'rf', 'extra_tree']`
            or
            `['auto']`.
        metric : str
            A string of the metric name or a function, e.g., ``'accuracy',
            'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_weighted',
            'roc_auc_ovo_weighted', 'roc_auc_ovr_weighted', 'f1', 'micro_f1',
            'macro_f1', 'log_loss', 'mae', 'mse', 'r2', 'mape'``. Default is
            ``'auto'``.
        """
        self.auto_ml = AutoML(*args, **kwargs)
        self.time = time
        self.estimator_list = estimator_list
        self.metric = metric

    def set_params(self, **params):
        """ """
        self.auto_ml.set_params(**params)
        return self

    def get_params(self, deep=True):
        """ """
        dict = self.auto_ml.get_params(deep)
        dict["time"] = self.time
        dict["estimator_list"] = self.estimator_list
        dict["metric"] = self.metric
        return dict

    def fit(self, X, y):
        """ """
        self.auto_ml.fit(
            X, y, task="regression", time_budget=self.time,
            estimator_list=self.estimator_list, metric=self.metric)
        self.tuned_model = self.auto_ml.model.estimator
        return self

    def predict(self, X):
        """ """
        preds = self.tuned_model.predict(X)
        return preds

    def set_predict_request(self, X):
        """ """
        pass


class FlamlClassifier(BaseEstimator):
    """
    Initialize instance of
    [``flaml.AutoML``](https://microsoft.github.io/FLAML/docs/reference/automl/automl){:target="_blank"}
     for classification to be used for
    tuning DML nuisance functions. See the
    [FLAML documentation](https://microsoft.github.io/FLAML/){:target="_blank"}
    for details.
    """
    _estimator_type = 'classifier'

    def __init__(self, time, estimator_list, metric, *args, **kwargs):
        """
        Parameters
        ----------
        time : float
            A float number of the time budget in seconds.
        estimator_list : List of strings
            Estimator names, e.g. `['lgbm', 'xgboost', 'xgb_limitdepth',
            'catboost', 'rf', 'extra_tree']`
            or
            `['auto']`.
        metric : str
            A string of the metric name or a function, e.g., ``'accuracy',
            'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_weighted',
            'roc_auc_ovo_weighted', 'roc_auc_ovr_weighted', 'f1', 'micro_f1',
            'macro_f1', 'log_loss', 'mae', 'mse', 'r2', 'mape'``. Default is
            ``'auto'``.
        """
        self.auto_ml = AutoML(*args, **kwargs)
        self.time = time
        self.estimator_list = estimator_list
        self.metric = metric

    def set_params(self, **params):
        """ """
        self.auto_ml.set_params(**params)
        return self

    def get_params(self, deep=True):
        """ """
        dict = self.auto_ml.get_params(deep)
        dict["time"] = self.time
        dict["estimator_list"] = self.estimator_list
        dict["metric"] = self.metric
        return dict

    def fit(self, X, y):
        """ """
        self.classes_ = unique_labels(y)
        self.auto_ml.fit(
            X, y, task="classification", time_budget=self.time,
            estimator_list=self.estimator_list, metric=self.metric)
        self.tuned_model = self.auto_ml.model.estimator
        return self

    def predict_proba(self, X):
        """ """
        preds = self.tuned_model.predict_proba(X)
        return preds

    def set_predict_proba_request(self, X):
        """ """
        pass
