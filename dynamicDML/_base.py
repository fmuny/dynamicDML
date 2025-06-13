from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.validation import check_random_state
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator as BaseEstimator
from sklearn.base import clone


class _BaseClass(ABC):
    """
    A base class that enforces a 'fit' method and
    handles shared parameters between dynamic and static method.
    """

    def __init__(
            self,
            d1treat,
            d2treat,
            random_state=None,
            nfolds=3,
            verbose=True):
        self.d1treat = d1treat
        self.d2treat = d2treat
        self.random_state = random_state
        self.nfolds = nfolds
        self.verbose = verbose
        # Initialize best_params parameter
        self.best_params = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, y2, d1, d2, x0, g1t=None, g2t=None):
        """All subclasses must implement this method."""
        pass

    @abstractmethod
    def tune_auto(self, y2, d1, d2, x0, g1t=None, g2t=None):
        """All subclasses must implement this method."""
        pass

    def _check_vartype(self, var, typ, varname):
        if not isinstance(var, typ):
            raise ValueError(
                f'{varname} must be of type {typ.__name__}, got '
                f'{type(var).__name__}')

    def _check_dicts(self, var, varname, keys):
        self._check_vartype(var, dict, varname)
        if not all(key in keys for key in var.keys()):
            raise ValueError(
                f'allowed keys for {varname} are {keys}, got '
                f'{list(var.keys())}')
        return True

    def _check_dicts_all(self, var, varname, keys):
        self._check_vartype(var, dict, varname)
        if not all(key in var.keys() for key in keys):
            raise ValueError(
                f'If {varname} provided as dict it must contain all keys '
                f'{keys}, got {list(var.keys())}')
        return True


class _dyntreatDML(_BaseClass):
    """
    Base class for estimation of an average potential outcome under dynamic
    confounding for one counterfactual sequence of interest.
    """

    # define init function
    def __init__(self,
                 d1treat,
                 d2treat,
                 MLmethod_p1=RandomForestClassifier(random_state=999),
                 MLmethod_p2=RandomForestClassifier(random_state=999),
                 MLmethod_mu=RandomForestRegressor(random_state=999),
                 MLmethod_nu=RandomForestRegressor(random_state=999),
                 random_state=None,
                 nfolds=3,
                 p2_on_d1_subset=False,
                 extra_split_train_mu=True,
                 extra_split_train_nu=True,
                 crossfit_nu=False,
                 doubly_robust_nu=False,
                 verbose=True):
        """
        Initialize base class for estimation under dynamic confounding.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        MLmethod_p1 : estimator implementing ``fit()`` and ``predict()``
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function :math:`p_{g_1}(X_0)`.
            If a classifier is used with method ``predict_proba()`` this
            method will be preferred over ``predict()``. Default is
            :py:class:`sklearn.ensemble.RandomForestClassifier`.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as :class:`FlamlClassifier`.
        MLmethod_p2 : estimator implementing ``fit()`` and ``predict()``
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function
            :math:`p_{g_2}(\\mathbf{X}_1,g_1)`.
            If a classifier is used with method ``predict_proba()`` this
            method will be preferred over ``predict()``. Default is
            :py:class:`sklearn.ensemble.RandomForestClassifier`.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as :class:`FlamlClassifier`.
        MLmethod_mu : estimator implementing ``fit()`` and ``predict()``
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function
            :math:`\\mu_{\\mathbf{d}_2}(\\mathbf{X}_1)`.
            If a classifier is used with method ``predict_proba()`` this
            method will be preferred over ``predict()``. Default is
            :py:class:`sklearn.ensemble.RandomForestRegressor`.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as :class:`FlamlRegressor`.
        MLmethod_nu : estimator implementing ``fit()`` and ``predict()``
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function :math:`\\nu_{\\mathbf{d}_2}(X)`.
            Default is :py:class:`sklearn.ensemble.RandomForestRegressor`.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as :class:`FlamlRegressor`.
        random_state : int, RandomState instance or None
            Controls randomness of the sample splits in cross-fitting. The
            default is None.
        nfolds : int
            Number of folds for cross-fitting. The default is 3.
        p2_on_d1_subset : bool
            This parameter regulates if we train p2 stratified on the subset
            D_1=d_1 (Bradic, True) or on the full sample with D1 as a regressor
            (Bodory, False). The stratification should reduce bias in
            the estimates of p2 but may increase variance if there are few
            individuals following a particular program d1.
            The default is False.
        extra_split_train_mu : bool
            Whether the mu that directly enters the DML score should be
            estimated on the full (``False``) or only half of the training
            sample (``True``). If ``extra_split_train_nu=True``, setting
            ``extra_split_train_mu=True`` is computationally faster as mu does
            not need to be re-estimated on one half of the training sample
            later. The default is True.
        extra_split_train_nu : bool
            Whether to perform an additional split of the training sample for
            the estimation of nu. This is recommended to avoid overfitting as
            the estimation of nu takes estimates of mu as a pseudo outcome. The
            default is True.
        crossfit_nu : bool
            If ``extra_split_train_nu=True``, cross-fitting allows to regain
            full-sample efficiency by reversing the roles of the subsamples
            created to estimate mu and nu, respectively. However, this leads
            to higher computation time. The default is False.
        doubly_robust_nu : bool
            This parameter indicates whether nu should be estimated in a
            doubly-robust manner as proposed in [2] (``True``). Otherwise, it
            is estimated by regressing mu on pre-treatment covariates as
            proposed in [1] (``False``). The former computationally more
            expensive. The default is False.
        verbose : bool
            Controls the verbosity when fitting and predicting. The default is
            True.

        Notes
        -----
        This method estimates the average potential outcome according to the
        DML-formula

        .. math::

            \\hat\\theta^{\\mathbf{g}_2} = \\frac{1}{N} \\sum_{i=1}^N
            \\hat{\\nu}_{\\mathbf{g}_2}(X_{0i})
            &+ \\frac{\\left(\\hat{\\mu}_{\\mathbf{g}_2}(\\mathbf{X}_{1i})  -
            \\hat{\\nu}_{\\mathbf{g}_2}(X_{0i})  \\right) \\cdot
            \\mathbf{1}\\{D_{1i}=g_1(V_{0i})\\} }{\\hat{p}_{g_1}(X_{0i})}\\\\
            &+ \\frac{\\left(Y_i -  \\hat{\\mu}_{\\mathbf{g}_2}
            (\\mathbf{X}_{1i})\\right)
            \\cdot \\mathbf{1}\\{\\mathbf{D}_{2i}=(g_1(V_{0i}),
            g_2(\\mathbf{V}_{1i}))\\}}{\\hat{p}_{g_2}
            (\\mathbf{X}_{1i}, g_{1})\\hat{p}_{g_1}(X_{0i})}

        For details see [3].

        When estimating counterfactual outcomes of treatment sequences we
        typically want to make an extra split of the training sample to avoid
        overfitting in the estimation of nu, regulated by
        ``extra_split_tain_nu``. The estimate of mu is needed for two things:
        - (a) as prediction to enter the dynamic DML formula
        - (b) as prediction to be used as outcome in the estimation of nu.
        For (a) the extra split is not neccessary but it can
        computationally more convenient re-use the mu estimated to get nu,
        instead of additionally fitting mu on the full sample. This is
        regulated with the paramter ``extra_split_tain_mu``. If it is ``True``,
        mu is trained only on one half of the training sample. Then
        this trained estimate can later be re-used for nu.
        If it is False, mu is trained on the whole training sample.
        Then, if we want an additional split for nu (i.e.
        ``extra_split_train_nu=True``), mu has to be
        re-estimated on one half of the training sample later.

        To recover the estimates from the original papers, use the following
        settings, where ``Bodory`` and ``fewsplits`` are propsed in [1], and
        ``D-DRL``, ``DTL``, and ``D-DRL'`` are proposed in [2]:

        | Overview             | Bodory | fewspl. | D-DRL | DTL   | D-DRL' |
        | -------------------- | ------ | ------- | ----- | ----- | ------ |
        | p2_on_d1_subset      | False  | True    | True  | True  | True   |
        | extra_split_train_mu | True   | False   | False | False | False  |
        | extra_split_train_nu | True   | True    | True  | False | False  |
        | crossfit_nu          | False  | False   | True  | False | False  |
        | doubly_robust_nu     | False  | False   | True  | False | True   |

        The settings ``Bodory`` are used as the default.

        References
        ----------
        .. [1] Bodory, H., Huber, M., & Lafférs, L. (2022). Evaluating
               (weighted) dynamic treatment effects by double machine learning.
               The Econometrics Journal, 25(3), 628–648.
        .. [2] Bradic, J., Ji, W., & Zhang, Y. (2024). High-dimensional
               inference for dynamic treatment effects. The Annals of
               Statistics, 52(2), 415–440.
        .. [3] Muny, F. (2025). Evaluating Program Sequences with Double
               Machine Learning: An Application to Labor Market Policies.
               Manuscript in preparation.
        """
        super().__init__(
            d1treat=d1treat,
            d2treat=d2treat,
            random_state=random_state,
            nfolds=nfolds,
            verbose=verbose)
        self.MLmethod_p1 = MLmethod_p1
        self.MLmethod_p2 = MLmethod_p2
        self.MLmethod_mu = MLmethod_mu
        self.MLmethod_nu = MLmethod_nu
        self.p2_on_d1_subset = p2_on_d1_subset
        self.extra_split_train_mu = extra_split_train_mu
        self.extra_split_train_nu = extra_split_train_nu
        self.crossfit_nu = crossfit_nu
        self.doubly_robust_nu = doubly_robust_nu
        self.is_static = False

    def _check_inputs(self):
        self._check_vartype(
            self.MLmethod_p1, BaseEstimator, 'MLmethod_p1')
        self._check_vartype(
            self.MLmethod_p2, BaseEstimator, 'MLmethod_p2')
        self._check_vartype(
            self.MLmethod_mu, BaseEstimator, 'MLmethod_mu')
        self._check_vartype(
            self.MLmethod_nu, BaseEstimator, 'MLmethod_nu')
        self._check_vartype(self.d1treat, (int, str), 'd1treat')
        self._check_vartype(self.d2treat, (int, str), 'd2treat')
        self._check_vartype(self.nfolds, int, 'nfolds')
        self._check_vartype(self.p2_on_d1_subset, bool, 'p2_on_d1_subset ')
        self._check_vartype(
            self.extra_split_train_mu, bool, 'extra_split_train_mu')
        self._check_vartype(
            self.extra_split_train_nu, bool, 'extra_split_train_nu')
        self._check_vartype(self.crossfit_nu, bool, 'crossfit_nu')
        self._check_vartype(self.doubly_robust_nu, bool, 'doubly_robust_nu')
        self._check_vartype(self.verbose, bool, 'verbose')
        # check whether seed is set (using scikitlearn check_random_state)
        self.random_state = check_random_state(self.random_state)
        # Check dependencies
        if self.crossfit_nu and not self.extra_split_train_nu:
            extra_split_train_nu = self.extra_split_train_nu
            raise ValueError(
                f'crossfit_nu=True requires extra_split_train_nu=True, got '
                f'{extra_split_train_nu=}')

    def _get_treatment_dummies(self, d1, d2):
        # Create treatment indicators
        d1tre = 1*(d1 == self.g1t)
        d2tre = 1*(d2 == self.g2t)
        return d1tre, d2tre

    def _get_dr_nu(
            self,
            nu_hat,
            d2dum_fit,
            d2dum_pred,
            x_fit,
            x_pred,
            y_pred,
            tkey):
        # Re-estimate p2
        p2_fit = self.MLmethod_dict['p2'][tkey].fit(y=d2dum_fit, X=x_fit)
        # Predict p2
        if hasattr(p2_fit, 'predict_proba'):
            p2_pred = p2_fit.predict_proba(x_pred)[:, -1]
        else:
            p2_pred = p2_fit.predict(x_pred)
        # Compute transformed outcome y1 according to DR formula
        y1_new = nu_hat + d2dum_pred * ((y_pred-nu_hat)/p2_pred)
        return y1_new, p2_pred

    def _nuisest(
            self, y2, d1dum, d2dum, x0, x1, tkey, p1_ext=None, p2_ext=None):
        # Generate combined dataframes
        x0x1 = np.concatenate((x0, x1), axis=1)
        if not self.p2_on_d1_subset:
            d1x0x1 = np.concatenate([d1dum[:, None], x0, x1], axis=1)
        # Get folds
        stepsize = np.ceil((1/self.nfolds) * len(y2)).astype(int)
        nobs = min(self.nfolds * stepsize, len(y2))
        # initialize random state for sample splitting
        subsample_random_state = check_random_state(self.random_state)
        idx = subsample_random_state.choice(nobs, nobs, replace=False)
        # Get empty containers
        score = np.empty((0, 7))
        for i in range(1, self.nfolds+1):
            if self.verbose:
                print(f"Fold {i} of {self.nfolds}")
            # Get indices of test sample
            tesample = idx[((i-1)*stepsize):min((i*stepsize), nobs)]
            # Get indices of training sample
            if self.nfolds == 1:
                trsample = tesample
            else:
                trsample = idx[(~np.isin(idx, tesample))]
            # 50:50 split of trsample
            midpoint_tr = np.ceil(len(trsample) / 2).astype('int')
            trsample1 = trsample[:midpoint_tr]
            trsample2 = trsample[midpoint_tr:]
            # Get particulat subsets
            trsample_d1 = trsample[d1dum[trsample] == 1]
            trsample_d11 = trsample[(
                d1dum[trsample] == 1) & (d2dum[trsample] == 1)]
            trsample1_d1 = trsample1[d1dum[trsample1] == 1]
            trsample1_d11 = trsample1[(
                d1dum[trsample1] == 1) & (d2dum[trsample1] == 1)]
            trsample2_d1 = trsample2[d1dum[trsample2] == 1]
            trsample2_d11 = trsample2[(
                d1dum[trsample2] == 1) & (d2dum[trsample2] == 1)]

            # Estimations:
            # 1) p1
            # p1 trained on full trsample, predicted on tesample
            if p1_ext is not None:
                p1te = p1_ext[tesample]
            else:
                p1 = self.MLmethod_dict['p1'][tkey].fit(
                        y=d1dum[trsample], X=x0[trsample, :])
                # Predict
                if hasattr(p1, 'predict_proba'):
                    p1te = p1.predict_proba(x0[tesample, :])[:, -1]
                else:
                    p1te = p1.predict(x0[tesample, :])
                # Return best params if tuning
                if self.best_params is not None:
                    self.best_params['p1'][tkey] = getattr(p1, 'best_params_')

            # 2) p2
            # p2 trained either on subset D_1=d_1 (Bradic) or on full
            # full trsample (Bodory), predicted on tesample
            if p2_ext is not None:
                p2te = p2_ext[tesample]
            else:
                if self.p2_on_d1_subset:
                    p2 = self.MLmethod_dict['p2'][tkey].fit(
                            y=d2dum[trsample_d1], X=x0x1[trsample_d1, :])
                    if hasattr(p2, 'predict_proba'):
                        p2te = p2.predict_proba(x0x1[tesample, :])[:, -1]
                    else:
                        p2te = p2.predict(x0x1[tesample, :])
                else:
                    p2 = self.MLmethod_dict['p2'][tkey].fit(
                            y=d2dum[trsample], X=d1x0x1[trsample, :])
                    if hasattr(p2, 'predict_proba'):
                        p2te = p2.predict_proba(d1x0x1[tesample, :])[:, -1]
                    else:
                        p2te = p2.predict(d1x0x1[tesample, :])
                # Return best params if tuning
                if self.best_params is not None:
                    self.best_params['p2'][tkey] = getattr(p2, 'best_params_')

            # 3) mu
            # estimate and predict mu:
            # Bodory:       train on trsample1_d11,  predict on tesample
            # fewsplits:    train on trsample_d11,   predict on tesample
            # Bradic D-DRL: train on trsample_d11,   predict on tesample
            # Bradic DTL:   train on trsample_d11,   predict on tesample
            # -> for all but Bodory extra_split_train_mu = False
            if self.extra_split_train_mu:
                y2d1d2 = self.MLmethod_dict['mu'][tkey].fit(
                        y=y2[trsample1_d11], X=x0x1[trsample1_d11, :])
            else:
                y2d1d2 = self.MLmethod_dict['mu'][tkey].fit(
                        y=y2[trsample_d11], X=x0x1[trsample_d11, :])
            if hasattr(y2d1d2, 'predict_proba'):
                y2d1d2te = y2d1d2.predict_proba(x0x1[tesample, :])[:, -1]
            else:
                y2d1d2te = y2d1d2.predict(x0x1[tesample, :])
            # Return best params if tuning
            if self.best_params is not None:
                self.best_params['mu'][tkey] = getattr(y2d1d2, 'best_params_')

            # 4) nu
            # The conditional outcome in the first period is estimated using
            # the estimated y2d1d2 as y in the trsample2, only among those
            # that actually received treatment 1. It is predicted on the test
            # sample.

            # estimate and predict nu:
            # Bodory:       train y2 on trsample1_d11,  predict on trsample2
            #               train y1 on trsample2_d1,   predict on tesample
            # fewsplits:    train y2 on trsample_d11,   predict on trsample2
            #               train y1 on trsample2_d1,   predict on tesample
            # Bradic D-DRL: train y2 on trsample1_d11,  predict on trsample2_d1
            #               train p2 on trsample1_d1,   predict on trsample2_d1
            #               train y1 on trsample2_d1,   predict on tesample
            #               train y2 on trsample2_d11,  predict on trsample1_d1
            #               train p2 on trsample2_d1,   predict on trsample1_d1
            #               train y1 on trsample1_d1,   predict on tesample
            # Bradic DTL:   train y2 on trsample_d11,   predict on trsample_d1
            #               train y1 on trsample_d1,    predict on tesample
            # Bradic D-DRL':train y2 on trsample_d11,   predict on trsample_d1
            #               train p2 on trsample_d1,    predict on trsample_d1
            #               train y1 on trsample_d1,    predict on tesample

            # Overview             | Bodory | fewspl. | D-DRL | DTL   | D-DRL'
            # -----------------------------------------------------------------
            # p2_on_d1_subset      | False  | True    | True  | True  | True
            # extra_split_train_mu | True   | False   | False | False | False
            # extra_split_train_nu | True   | True    | True  | False | False
            # crossfit_nu          | False  | False   | True  | False | False
            # doubly_robust_nu     | False  | False   | True  | False | True

            # estimate and predict nu:
            # Case 1: no extra split for nu
            if not self.extra_split_train_nu:
                # Predict y2d1d2 on train sample with d1
                if hasattr(y2d1d2, 'predict_proba'):
                    y2d1d2trd1 = y2d1d2.predict_proba(
                        x0x1[trsample_d1, :])[:, -1]
                else:
                    y2d1d2trd1 = y2d1d2.predict(x0x1[trsample_d1, :])
                # Case 1b (otherwise it's 1a)
                if self.doubly_robust_nu:
                    # Predict p2 on trsample_d1
                    if p2_ext is not None:
                        p2trd1 = p2_ext[trsample_d1]
                    else:
                        if self.p2_on_d1_subset:
                            if hasattr(p2, 'predict_proba'):
                                p2trd1 = p2.predict_proba(
                                        x0x1[trsample_d1, :])[:, -1]
                            else:
                                p2trd1 = p2.predict(x0x1[trsample_d1, :])
                        else:
                            if hasattr(p2, 'predict_proba'):
                                p2trd1 = p2.predict_proba(
                                        d1x0x1[trsample_d1, :])[:, -1]
                            else:
                                p2trd1 = p2.predict(
                                        d1x0x1[trsample_d1, :])
                    # Compute transformed outcome y1 according to DR formula
                    y2d1d2trd1 = y2d1d2trd1 + d2dum[trsample_d1] * ((
                        y2[trsample_d1]-y2d1d2trd1)/p2trd1)
                # Fit y1d1 on whole train sampe
                y1d1 = self.MLmethod_dict['nu'][tkey].fit(
                        y=y2d1d2trd1, X=x0[trsample_d1, :])
                # Predict on test sample (outcome is always a probability)
                y1d1te = y1d1.predict(x0[tesample, :])
            # Case 2: Extra split for nu
            else:
                # If extra split for mu, take trained model from before
                # otherwise train mu again on 1st 1/2 of train set with d1 & d2
                if not self.extra_split_train_mu:
                    y2d1d2 = self.MLmethod_dict['mu'][tkey].fit(
                            y=y2[trsample1_d11], X=x0x1[trsample1_d11, :])
                # predict mu on 2nd 1/2 of train set with d1
                if hasattr(y2d1d2, 'predict_proba'):
                    y2d1d2tr2d1 = y2d1d2.predict_proba(
                            x0x1[trsample2_d1, :])[:, -1]
                else:
                    y2d1d2tr2d1 = y2d1d2.predict(
                            x0x1[trsample2_d1, :])
                if self.doubly_robust_nu:
                    # Re-estimate p2 on 1st 1/2 of train set (considering
                    # p2_on_d1_subset parameter)
                    # predict p2 on 2nd 1/2 of train set with d1
                    # Compute transformed outcome y1 according to DR formula.
                    # For DR nu I do not consider pre-specified pscores
                    # because they should only be learned from one half of
                    # the training set: for a particular individual, p2_new
                    # may be different from p2te but only p2te is provided
                    # externally
                    if self.p2_on_d1_subset:
                        y2d1d2tr2d1, p2_new = self._get_dr_nu(
                                nu_hat=y2d1d2tr2d1,
                                d2dum_fit=d2dum[trsample1_d1],
                                d2dum_pred=d2dum[trsample2_d1],
                                x_fit=x0x1[trsample1_d1, :],
                                x_pred=x0x1[trsample2_d1, :],
                                y_pred=y2[trsample2_d1],
                                tkey=tkey)
                    else:
                        y2d1d2tr2d1, p2_new = self._get_dr_nu(
                                nu_hat=y2d1d2tr2d1,
                                d2dum_fit=d2dum[trsample1],
                                d2dum_pred=d2dum[trsample2_d1],
                                x_fit=d1x0x1[trsample1, :],
                                x_pred=d1x0x1[trsample2_d1, :],
                                y_pred=y2[trsample2_d1],
                                tkey=tkey)
                    # There might be observations with predicted pscore of 0.
                    # I exclude those observations from the training.
                    not_trim = p2_new > 0
                    if np.sum(p2_new == 0) > 0:
                        print(f"WARNING: {np.sum(p2_new == 0)} observations"
                              f" not considered in DR-pseudo-outcomes due to"
                              f" propensity score predicted to be zero.")
                else:
                    not_trim = np.full(len(y2d1d2tr2d1), True)
                # Fit y1d1 on trainsample2_d1
                y1d1 = self.MLmethod_dict['nu'][tkey].fit(
                        y=y2d1d2tr2d1[not_trim], X=x0[trsample2_d1, :][
                            not_trim, :])
                # Predict on test sample
                y1d1te = y1d1.predict(x0[tesample, :])
                # If nu should be cross-fitted exchange roles
                if self.crossfit_nu:
                    # Fit mu on 2nd 1/2 of train set with d1 & d2
                    y2d1d2 = self.MLmethod_dict['mu'][tkey].fit(
                            y=y2[trsample2_d11], X=x0x1[trsample2_d11, :])
                    # predict mu on 1st 1/2 of train set with d1
                    if hasattr(y2d1d2, 'predict_proba'):
                        y2d1d2tr1d1 = y2d1d2.predict_proba(x0x1[
                                trsample1_d1, :])[:, -1]
                    else:
                        y2d1d2tr1d1 = y2d1d2.predict(
                                x0x1[trsample1_d1, :])
                    if self.doubly_robust_nu:
                        # DR transformation of y2d1d2
                        if self.p2_on_d1_subset:
                            y2d1d2tr2d1, p2_new = self._get_dr_nu(
                                    nu_hat=y2d1d2tr1d1,
                                    d2dum_fit=d2dum[trsample2_d1],
                                    d2dum_pred=d2dum[trsample1_d1],
                                    x_fit=x0x1[trsample2_d1, :],
                                    x_pred=x0x1[trsample1_d1, :],
                                    y_pred=y2[trsample1_d1],
                                    tkey=tkey)
                        else:
                            y2d1d2tr2d1, p2_new = self._get_dr_nu(
                                    nu_hat=y2d1d2tr1d1,
                                    d2dum_fit=d2dum[trsample2],
                                    d2dum_pred=d2dum[trsample1_d1],
                                    x_fit=d1x0x1[trsample2, :],
                                    x_pred=d1x0x1[trsample1_d1, :],
                                    y_pred=y2[trsample1_d1],
                                    tkey=tkey)
                        not_trim = p2_new > 0
                        if np.sum(p2_new == 0) > 0:
                            print(f"WARNING: {np.sum(p2_new == 0)} observation"
                                  f"s not considered in DR-pseudo-outcomes due"
                                  f" to propensity score predicted to be zero"
                                  f".")
                        # Fit y1d1 on train sample1_d1
                    else:
                        not_trim = np.full(len(y2d1d2tr1d1), True)
                    y1d1 = self.MLmethod_dict['nu'][tkey].fit(
                            y=y2d1d2tr1d1[not_trim], X=x0[trsample1_d1, :][
                                not_trim, :])
                    # Predict on test sample
                    y1d1te_fit1 = y1d1.predict(x0[tesample, :])
                    # Average predictions of the two runs
                    y1d1te = np.mean(np.array([y1d1te, y1d1te_fit1]), axis=0)
            # Return best params if tuning
            if self.best_params is not None:
                self.best_params['nu'][tkey] = getattr(y1d1, 'best_params_')
            # combine to one matrix
            res = np.concatenate([
                d1dum[tesample].reshape(-1, 1),
                d2dum[tesample].reshape(-1, 1),
                y2[tesample].reshape(-1, 1),
                y2d1d2te.reshape(-1, 1),
                p1te.reshape(-1, 1),
                p2te.reshape(-1, 1),
                y1d1te.reshape(-1, 1)], axis=1)
            # Add to score
            score = np.concatenate([score, res], axis=0)
        # Sort according to indices
        score = score[idx.argsort(), :]
        # Compute DR score
        cond = (score[:, 4] > 0) & (score[:, 5] > 0)
        tscores = np.empty_like(score[:, 4])*np.nan
        tscores1 = (score[cond, 0]*score[cond, 1]*(score[cond, 2] - score[
            cond, 3])/(score[cond, 4]*score[cond, 5]))
        tscores2 = (score[cond, 0]*(score[cond, 3]-score[cond, 6])/(score[
            cond, 4]))
        tscores[cond] = tscores1 + tscores2 + score[cond, 6]
        return score[:, 3], score[:, 4], score[:, 5], score[:, 6], tscores

    def _prepare_fit(
            self, y2, d1, d2, x0, x1, p1t=None, p2t=None, g1t=None, g2t=None):
        # Use sklearn input checks to allow for multiple types of inputs:
        # - returns numpy arrays for X and y (no matter which input type)
        # - makes sure dimensions are consistent
        # - forces y to be numeric
        x0, y2 = check_X_y(x0, y2, y_numeric=True)
        x0, d1 = check_X_y(x0, d1, y_numeric=True)
        if x1 is not None:
            x1, d2 = check_X_y(x1, d2, y_numeric=True)
        else:
            x0, d2 = check_X_y(x0, d2, y_numeric=True)
            x1 = np.array([]).reshape(x0.shape[0], 0)
        self.n_feat_x0 = x0.shape[1]
        self.n_feat_x1 = x1.shape[1]
        # Check if propensity scores known (e.g. if estimation for different
        # outcome, propensity scores do not need to be re-estimated)
        if p1t is None and p2t is None:
            self.p1_known = False
            self.p2_known = False
        if (p1t is not None):
            self.p1_known = True
            x0, p1t = check_X_y(x0, p1t, y_numeric=True)
            if (p1t.max() > 1) or (p1t.min() < 0):
                raise ValueError(
                    f'Provided propensity scores p1t must be in interval [0, 1'
                    f'], got [{p1t.min()}, {p1t.max()}].')
        if (p2t is not None):
            self.p2_known = True
            x0, p2t = check_X_y(x0, p2t, y_numeric=True)
            if (p2t.max() > 1) or (p2t.min() < 0):
                raise ValueError(
                    f'Provided propensity scores p2t must be in interval [0, 1'
                    f'], got [{p2t.min()}, {p2t.max()}].')
        # Check other params
        self._check_inputs()
        # Check if treatment rule provided
        if g1t is None and g2t is None:
            self.g1_known = False
            self.g2_known = False
        if (g1t is not None):
            self.g1_known = True
            x0, g1t = check_X_y(x0, g1t, y_numeric=True)
            # Make sure that all values in g1t exist in d1
            not_in_d1 = np.unique(g1t)[~np.isin(np.unique(g1t), d1)]
            if (not_in_d1.size > 0):
                raise ValueError(
                    f'Values {not_in_d1} in g1t are not present in d1.')
        if (g2t is not None):
            self.g2_known = True
            x0, g2t = check_X_y(x0, g2t, y_numeric=True)
            # Make sure that all values in g2t exist in d2
            not_in_d2 = np.unique(g2t)[~np.isin(np.unique(g2t), d2)]
            if (not_in_d2.size > 0):
                raise ValueError(
                    f'Values {not_in_d2} in g1t are not present in d2.')
        # Check type of treatment provided
        if type(self.d1treat) is int:
            self.d1treat_provided_int = True
        else:
            self.d1treat_provided_int = False
        if type(self.d2treat) is int:
            self.d2treat_provided_int = True
        else:
            self.d2treat_provided_int = False
        if (not self.d1treat_provided_int) and (g1t is None):
            raise ValueError(
                'If d1treat provided as string, treatment indicator g1t must '
                'be provided to fit. Otherwise povide d1treat as int.')
        if (not self.d2treat_provided_int) and (g2t is None):
            raise ValueError(
                'If d2treat provided as string, treatment indicator g2t must '
                'be provided to fit. Otherwise povide d1treat as int.')
        if (self.d1treat_provided_int) and (g1t is not None):
            self.d1treat = str(self.d1treat)
            print(
                'Warning: d1treat provided as int while g1t is provided. Using'
                ' g1t for estimation; converting d1treat to string.')
        if (self.d2treat_provided_int) and (g2t is not None):
            self.d2treat = str(self.d2treat)
            print(
                'Warning: d2treat provided as int while g2t is provided. Using'
                ' g2t for estimation; converting d2treat to string.')
        # Check existence and unique values of treatments
        d1_unique = np.unique(d1)
        if (self.d1treat_provided_int) and (g1t is None) and (
                self.d1treat not in d1_unique):
            raise ValueError(f'd1treat = {self.d1treat} not found in d1.')
        d2_unique = np.unique(d2)
        if (self.d2treat_provided_int) and (g2t is None) and (
                self.d2treat not in d2_unique):
            raise ValueError(f'd2treat = {self.d2treat} not found in d2.')
        # If g1t and g2t not provided, create them from d1treat and d2treat
        if g1t is None:
            g1t = np.ones_like(d1)*self.d1treat
        if g2t is None:
            g2t = np.ones_like(d2)*self.d2treat
        # create dictionary of methods
        self.MLmethod_dict = {'p1': {}, 'p2': {}, 'mu': {}, 'nu': {}}
        self.MLmethod_dict['p1']['treat'] = clone(self.MLmethod_p1)
        self.MLmethod_dict['p2']['treat'] = clone(self.MLmethod_p2)
        self.MLmethod_dict['mu']['treat'] = clone(self.MLmethod_mu)
        self.MLmethod_dict['nu']['treat'] = clone(self.MLmethod_nu)
        return y2, d1, d2, x0, x1, p1t, p2t, g1t, g2t

    def fit(self, y2, d1, d2, x0, x1, p1t=None, p2t=None, g1t=None, g2t=None):
        """
        Fit dyntreatDML object

        Parameters
        ----------
        y2 : array-like of shape (n_samples,)
            The outcome variable.
        d1 : array-like of shape (n_samples,)
            The (observed) treatment in the first period. Array of integers.
        d2 : array-like of shape (n_samples,)
            The (observed) treatment in the second period. Array of integers.
        x0 : array-like of shape (n_samples, n_features_t0)
            Array of pre-treatment covariates.
        x1 : array-like of shape (n_samples, n_features_t1)
            Array of time-varying covariates.
        p1t : array-like of shape (n_samples,) or None
            Array of first-period propensity scores or None. If None,
            propensity scores are estimated. The default is None.
        p2t : array-like of shape (n_samples,) or None
            Array of second-period propensity scores or None. If None,
            propensity scores are estimated. The default is None.
        g1t : array-like of shape (n_samples,) or None
            Array of first-period counterfactual treatments or None. If None,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        g2t : array-like of shape (n_samples,) or None
            Array of second-period counterfactual treatments or None. If None,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        """
        # Check data & prepare inputs
        y2, d1, d2, x0, x1, p1t, p2t, g1t, g2t = self._prepare_fit(
            y2, d1, d2, x0, x1, p1t, p2t, g1t, g2t)
        self.g1t = g1t
        self.g2t = g2t
        # Get treatment dummies
        d1tre, d2tre = self._get_treatment_dummies(d1, d2)
        # Get predictions of nuisance functions
        if self.verbose:
            print("Scorestreat")
        y2d1d2te_tre, p1te_tre, p2te_tre, y1d1te_tre, tscores = self._nuisest(
            y2=y2,
            d1dum=d1tre,
            d2dum=d2tre,
            x0=x0,
            x1=x1,
            tkey='treat',
            p1_ext=p1t,
            p2_ext=p2t)
        self.psd1treat = p1te_tre
        self.psd2treat = p2te_tre
        self.d1tre = d1tre
        self.d2tre = d2tre
        self.y2d1d2treat = y2d1d2te_tre
        self.y1d1treat = y1d1te_tre
        self.tscores = tscores
        self.y2 = y2
        self.trimmed = np.isnan(tscores)
        self.is_fitted = True
        # return the output
        return self

    # TODO add GridSearchCV as option in addition to AutoML
    def tune_auto(
            self, y2, d1, d2, x0, x1, p1t=None, p2t=None, g1t=None, g2t=None):
        """
        Tune dyntreatDML object using AutoML

        Parameters
        ----------
        y2 : array-like of shape (n_samples,)
            The outcome variable.
        d1 : array-like of shape (n_samples,)
            The (observed) treatment in the first period. Array of integers.
        d2 : array-like of shape (n_samples,)
            The (observed) treatment in the second period. Array of integers.
        x0 : array-like of shape (n_samples, n_features_t0)
            Array of pre-treatment covariates.
        x1 : array-like of shape (n_samples, n_features_t1)
            Array of time-varying covariates.
        p1t : array-like of shape (n_samples,) or None
            Array of first-period propensity scores or None. If None,
            propensity scores are estimated. The default is None.
        p2t : array-like of shape (n_samples,) or None
            Array of second-period propensity scores or None. If None,
            propensity scores are estimated. The default is None.
        g1t : array-like of shape (n_samples,) or None
            Array of first-period counterfactual treatments or None. If None,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        g2t : array-like of shape (n_samples,) or None
            Array of second-period counterfactual treatments or None. If None,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        """
        # Check if methods are AutoML
        if not type(self.MLmethod_mu).__name__ in (
                'FlamlRegressor', 'FlamlClassifier'):
            raise ValueError(
                f'For tuning, '
                f'MLmethod_mu must be of type FlamlRegressor or Flaml'
                f'Classifier, got {type(self.MLmethod_mu).__name__}')
        if not type(self.MLmethod_nu).__name__ in (
                'FlamlRegressor', 'FlamlClassifier'):
            raise ValueError(
                f'For tuning, '
                f'MLmethod_nu must be of type FlamlRegressor or Flaml'
                f'Classifier, got {type(self.MLmethod_nu).__name__}')
        if not type(self.MLmethod_p1).__name__ in ('FlamlClassifier'):
            raise ValueError(
                f'For tuning, '
                f'MLmethod_p1 must be of type FlamlClassifier, got '
                f'{type(self.MLmethod_p1).__name__}')
        if not type(self.MLmethod_p2).__name__ in ('FlamlClassifier'):
            raise ValueError(
                f'For tuning, '
                f'MLmethod_p2 must be of type FlamlClassifier, got '
                f'{type(self.MLmethod_p2).__name__}')
        # Only one fold -> fit on whole data. since CV for tuning has same
        # number of folds as cross-fitting of dml -> same sample size
        self.nfolds = 1
        # Fit
        # Check data & prepare inputs
        y2, d1, d2, x0, x1, p1t, p2t, g1t, g2t = self._prepare_fit(
            y2, d1, d2, x0, x1, p1t, p2t, g1t, g2t)
        self.g1t = g1t
        self.g2t = g2t
        # Get treatment dummies
        d1tre, d2tre = self._get_treatment_dummies(d1, d2)
        # Get predictions of nuisance functions
        if self.verbose:
            print("Scorestreat")
        _, _, _, _, _ = self._nuisest(
            y2=y2,
            d1dum=d1tre,
            d2dum=d2tre,
            x0=x0,
            x1=x1,
            tkey='treat',
            p1_ext=p1t,
            p2_ext=p2t)
        # Get best estimators
        best_estimators = {}
        for nkey in self.MLmethod_dict.keys():
            best_estimators[nkey] = {
                'estimator': {}, 'score': {}}
            for tkey in self.MLmethod_dict[nkey].keys():
                if self.MLmethod_dict[nkey][
                        tkey].auto_ml.model is not None:
                    best_estimators[nkey]['estimator'][tkey] = (
                        self.MLmethod_dict[nkey][
                            tkey].auto_ml.model.estimator)
                    best_estimators[nkey]['score'][tkey] = (
                        self.MLmethod_dict[nkey][
                            tkey].auto_ml.best_loss)
        # Return result
        return best_estimators


class _stattreatDML(_BaseClass):
    """
    Base class for estimation of an average potential outcome under static
    confounding for one counterfactual sequence of interest.
    """

    # define init function
    def __init__(self,
                 d1treat,
                 d2treat,
                 MLmethod_p=RandomForestClassifier(random_state=999),
                 MLmethod_mu=RandomForestRegressor(random_state=999),
                 random_state=None,
                 nfolds=3,
                 verbose=True):
        """
        Initialize base class for estimation under static confounding.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        MLmethod_p : estimator implementing ``fit()`` and ``predict()``
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function :math:`p_{\\mathbf{d}_2}(X_0)`.
            If a classifier is used with method ``predict_proba()`` this
            method will be preferred over ``predict()``. Default is
            :py:class:`sklearn.ensemble.RandomForestClassifier`.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as :class:`FlamlClassifier`.
        MLmethod_mu : estimator implementing ``fit()`` and ``predict()``
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function
            :math:`\\mu_{\\mathbf{d}_2}({X}_0)`.
            If a classifier is used with method ``predict_proba()`` this
            method will be preferred over ``predict()``. Default is
            :py:class:`sklearn.ensemble.RandomForestRegressor`.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as :class:`FlamlRegressor`.
        random_state : int, RandomState instance or None
            Controls randomness of the sample splits in cross-fitting. The
            default is None.
        nfolds : int
            Number of folds for cross-fitting. The default is 3.
        verbose : bool
            Controls the verbosity when fitting and predicting. The default is
            True.

        Notes
        -----
        This method estimates the average potential outcome using static DML
        treating the sequence as a single treatment. For details see [1].

        References
        ----------
        .. [1] Muny, F. (2025). Evaluating Program Sequences with Double
               Machine Learning: An Application to Labor Market Policies.
               Manuscript in preparation.
        """
        super().__init__(
            d1treat=d1treat,
            d2treat=d2treat,
            random_state=random_state,
            nfolds=nfolds,
            verbose=verbose)
        self.MLmethod_p = MLmethod_p
        self.MLmethod_mu = MLmethod_mu
        self.is_static = True

    def _check_inputs(self):
        self._check_vartype(
            self.MLmethod_p, (BaseEstimator, dict), 'MLmethod_p')
        self._check_vartype(
            self.MLmethod_mu, (BaseEstimator, dict), 'MLmethod_mu')
        self._check_vartype(self.d1treat, (int, str), 'd1treat')
        self._check_vartype(self.d2treat, (int, str), 'd2treat')
        self._check_vartype(self.nfolds, int, 'nfolds')
        self._check_vartype(self.verbose, bool, 'verbose')
        # check whether seed is set (using scikitlearn check_random_state)
        self.random_state = check_random_state(self.random_state)

    def _get_treatment_dummies(self, d1, d2):
        # Create treatment indicators
        dtre = 1*(d1 == self.g1t)*(d2 == self.g2t)
        return dtre

    def _nuisest(self, y2, ddum, x0, tkey, p_ext=None):
        # Get folds
        stepsize = np.ceil((1/self.nfolds) * len(y2)).astype(int)
        nobs = min(self.nfolds * stepsize, len(y2))
        # initialize random state for sample splitting
        subsample_random_state = check_random_state(self.random_state)
        idx = subsample_random_state.choice(nobs, nobs, replace=False)
        # Get empty containers
        score = np.empty((0, 4))
        for i in range(1, self.nfolds+1):
            if self.verbose:
                print(f"Fold {i} of {self.nfolds}")
            # Get indices of test sample
            tesample = idx[((i-1)*stepsize):min((i*stepsize), nobs)]
            # Get indices of training sample
            if self.nfolds == 1:
                trsample = tesample
            else:
                trsample = idx[(~np.isin(idx, tesample))]
            # Get particulat subsets
            trsample_d = trsample[ddum[trsample] == 1]

            # Estimations:
            # 1) p1
            # p1 trained on full trsample, predicted on tesample
            if p_ext is not None:
                pte = p_ext[tesample]
            else:
                p = self.MLmethod_dict['p'][tkey].fit(
                        y=ddum[trsample], X=x0[trsample, :])
                if hasattr(p, 'predict_proba'):
                    pte = p.predict_proba(x0[tesample, :])[:, -1]
                else:
                    pte = p.predict(x0[tesample, :])
                # Return best params if tuning
                if self.best_params is not None:
                    self.best_params['p'][tkey] = getattr(p, 'best_params_')

            # 2) mu
            mu = self.MLmethod_dict['mu'][tkey].fit(
                    y=y2[trsample_d], X=x0[trsample_d, :])
            if hasattr(mu, 'predict_proba'):
                mute = mu.predict_proba(x0[tesample, :])[:, -1]
            else:
                mute = mu.predict(x0[tesample, :])
            # Return best params if tuning
            if self.best_params is not None:
                self.best_params['mu'][tkey] = getattr(mu, 'best_params_')
            # combine to one matrix
            res = np.concatenate([
                ddum[tesample].reshape(-1, 1),
                y2[tesample].reshape(-1, 1),
                mute.reshape(-1, 1),
                pte.reshape(-1, 1)], axis=1)
            # Add to score
            score = np.concatenate([score, res], axis=0)
        # Sort according to indices
        score = score[idx.argsort(), :]
        # Compute DR score
        cond = (score[:, 3] > 0)
        tscores = np.empty_like(score[:, 3])*np.nan
        tscores[cond] = score[cond, 2] + (score[cond, 0] * (
            score[cond, 1]-score[cond, 2]))/score[cond, 3]
        return score[:, 2], score[:, 3], tscores

    def _prepare_fit(
            self, y2, d1, d2, x0, pt=None, g1t=None, g2t=None):
        # Use sklearn input checks to allow for multiple types of inputs:
        # - returns numpy arrays for X and y (no matter which input type)
        # - makes sure dimensions are consistent
        # - forces y to be numeric
        x0, y2 = check_X_y(x0, y2, y_numeric=True)
        x0, d1 = check_X_y(x0, d1, y_numeric=True)
        x0, d2 = check_X_y(x0, d2, y_numeric=True)
        self.n_feat_x0 = x0.shape[1]
        # Check if propensity scores known (e.g. if estimation for different
        # outcome, propensity scores do not need to be re-estimated)
        # Either all p None or all p not None
        if pt is None:
            self.p_known = False
        else:
            self.p_known = True
            x0, pt = check_X_y(x0, pt, y_numeric=True)
            if (pt.max() > 1) or (pt.min() < 0):
                raise ValueError(
                    f'Provided propensity scores pt must be in interval [0, 1'
                    f'], got [{pt.min()}, {pt.max()}].')
        # Check other params
        self._check_inputs()
        # Check if treatment rule provided
        if g1t is None and g2t is None:
            self.g1_known = False
            self.g2_known = False
        if (g1t is not None):
            self.g1_known = True
            x0, g1t = check_X_y(x0, g1t, y_numeric=True)
            # Make sure that all values in g1t exist in d1
            not_in_d1 = np.unique(g1t)[~np.isin(np.unique(g1t), d1)]
            if (not_in_d1.size > 0):
                raise ValueError(
                    f'Values {not_in_d1} in g1t are not present in d1.')
        if (g2t is not None):
            self.g2_known = True
            x0, g2t = check_X_y(x0, g2t, y_numeric=True)
            # Make sure that all values in g2t exist in d2
            not_in_d2 = np.unique(g2t)[~np.isin(np.unique(g2t), d2)]
            if (not_in_d2.size > 0):
                raise ValueError(
                    f'Values {not_in_d2} in g1t are not present in d2.')
        # Check type of treatment provided
        if type(self.d1treat) is int:
            self.d1treat_provided_int = True
        else:
            self.d1treat_provided_int = False
        if type(self.d2treat) is int:
            self.d2treat_provided_int = True
        else:
            self.d2treat_provided_int = False
        if (not self.d1treat_provided_int) and (g1t is None):
            raise ValueError(
                'If d1treat provided as string, treatment indicator g1t must '
                'be provided to fit. Otherwise povide d1treat as int.')
        if (not self.d2treat_provided_int) and (g2t is None):
            raise ValueError(
                'If d2treat provided as string, treatment indicator g2t must '
                'be provided to fit. Otherwise povide d1treat as int.')
        if (self.d1treat_provided_int) and (g1t is not None):
            self.d1treat = str(self.d1treat)
            print(
                'Warning: d1treat provided as int while g1t is provided. Using'
                ' g1t for estimation; converting d1treat to string.')
        if (self.d2treat_provided_int) and (g2t is not None):
            self.d2treat = str(self.d2treat)
            print(
                'Warning: d2treat provided as int while g2t is provided. Using'
                ' g2t for estimation; converting d2treat to string.')
        # Check existence and unique values of treatments
        d1_unique = np.unique(d1)
        if (self.d1treat_provided_int) and (g1t is None) and (
                self.d1treat not in d1_unique):
            raise ValueError(f'd1treat = {self.d1treat} not found in d1.')
        d2_unique = np.unique(d2)
        if (self.d2treat_provided_int) and (g2t is None) and (
                self.d2treat not in d2_unique):
            raise ValueError(f'd2treat = {self.d2treat} not found in d2.')
        # If g1t and g2t not provided, create them from d1treat and d2treat
        if g1t is None:
            g1t = np.ones_like(d1)*self.d1treat
        if g2t is None:
            g2t = np.ones_like(d2)*self.d2treat
        # create dictionary of methods
        self.MLmethod_dict = {'p': {}, 'mu': {}}
        self.MLmethod_dict['p']['treat'] = clone(self.MLmethod_p)
        self.MLmethod_dict['mu']['treat'] = clone(self.MLmethod_mu)
        return y2, d1, d2, x0, pt, g1t, g2t

    def fit(self, y2, d1, d2, x0, pt=None, g1t=None, g2t=None):
        """
        Fit stattreatDML object

        Parameters
        ----------
        y2 : array-like of shape (n_samples,)
            The outcome variable.
        d1 : array-like of shape (n_samples,)
            The (observed) treatment in the first period. Array of integers.
        d2 : array-like of shape (n_samples,)
            The (observed) treatment in the second period. Array of integers.
        x0 : array-like of shape (n_samples, n_features_t0)
            Array of pre-treatment covariates.
        pt : array-like of shape (n_samples,) or None
            Array of propensity scores or None. If None,
            propensity scores are estimated. The default is None.
        g1t : array-like of shape (n_samples,) or None
            Array of first-period counterfactual treatments or None. If None,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        g2t : array-like of shape (n_samples,) or None
            Array of second-period counterfactual treatments or None. If None,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        """
        # Check data & prepare inputs
        y2, d1, d2, x0, pt, g1t, g2t = self._prepare_fit(
            y2, d1, d2, x0, pt, g1t, g2t)
        self.g1t = g1t
        self.g2t = g2t
        # Get treatment dummies
        dtre = self._get_treatment_dummies(d1, d2)
        # Get predictions of nuisance functions
        if self.verbose:
            print("Scorestreat")
        mute_tre, pte_tre, tscores = (
            self._nuisest(
                y2=y2,
                ddum=dtre,
                x0=x0,
                tkey='treat',
                p_ext=pt))
        self.psdtreat = pte_tre
        self.dtre = dtre
        self.mutreat = mute_tre
        self.tscores = tscores
        self.y2 = y2
        self.trimmed = np.isnan(tscores)
        self.is_fitted = True
        # return the output
        return self

    # TODO add GridSearchCV
    def tune_auto(self, y2, d1, d2, x0, pt=None, g1t=None, g2t=None):
        """
        Tune stattreatDML object using AutoML

        Parameters
        ----------
        y2 : array-like of shape (n_samples,)
            The outcome variable.
        d1 : array-like of shape (n_samples,)
            The (observed) treatment in the first period. Array of integers.
        d2 : array-like of shape (n_samples,)
            The (observed) treatment in the second period. Array of integers.
        x0 : array-like of shape (n_samples, n_features_t0)
            Array of pre-treatment covariates.
        pt : array-like of shape (n_samples,) or None
            Array of propensity scores or None. If None,
            propensity scores are estimated. The default is None.
        g1t : array-like of shape (n_samples,) or None
            Array of first-period counterfactual treatments or None. If None,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        g2t : array-like of shape (n_samples,) or None
            Array of second-period counterfactual treatments or None. If None,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        """
        # Check if methods are AutoML
        if not type(self.MLmethod_mu).__name__ in (
                'FlamlRegressor', 'FlamlClassifier'):
            raise ValueError(
                f'MLmethod_mu must be of type FlamlRegressor or Flaml'
                f'Classifier, got {type(self.MLmethod_mu).__name__}')
        if not type(self.MLmethod_p).__name__ in ('FlamlClassifier'):
            raise ValueError(
                f'MLmethod_p must be of type FlamlClassifier, got '
                f'{type(self.MLmethod_p).__name__}')
        # Only one fold -> fit on whole data. since CV for tuning has same
        # number of folds as cross-fitting of dml -> same sample size
        self.nfolds = 1
        # Check data & prepare inputs
        y2, d1, d2, x0, pt, g1t, g2t = self._prepare_fit(
            y2, d1, d2, x0, pt, g1t, g2t)
        self.g1t = g1t
        self.g2t = g2t
        # Get treatment dummies
        dtre = self._get_treatment_dummies(d1, d2)
        # Get predictions of nuisance functions
        if self.verbose:
            print("Scorestreat")
        _, _, _ = (
            self._nuisest(
                y2=y2,
                ddum=dtre,
                x0=x0,
                tkey='treat',
                p_ext=pt))
        # Get best estimators
        best_estimators = {}
        for nkey in self.MLmethod_dict.keys():
            best_estimators[nkey] = {'estimator': {}, 'score': {}}
            for tkey in self.MLmethod_dict[nkey].keys():
                if self.MLmethod_dict[nkey][
                        tkey].auto_ml.model is not None:
                    best_estimators[nkey]['estimator'][tkey] = (
                        self.MLmethod_dict[
                            nkey][tkey].auto_ml.model.estimator)
                    best_estimators[nkey]['score'][tkey] = (
                        self.MLmethod_dict[
                            nkey][tkey].auto_ml.best_loss)
        # Return result
        return best_estimators
