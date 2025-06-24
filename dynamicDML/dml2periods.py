import numpy as np
import pandas as pd
import os
import gzip
import mgzip
import pickle
import copy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.validation import check_array
from sklearn.utils import check_X_y
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from dynamicDML._base import _dyntreatDML, _stattreatDML


class dml2periods:
    def __init__(
            self,
            dynamic_confounding=True,
            random_state=None,
            nfolds=3,
            p2_on_d1_subset=False,
            extra_split_train_mu=True,
            extra_split_train_nu=True,
            crossfit_nu=False,
            doubly_robust_nu=False,
            verbose=True):
        """
        Estimation of average treatment effects under static or dynamic
        confounding for a wide range of counterfactual sequences.

        Parameters
        ----------
        dynamic_confounding : bool
            Whether to assume dynamic confounding. If `False` static
            confounding is assumed, treating sequences as single treatment
            states. The default is `True`.
        random_state : int, RandomState instance or None
            Controls randomness of the sample splits in cross-fitting. The
            default is `None`.
        nfolds : int
            Number of folds for cross-fitting. The default is 3.
        p2_on_d1_subset : bool
            This parameter regulates if we train \\(p_{d_2}\\) stratified on
            the subset \\(D_1=d_1\\) (`True`) as implemented in
            Bradic et al.[^Bradic2] or on
            the full sample with \\(D_1\\) as a regressor (`False`) as
            implemented in Bodory et al.[^Bodory2] The stratification should
            reduce bias in
            the estimates of \\(p_{d_2}\\) but may increase variance if there
            are few individuals following a particular program \\(d_1\\).
            The default is `False`.
        extra_split_train_mu : bool
            Whether the \\(\\mu\\) that directly enters the DML score should be
            estimated on the full (``False``) or only half of the training
            sample (``True``). If ``extra_split_train_nu=True``, setting
            ``extra_split_train_mu=True`` is computationally faster as
            \\(\\mu\\) does
            not need to be re-estimated on one half of the training sample
            later. The default is `True`.
        extra_split_train_nu : bool
            Whether to perform an additional split of the training sample for
            the estimation of \\(\\nu\\). This is recommended to avoid
            overfitting as
            the estimation of \\(\\nu\\) takes estimates of \\(\\mu\\) as a
            pseudo outcome. The default is `True`.
        crossfit_nu : bool
            If ``extra_split_train_nu=True``, cross-fitting allows to regain
            full-sample efficiency by reversing the roles of the subsamples
            created to estimate \\(\\mu\\) and \\(\\nu\\), respectively.
            However, this leads
            to higher computation time. The default is `False`.
        doubly_robust_nu : bool
            This parameter indicates whether \\(\\nu\\) should be estimated in
            a doubly-robust manner as proposed in Bradic et al.[^Bradic2]
            (``True``). Otherwise, it
            is estimated by regressing \\(\\mu\\) on pre-treatment covariates
            as proposed in Bodory et al.[^Bodory2] (``False``). The former
            is computationally more expensive. The default is `False`.
        verbose : bool
            Controls the verbosity when fitting and predicting. The default is
            `True`.

        Details
        -------
        When `dynamic_confounding=True`, this method estimates the average
        potential outcome according to the DML-formula


        \\[ \\hat\\theta^{\\mathbf{d}_2} = \\frac{1}{N} \\sum_{i=1}^N
            \\hat{\\nu}_{\\mathbf{d}_2}(X_{0i})
            + \\frac{\\left(\\hat{\\mu}_{\\mathbf{d}_2}(\\mathbf{X}_{1i})  -
            \\hat{\\nu}_{\\mathbf{d}_2}(X_{0i})  \\right) \\cdot
            \\mathbf{1}\\{D_{1i}=d_1\\} }{\\hat{p}_{d_1}(X_{0i})}\\\\
            + \\frac{\\left(Y_i -  \\hat{\\mu}_{\\mathbf{d}_2}
            (\\mathbf{X}_{1i})\\right)
            \\cdot \\mathbf{1}\\{\\mathbf{D}_{2i}=\\mathbf{d}_2
            \\}}{\\hat{p}_{d_2}
            (\\mathbf{X}_{1i}, d_{1})\\hat{p}_{d_1}(X_{0i})} \\]

        For details and a generalization to dynamic policies see
        Muny (2025).[^Muny2]

        When estimating counterfactual outcomes of treatment sequences we
        typically want to make an extra split of the training sample to avoid
        overfitting in the estimation of \\({\\nu}_{\\mathbf{d}_2}\\),
        regulated by
        `extra_split_tain_nu`. The estimate of \\({\\mu}_{\\mathbf{d}_2}
        (\\mathbf{X}_{1i})\\) is needed for two things:

        1. as prediction to enter the dynamic DML formula
        2. as prediction to be used as outcome in the estimation of
           \\(\\hat{\\nu}_{\\mathbf{d}_2}\\)

        For (1.) the extra split is not neccessary but it can
        computationally more convenient re-use the
        \\(\\hat{\\mu}_{\\mathbf{d}_2}(\\mathbf{X}_{1i})\\)
        estimated to get
        \\(\\hat{\\nu}_{\\mathbf{d}_2}\\),
        instead of additionally fitting
        \\(\\hat{\\mu}_{\\mathbf{d}_2}(\\mathbf{X}_{1i})\\)
        on the full training sample. This is
        regulated with the paramter ``extra_split_tain_mu``. If it is ``True``,
        \\(\\hat{\\mu}_{\\mathbf{d}_2}(\\mathbf{X}_{1i})\\)
        is trained only on one half of the training sample. Then
        this trained estimate can later be re-used for
        \\(\\hat{\\nu}_{\\mathbf{d}_2}\\).
        If it is `False`,
        \\(\\hat{\\mu}_{\\mathbf{d}_2}(\\mathbf{X}_{1i})\\)
        is trained on the whole training sample.
        Then, if we want an additional split for
        \\(\\hat{\\nu}_{\\mathbf{d}_2}\\)
        (i.e. `extra_split_train_nu=True`),
        \\(\\hat{\\mu}_{\\mathbf{d}_2}(\\mathbf{X}_{1i})\\)
        has to be re-estimated on one half of the training sample later.

        To recover the estimates from the original papers, use the following
        settings, where ``Bodory`` and ``fewsplits`` are propsed in Bodory et
        al.[^Bodory2], and ``D-DRL``, ``DTL``, and ``D-DRL'`` are proposed in
        Bradic et al.[^Bradic2]:

        | Overview             | Bodory | fewspl. | D-DRL | DTL   | D-DRL' |
        | -------------------- | ------ | ------- | ----- | ----- | ------ |
        | p2_on_d1_subset      | False  | True    | True  | True  | True   |
        | extra_split_train_mu | True   | False   | False | False | False  |
        | extra_split_train_nu | True   | True    | True  | False | False  |
        | crossfit_nu          | False  | False   | True  | False | False  |
        | doubly_robust_nu     | False  | False   | True  | False | True   |

        The settings from Bodory et al.[^Bodory2] are used as the default.

        When `dynamic_confounding=False`, each treatment sequence is considered
        to ba a single treatment state and estimation proceeds with standard
        single-period DML as originally proposed in Chernozhukov et al. (2018).
        [^Chern2]

        References
        ----------
        [^Bradic2]:
            Bradic, J., Ji, W., & Zhang, Y. (2024). High-dimensional inference
            for dynamic treatment effects. The Annals of Statistics, 52(2),
            415–440.
        [^Bodory2]:
            Bodory, H., Huber, M., & Lafférs, L. (2022). Evaluating (weighted)
            dynamic treatment effects by double machine learning. The
            Econometrics Journal, 25(3), 628–648.
        [^Muny2]:
            Muny, F. (2025). Evaluating Program Sequences with Double Machine
            Learning: An Application to Labor Market Policies. arXiv preprint arXiv:2506.11960.
        [^Chern2]:
            Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
            C., Newey, W., & Robins, J. (2018). Double/debiased machine
            learning for treatment and structural parameters.
        """
        self.dynamic_confounding = dynamic_confounding
        self.random_state = random_state
        self.nfolds = nfolds
        self.p2_on_d1_subset = p2_on_d1_subset
        self.extra_split_train_mu = extra_split_train_mu
        self.extra_split_train_nu = extra_split_train_nu
        self.crossfit_nu = crossfit_nu
        self.doubly_robust_nu = doubly_robust_nu
        self.verbose = verbose
        # Initialize sequences
        self.sequences = {}
        # Containers for results
        self.APO = {}
        self.ATE = {}
        self.GATEmATE = {}
        self.GAPOmAPO = {}

    def _check_vartype(self, var, typ, varname):
        if not isinstance(var, typ):
            raise ValueError(
                f'{varname} must be of type {typ.__name__}, got '
                f'{type(var).__name__}')

    def _quantile_fixed(self, p):
        return lambda x: np.quantile(x, p)

    def init_sequence(
            self,
            d1treat,
            d2treat,
            MLmethod_p1=RandomForestClassifier(random_state=999),
            MLmethod_p2=RandomForestClassifier(random_state=999),
            MLmethod_mu=RandomForestRegressor(random_state=999),
            MLmethod_nu=RandomForestRegressor(random_state=999),
            replace=False):
        """
        Initialize a counterfactual treatment sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        MLmethod_p1 : estimator implementing fit() and predict()
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function \\(p_{d_1}(X_0)\\).
            If a classifier is used with method ``predict_proba()`` this
            method will be preferred over ``predict()``. Default is
            [`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html){:target="_blank"}.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as `FlamlClassifier`.
        MLmethod_p2 : estimator implementing fit() and predict()
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function
            \\(p_{d_2}(\\mathbf{X}_1,d_1)\\).
            If a classifier is used with method ``predict_proba()`` this
            method will be preferred over ``predict()``. Default is
            [`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html){:target="_blank"}.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as `FlamlClassifier`.
        MLmethod_mu : estimator implementing fit() and predict()
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function
            \\(\\mu_{\\mathbf{d}_2}(\\mathbf{X}_1)\\).
            If a classifier is used with method ``predict_proba()`` this
            method will be preferred over ``predict()``. Default is
            [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html){:target="_blank"}.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as `FlamlRegressor`.
        MLmethod_nu : estimator implementing fit() and predict()
            A machine learner implementing ``fit()`` and ``predict()`` methods
            for the nuisance function \\(\\nu_{\\mathbf{d}_2}(X)\\).
            Default is
            [`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html){:target="_blank"}.
            If nuisance estimators should be tuned with with AutoML, the
            estimator needs to be specified as `FlamlRegressor`.
        replace : bool
            Whether to re-initialize the sequence if it already exists. This
            will delete any previous information for this sequence. The
            default is `False`.
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        if (type(d1treat) is str):
            if "_" in d1treat:
                raise ValueError(
                    "If d1treat provided as string it may not contain '_'.")
        if (type(d2treat) is str):
            if "_" in d2treat:
                raise ValueError(
                    "If d1treat provided as string it may not contain '_'.")
        # Get sequence name: d1treat_d2treat
        seq_name = str(d1treat) + '_' + str(d2treat)
        # Check if sequence exists already
        if not replace and (seq_name in self.sequences):
            raise ValueError(
                f'Sequence with d1treat={d1treat} and d2treat={d2treat} '
                f'already exists. To overwrite set replace=True.')
        if self.dynamic_confounding:
            self.sequences[seq_name] = _dyntreatDML(
                d1treat=d1treat,
                d2treat=d2treat,
                MLmethod_p1=MLmethod_p1,
                MLmethod_p2=MLmethod_p2,
                MLmethod_mu=MLmethod_mu,
                MLmethod_nu=MLmethod_nu,
                random_state=self.random_state,
                nfolds=self.nfolds,
                p2_on_d1_subset=self.p2_on_d1_subset,
                extra_split_train_mu=self.extra_split_train_mu,
                extra_split_train_nu=self.extra_split_train_nu,
                crossfit_nu=self.crossfit_nu,
                doubly_robust_nu=self.doubly_robust_nu,
                verbose=self.verbose)
        else:
            print("Static sequence initialized. The following dynamic "
                  "parameters are ignored:")
            print("MLmethod_p2, MLmethod_nu, p2_on_d1_subset, "
                  "extra_split_train_mu, extra_split_train_nu, crossfit_nu, "
                  "doubly_robust_nu")
            self.sequences[seq_name] = _stattreatDML(
                d1treat=d1treat,
                d2treat=d2treat,
                MLmethod_p=MLmethod_p1,
                MLmethod_mu=MLmethod_mu,
                random_state=self.random_state,
                nfolds=self.nfolds,
                verbose=self.verbose)
        self.sequences[seq_name].is_tuned = False
        return self

    def sequence_summary(self, methods=False):
        """
        Print a summary of the initialized sequences.

        Parameters
        ----------
        methods : bool
            Whether to print additional information about estimators. The
            default is `False`.
        """
        self._check_vartype(methods, bool, 'methods')
        print("-"*60)
        print("Summary of initialized sequences:")
        print("-"*60)
        for key in self.sequences.keys():
            fst = self.sequences[key].d1treat
            snd = self.sequences[key].d2treat
            ftd = self.sequences[key].is_fitted
            tnd = self.sequences[key].is_tuned
            print(f"d1treat: {fst}, d2treat: {snd}, is_tuned: {tnd}, "
                  f"is_fitted: {ftd}")
            if methods:
                if self.dynamic_confounding:
                    print(f" MLmethod_p1: {self.sequences[key].MLmethod_p1}")
                    print(f" MLmethod_p2: {self.sequences[key].MLmethod_p2}")
                    print(f" MLmethod_mu: {self.sequences[key].MLmethod_mu}")
                    print(f" MLmethod_nu: {self.sequences[key].MLmethod_nu}")
                else:
                    print(f" MLmethod_p: {self.sequences[key].MLmethod_p}")
                    print(f" MLmethod_mu: {self.sequences[key].MLmethod_mu}")
        if (len(list(self.sequences.keys())) == 0):
            print("No sequences initialized.")
        print("-"*60)

    def drop_sequence(self, d1treat, d2treat):
        """
        Delete an initialized treatment sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        # Get sequence name: d1treat_d2treat
        seq_name = str(d1treat) + '_' + str(d2treat)
        # Check if sequence exists already
        if not (seq_name in self.sequences):
            raise ValueError(
                f'Sequence with d1treat={d1treat} and d2treat={d2treat} '
                f'does not exist.')
        del self.sequences[seq_name]
        return self

    def fit_sequence(
            self, d1treat, d2treat, y2, d1, d2, x0, x1=None, p1t=None,
            p2t=None, g1t=None, g2t=None):
        """
        Fit an initialized treatment sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        y2 : array-like of shape (n_samples,)
            The outcome variable.
        d1 : array-like of shape (n_samples,)
            The (observed) treatment in the first period. Array of integers.
        d2 : array-like of shape (n_samples,)
            The (observed) treatment in the second period. Array of integers.
        x0 : array-like of shape (n_samples, n_features_t0)
            Array of pre-treatment covariates.
        x1 : array-like of shape (n_samples, n_features_t1) or None
            Array of time-varying covariates or `None` if static confounding.
            The default is `None`.
        p1t : array-like of shape (n_samples,) or None
            Array of first-period propensity scores or `None`. If `None`,
            propensity scores are estimated. The default is `None`.
        p2t : array-like of shape (n_samples,) or None
            Array of second-period propensity scores or `None`. If `None`,
            propensity scores are estimated. The default is `None`.
        g1t : array-like of shape (n_samples,) or None
            Array of first-period counterfactual treatments or `None`. If
            `None`,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        g2t : array-like of shape (n_samples,) or None
            Array of second-period counterfactual treatments or `None`. If
            `None`,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        # Get sequence name: d1treat_d2treat
        seq_name = str(d1treat) + '_' + str(d2treat)
        # Check if sequence exists already
        if not (seq_name in self.sequences):
            raise ValueError(
                f'Sequence with d1treat={d1treat} and d2treat={d2treat} '
                f'does not exist.')
        if self.dynamic_confounding:
            self.sequences[seq_name] = self.sequences[seq_name].fit(
                y2, d1, d2, x0, x1, p1t=p1t, p2t=p2t, g1t=g1t, g2t=g2t)
        else:
            if x1 is not None:
                print("x1 set to None since static sequence")
            if p2t is not None:
                print("p2t set to None since static sequence")
            self.sequences[seq_name] = self.sequences[seq_name].fit(
                y2, d1, d2, x0, pt=p1t, g1t=g1t, g2t=g2t)
            # Make compatible with dynamic functions
            self.sequences[seq_name].d1tre = self.sequences[seq_name].dtre
            self.sequences[seq_name].d2tre = self.sequences[seq_name].dtre*0
            self.sequences[seq_name].psd1treat = self.sequences[
                seq_name].psdtreat
            self.sequences[seq_name].psd2treat = np.ones_like(self.sequences[
                seq_name].psdtreat)
            self.sequences[seq_name].y2d1d2treat = self.sequences[seq_name].y2
            self.sequences[seq_name].y1d1treat = self.sequences[
                seq_name].mutreat
        return self

    def tune_auto_sequence(
            self, d1treat, d2treat, y2, d1, d2, x0, x1=None, p1t=None,
            p2t=None, g1t=None, g2t=None):
        """
        Tune initialized treatment sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        y2 : array-like of shape (n_samples,)
            The outcome variable.
        d1 : array-like of shape (n_samples,)
            The (observed) treatment in the first period. Array of integers.
        d2 : array-like of shape (n_samples,)
            The (observed) treatment in the second period. Array of integers.
        x0 : array-like of shape (n_samples, n_features_t0)
            Array of pre-treatment covariates.
        x1 : array-like of shape (n_samples, n_features_t1) or None
            Array of time-varying covariates or `None` if static confounding.
            The default is `None`.
        p1t : array-like of shape (n_samples,) or None
            Array of first-period propensity scores or `None`. If `None`,
            propensity scores are estimated. The default is `None`.
        p2t : array-like of shape (n_samples,) or None
            Array of second-period propensity scores or `None`. If `None`,
            propensity scores are estimated. The default is `None`.
        g1t : array-like of shape (n_samples,) or None
            Array of first-period counterfactual treatments or `None`. If
            `None`,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        g2t : array-like of shape (n_samples,) or None
            Array of second-period counterfactual treatments or `None`. If
            `None`,
            ``d1treat`` will be used as counterfactual for all individuals.
            If array provided, this will be used as counterfactual and
            ``d1treat`` serves only as a name of the counterfactual. Can be
            used to implement dynamic policies.
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        # Get sequence name: d1treat_d2treat
        seq_name = str(d1treat) + '_' + str(d2treat)
        # Check if sequence exists already
        if not (seq_name in self.sequences):
            raise ValueError(
                f'Sequence with d1treat={d1treat} and d2treat={d2treat} '
                f'does not exist.')
        # Create deepcopy
        to_tune = copy.deepcopy(self.sequences[seq_name])
        if self.dynamic_confounding:
            to_tune = to_tune.tune_auto(
                y2, d1, d2, x0, x1, p1t=p1t, p2t=p2t, g1t=g1t, g2t=g2t)
            if p1t is None:
                self.sequences[seq_name].MLmethod_p1 = to_tune['p1'][
                    'estimator']['treat']
            else:
                self.sequences[seq_name].MLmethod_p1 = None
            if p2t is None:
                self.sequences[seq_name].MLmethod_p2 = to_tune['p2'][
                    'estimator']['treat']
            else:
                self.sequences[seq_name].MLmethod_p2 = None
            self.sequences[seq_name].MLmethod_mu = to_tune['mu']['estimator'][
                'treat']
            self.sequences[seq_name].MLmethod_nu = to_tune['nu']['estimator'][
                'treat']
        else:
            to_tune = to_tune.tune_auto(
                y2, d1, d2, x0, pt=p1t, g1t=g1t, g2t=g2t)
            if p1t is None:
                self.sequences[seq_name].MLmethod_p = to_tune['p'][
                    'estimator']['treat']
            else:
                self.sequences[seq_name].MLmethod_p = None
            self.sequences[seq_name].MLmethod_mu = to_tune['mu']['estimator'][
                'treat']
        self.sequences[seq_name].tune_result = to_tune
        self.sequences[seq_name].is_tuned = True
        return self

    def export_sequence(self, d1treat, d2treat, path, threads=1):
        """
        Export sequence object as compressed gzip file.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        path : str
            Path of directory where file should be saved. File will be saved
            with name ``d1treat_d2treat.gz`` in provided directory.
        threads : int
            Number of threads to be used to compress. Zero means using all
            CPUs. For ``threads=1`` the basic gzip module is used instead of
            the experimental mgzip module. The default is 1.
        """
        # Check v ar types
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        self._check_vartype(path, str, 'path')
        # Get sequence name: d1treat_d2treat
        seq_name = str(d1treat) + '_' + str(d2treat)
        # Check if sequence exists already
        if not (seq_name in self.sequences):
            raise ValueError(
                f'Sequence with d1treat={d1treat} and d2treat={d2treat} '
                f'does not exist.')
        # Create path if it does not exist yet
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + "/" + seq_name + ".gz"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if threads == 1:
            with gzip.open(filename, "wb") as f:
                pickle.dump(self.sequences[seq_name], f)
        else:
            with mgzip.open(filename, "wb", thread=threads) as f:
                pickle.dump(self.sequences[seq_name], f)
        return self

    def import_sequence(
            self, d1treat, d2treat, path, filename=None, threads=1):
        """
        Import sequence object from compressed gzip file.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        path : str
            Path of directory where file to import is saved.
        filename : str or None
            Name of file to import. Should have .gz extension. If `None`, the
            filename 'd1treat_d2treat' is assumed.
        threads : int
            Number of threads to be used to decompress. Zero means using all
            CPUs. For ``threads=1`` the basic gzip module is used instead of
            the experimental mgzip module. The default is 1.
        """
        # Check v ar types
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        self._check_vartype(path, str, 'path')
        # Get sequence name: d1treat_d2treat
        seq_name = str(d1treat) + '_' + str(d2treat)
        if filename is None:
            filename = seq_name + ".gz"
        else:
            self._check_vartype(filename, str, 'filename')
        if threads == 1:
            with gzip.open(path + "/" + filename, "rb") as f:
                imported_sequence = pickle.load(f)
        else:
            with mgzip.open(path + "/" + filename, "rb", thread=threads) as f:
                imported_sequence = pickle.load(f)
        typ = str(type(imported_sequence))
        pos_dot = typ.rfind(".") + 1
        pos_quote = typ.rfind("'")
        typname = typ[pos_dot:pos_quote]
        if typname != '_dyntreatDML' and typname != '_stattreatDML':
            raise ValueError(
                f'Imported sequence must be of type _dyntreatDML or '
                f'_stattreatDML, got {typname}')
        self.sequences[seq_name] = imported_sequence
        return self

    def _check_trim(self, trim, n):
        # 3 options
        # - int: fixed trimming threshold, e.g. trim als pscores below 0.01
        # - tuple: relative minmax quantile rule, e.g. (0, 1) = minmax
        # - list/array: booleans indicating who to trim
        # - string no longer allowed!
        if isinstance(trim, tuple):
            trim_str = 'tuple'
            if len(trim) != 2:
                raise ValueError(
                    f"If trim given as tuple it must be of length 2, "
                    f"got length {len(trim)}.")
            if (trim[0] < 0) or (trim[0] > 1):
                raise ValueError(
                    f"If trim given as tuple, each element must be in interval"
                    f" between 0 and 1, got an element {trim[0]}.")
            if (trim[1] < 0) or (trim[1] > 1):
                raise ValueError(
                    f"If trim given as tuple, each element must be in interval"
                    f" between 0 and 1, got an element {trim[1]}.")
            if (trim[1] <= trim[0]):
                raise ValueError(
                    "If trim given as tuple, first element must be strictly "
                    "smaller than second element.")
        elif isinstance(trim, (int, float)):
            trim_str = 'numeric'
            # Make sure that trim in [0, 1)
            if (trim < 0) or (trim >= 1):
                raise ValueError(
                    f'trim must be in interval [0, 1), got {trim}')
        else:
            trim_str = 'array'
            _, trimmed = check_X_y(trim, trim, ensure_2d=False)
            if len(trimmed) != n:
                raise ValueError(
                    f"If trim given as array-type, its length must be the "
                    f"number of observations ({n}), "
                    f"got length {len(trim)}.")
            if trimmed.dtype != 'bool':
                raise ValueError(
                    "If trim provided as array-type, each element must be of "
                    "type bool.")
        return trim_str

    def _check_zero_pscores(
            self, trimmed, p1te_tre, p2te_tre, p1te_con=None, p2te_con=None):
        # Make compatible
        tre_only = True if p1te_con is None else False
        p1te_con = np.ones_like(p1te_tre) if p1te_con is None else p1te_con
        p2te_con = np.ones_like(p2te_tre) if p2te_con is None else p2te_con
        # Check for zero propensity scores
        p_0 = 1*((p1te_tre == 0) | (p2te_tre == 0) | (p1te_con == 0) | (
            p2te_con == 0))
        # Check if there are observations with zero propensity scores that
        # are not trimmed
        n_ps_0 = p_0.sum()
        n_ps_0_notrim = ((trimmed == 0) & p_0).sum()
        # Print warning if zero propensity scores
        if n_ps_0 > 0:
            p1te_tre[p1te_tre == 0] = np.nan
            p2te_tre[p2te_tre == 0] = np.nan
            p1te_con[p1te_con == 0] = np.nan
            p2te_con[p2te_con == 0] = np.nan
            trimmed[p_0 == 1] = 1
            # Set to NA such that there is no warning for division by zero
            print(f"WARNING: The propensity scores of {n_ps_0} "
                  f"observations are equal to zero and have been set to "
                  f"NA. {n_ps_0-n_ps_0_notrim} of these observations are "
                  f"trimmed according to the specified trimming rule. "
                  f"{n_ps_0_notrim} of these observations are trimmed "
                  f"even "
                  f"though they should not have been trimmed according to "
                  f"the specified trimming rule.")
        p1te_con = None if tre_only else p1te_con
        p2te_con = None if tre_only else p2te_con
        return (
            trimmed, p1te_tre, p2te_tre, p1te_con, p2te_con, n_ps_0,
            n_ps_0_notrim)

    def _summary_APO(self, result):
        self._check_vartype(result, dict, 'result')
        groupinfo = ""
        if result['name_groupvar'] is not None:
            groupinfo = ', by ' + result['name_groupvar']
        print("="*80)
        print(f"Estimation result for sequence {result['d1treat']}-"
              f"{result['d2treat']}{groupinfo}:")
        print("-"*80)
        summary = pd.DataFrame(index=[
                'coef', 'std err', 't', 'P>|t|', '2.5%', '97.5%', 'N',
                'Ntrim'], columns=result['results'].keys()).T
        for cat in result['results']:
            summary.loc[cat, 'coef'] = (
                f"{result['results'][cat]['meantreat']:.3f}")
            summary.loc[cat, 'std err'] = (
                f"{result['results'][cat]['se']:.3f}")
            tval = result['results'][cat]['meantreat']/result[
                'results'][cat]['se']
            summary.loc[cat, 't'] = (
                f"{tval:.3f}")
            summary.loc[cat, 'P>|t|'] = (
                f"{result['results'][cat]['pval']:.3f}")
            summary.loc[cat, '2.5%'] = (
                f"{result['results'][cat]['ci'][0][0]:.3f}")
            summary.loc[cat, '97.5%'] = (
                f"{result['results'][cat]['ci'][0][1]:.3f}")
            summary.loc[cat, 'N'] = (
                f"{result['results'][cat]['N']}")
            summary.loc[cat, 'Ntrim'] = (
                f"{result['results'][cat]['Ntrim']:.0f}")
        print(summary)
        print("="*80)

    def compute_APO(
            self, d1treat, d2treat, groupvar=None, name_groupvar=None, trim=0):
        """
        Compute (group) average potential outcome (APO) from fitted scores of a
        particular sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        groupvar : array-like of shape (n_samples,) or None
            Discrete group variable. If provided, APO is computed within
            each group. If `None`, the APO is computed for the whole sample.
            The default is `None`.
        name_groupvar : str or None
            The name of the group variable. Needs to be provided only if
            ``groupvar`` is provided. The default is `None`.
        trim : int, float or array-like of shape (n_samples,)
            The trimming variable can be provided in these ways:

            * ``int`` or ``float``: A fixed trimming threshold, e.g. trim
              all propensity scores below 0.01.
            * ``array-like``: A bool array indicating which observations to
              trim (observations with ``True`` are trimmed)

            The default is 0 (no trimming).
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        seq_name_treat = str(d1treat) + '_' + str(d2treat)
        seq_name_group = seq_name_treat
        if (groupvar is not None) and (name_groupvar is None):
            raise ValueError(
                'If groupvar supplied, name_groupvar must be supplied as'
                ' well.')
        if (groupvar is None) and (name_groupvar is not None):
            name_groupvar = None
            print('Warning: name_groupvar set to None as no groupvar supplied')
        if name_groupvar is not None:
            self._check_vartype(name_groupvar, str, 'name_groupvar')
            seq_name_group = seq_name_treat + '_' + name_groupvar
        # Check if sequence exists already
        if not (seq_name_treat in self.sequences):
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} does not exist.')
        # Check if fitted
        if not self.sequences[seq_name_treat].is_fitted:
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} is not fitted.')
        # Check groups
        if groupvar is not None:
            z0a = check_array(groupvar, ensure_2d=False, dtype=str)
            # Check if same length as dml_score
            if len(z0a) != len(self.sequences[seq_name_treat].tscores):
                raise ValueError(
                    f'Group indicator must have length '
                    f'{len(self.sequences[seq_name_treat].tscores)}, '
                    f'got {len(z0a)}.')
        else:
            name = "APO"
            z0a = np.array([name]*len(self.sequences[
                seq_name_treat].tscores))
        # Get unique values in z0
        z0a_unique = np.unique(z0a)
        # Check trim argument
        trim_str = self._check_trim(trim, n=len(
            self.sequences[seq_name_treat].y2))
        # Get inputs
        p1te_tre = self.sequences[seq_name_treat].psd1treat
        p2te_tre = self.sequences[seq_name_treat].psd2treat
        d1tre = self.sequences[seq_name_treat].d1tre
        d2tre = self.sequences[seq_name_treat].d2tre
        y2d1d2te_tre = self.sequences[seq_name_treat].y2d1d2treat
        y1d1te_tre = self.sequences[seq_name_treat].y1d1treat
        y2 = self.sequences[seq_name_treat].y2
        # Get indicator for trimmed observations
        if trim_str == 'array':
            _, trimmed = check_X_y(trim, trim, ensure_2d=False)
            trimmed = 1*trimmed
        else:
            trimmed = 1*((p1te_tre*p2te_tre) < trim)
        # Check for zero propensity scores
        trimmed, p1te_tre, p2te_tre, _, _, n_ps_0, n_ps_0_notrim = (
            self._check_zero_pscores(trimmed, p1te_tre, p2te_tre))
        # Compute treatment scores according to dml formula
        tscores1 = (d1tre*d2tre*(y2 - y2d1d2te_tre)/(p1te_tre*p2te_tre))
        tscores2 = (d1tre*(y2d1d2te_tre-y1d1te_tre)/(p1te_tre))
        tscores = tscores1 + tscores2 + y1d1te_tre
        # Create empty results dict
        gate = {}
        # Loop over unique categories
        for cat in z0a_unique:
            gate[cat] = {}
            scores = tscores[(z0a == cat) & (trimmed == 0)]
            # Compute means for trimmed observations
            meantreat = np.mean(scores)
            # Compute standard errors and p-value
            se = np.sqrt(np.mean(np.power((scores - meantreat), 2))/len(
                scores))
            pval = 2 * (norm.cdf(-np.abs(meantreat / se)))
            # Compute 95% confidence interval
            level = 0.95
            a = (1 - level)
            ab = np.array([a / 2, 1. - a / 2])
            fac = norm.ppf(ab)
            ci = np.vstack((meantreat + se * fac[0], meantreat + se * fac[
                1])).T
            gate[cat]['meantreat'] = meantreat
            gate[cat]['se'] = se
            gate[cat]['pval'] = pval
            gate[cat]['ci'] = ci
            gate[cat]['N'] = np.sum((z0a == cat))
            gate[cat]['Ntrim'] = np.sum((z0a == cat) & (trimmed != 0))
        # Get indicator which is 1 if individual treated in periods 1 and
        # 2 and not trimmed
        if self.dynamic_confounding:
            idx_treat_without_trimmed = d1tre & d2tre & (trimmed == 0)
        else:
            idx_treat_without_trimmed = d1tre & (trimmed == 0)
        # Set NAN propensity scores back to 0
        if n_ps_0 > 0:
            p1te_tre[np.isnan(p1te_tre)] = 0
            p2te_tre[np.isnan(p2te_tre)] = 0
        # create dictionary of results
        self.APO[seq_name_group] = {
            'd1treat': d1treat,
            'd2treat': d2treat,
            'trim': trim,
            'trim_str': trim_str,
            'p1te_tre': p1te_tre,
            'p2te_tre': p2te_tre,
            'd1tre': d1tre,
            'd2tre': d2tre,
            'y2d1d2te_tre': y2d1d2te_tre,
            'y1d1te_tre': y1d1te_tre,
            'y2': y2,
            'n_ps_0': n_ps_0,
            'n_ps_0_notrim': n_ps_0_notrim,
            'tscores': tscores,
            'idx_treat_without_trimmed': idx_treat_without_trimmed,
            'groupvar': z0a,
            'name_groupvar': name_groupvar,
            'results': gate
            }
        self._summary_APO(self.APO[seq_name_group])
        return self

    def compute_GAPOmAPO(
            self, d1treat, d2treat, groupvar, name_groupvar, trim=0):
        """
        Compute difference between group average potential outcome and
        average potential outcome (GAPOmAPO) from fitted scores of a
        particular sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        groupvar : array-like of shape (n_samples,) or None
            Discrete group variable. If provided, APO is computed within
            each group. If `None`, the APO is computed for the whole sample.
            The default is `None`.
        name_groupvar : str or None
            The name of the group variable. Needs to be provided only if
            ``groupvar`` is provided. The default is `None`.
        trim : int, float or array-like of shape (n_samples,)
            The trimming variable can be provided in these ways:

            * ``int`` or ``float``: A fixed trimming threshold, e.g. trim
              all propensity scores below 0.01.
            * ``array-like``: A bool array indicating which observations to
              trim (observations with ``True`` are trimmed)

            The default is 0 (no trimming).
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        seq_name_treat = str(d1treat) + '_' + str(d2treat)
        seq_name_group = seq_name_treat
        self._check_vartype(name_groupvar, str, 'name_groupvar')
        seq_name_group = seq_name_treat + '_' + name_groupvar
        # Check if sequence exists already
        if not (seq_name_treat in self.sequences):
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} does not exist.')
        # Check if fitted
        if not self.sequences[seq_name_treat].is_fitted:
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} is not fitted.')
        # Check groups
        z0a = check_array(groupvar, ensure_2d=False, dtype=str)
        # Check if same length as dml_score
        if len(z0a) != len(self.sequences[seq_name_treat].tscores):
            raise ValueError(
                f'Group indicator must have length '
                f'{len(self.sequences[seq_name_treat].tscores)}, '
                f'got {len(z0a)}.')
        # Get unique values in z0
        z0a_unique = np.unique(z0a)
        # Check trim argument
        trim_str = self._check_trim(trim, n=len(
            self.sequences[seq_name_treat].y2))
        # Get inputs
        p1te_tre = self.sequences[seq_name_treat].psd1treat
        p2te_tre = self.sequences[seq_name_treat].psd2treat
        d1tre = self.sequences[seq_name_treat].d1tre
        d2tre = self.sequences[seq_name_treat].d2tre
        y2d1d2te_tre = self.sequences[seq_name_treat].y2d1d2treat
        y1d1te_tre = self.sequences[seq_name_treat].y1d1treat
        y2 = self.sequences[seq_name_treat].y2
        # Get indicator for trimmed observations
        if trim_str == 'array':
            _, trimmed = check_X_y(trim, trim, ensure_2d=False)
            trimmed = 1*trimmed
        else:
            trimmed = 1*((p1te_tre*p2te_tre) < trim)
        # Check for zero propensity scores
        trimmed, p1te_tre, p2te_tre, _, _, n_ps_0, n_ps_0_notrim = (
            self._check_zero_pscores(trimmed, p1te_tre, p2te_tre))
        # Compute treatment scores according to dml formula
        tscores1 = (d1tre*d2tre*(y2 - y2d1d2te_tre)/(p1te_tre*p2te_tre))
        tscores2 = (d1tre*(y2d1d2te_tre-y1d1te_tre)/(p1te_tre))
        tscores = tscores1 + tscores2 + y1d1te_tre
        # Compute APO
        scores_apo = tscores[(trimmed == 0)]
        apo_n = len(scores_apo)
        apo_effect = np.mean(scores_apo)
        apo_se = np.sqrt(np.mean(np.power(scores_apo-apo_effect, 2))/apo_n)
        # Create empty results dict
        gapo = {}
        # Loop over unique categories
        for cat in z0a_unique:
            gapo[cat] = {}
            scores_gapo = tscores[(z0a == cat) & (trimmed == 0)]
            gapo_n = len(scores_gapo)
            gapo_effect = np.mean(scores_gapo)
            gapo_se = np.sqrt(np.mean(np.power(
                scores_gapo-gapo_effect, 2))/gapo_n)
            # Compute Gapo-apo
            gapo_m_apo_effect = gapo_effect - apo_effect
            gapo_m_apo_var = np.power(apo_se, 2) + np.power(gapo_se, 2) - (
                2*(gapo_n/apo_n)*np.power(gapo_se, 2))
            gapo_m_apo_se = np.sqrt(gapo_m_apo_var)
            pval = 2 * (norm.cdf(-np.abs(gapo_m_apo_effect / gapo_m_apo_se)))
            # Compute 95% confidence interval
            level = 0.95
            a = (1 - level)
            ab = np.array([a / 2, 1. - a / 2])
            fac = norm.ppf(ab)
            ci = np.vstack((gapo_m_apo_effect + gapo_m_apo_se * fac[
                0], gapo_m_apo_effect + gapo_m_apo_se * fac[1])).T
            gapo[cat]['meantreat'] = gapo_m_apo_effect
            gapo[cat]['se'] = gapo_m_apo_se
            gapo[cat]['pval'] = pval
            gapo[cat]['ci'] = ci
            gapo[cat]['N'] = np.sum((z0a == cat))
            gapo[cat]['Ntrim'] = np.sum((z0a == cat) & (trimmed != 0))
        # Get indicator which is 1 if individual treated in periods 1 and
        # 2 and not trimmed
        if self.dynamic_confounding:
            idx_treat_without_trimmed = d1tre & d2tre & (trimmed == 0)
        else:
            idx_treat_without_trimmed = d1tre & (trimmed == 0)
        # Set NAN propensity scores back to 0
        if n_ps_0 > 0:
            p1te_tre[np.isnan(p1te_tre)] = 0
            p2te_tre[np.isnan(p2te_tre)] = 0
        # create dictionary of results
        self.GAPOmAPO[seq_name_group] = {
            'd1treat': d1treat,
            'd2treat': d2treat,
            'trim': trim,
            'trim_str': trim_str,
            'p1te_tre': p1te_tre,
            'p2te_tre': p2te_tre,
            'd1tre': d1tre,
            'd2tre': d2tre,
            'y2d1d2te_tre': y2d1d2te_tre,
            'y1d1te_tre': y1d1te_tre,
            'y2': y2,
            'n_ps_0': n_ps_0,
            'n_ps_0_notrim': n_ps_0_notrim,
            'tscores': tscores,
            'idx_treat_without_trimmed': idx_treat_without_trimmed,
            'groupvar': z0a,
            'name_groupvar': name_groupvar,
            'results': gapo
            }
        self._summary_APO(self.GAPOmAPO[seq_name_group])
        return self

    def _summary_ATE(self, result):
        self._check_vartype(result, dict, 'result')
        groupinfo = ""
        if result['name_groupvar'] is not None:
            groupinfo = ', by ' + result['name_groupvar']
        print("="*80)
        print(f"Estimation results of sequence {result['d1treat']}-"
              f"{result['d2treat']} vs. "
              f"{result['d1control']}-{result['d2control']}{groupinfo}:")
        print("-"*80)
        summary = pd.DataFrame(index=[
                'coef', 'std err', 't', 'P>|t|', '2.5%', '97.5%', 'N',
                'Ntrim'], columns=result['results'].keys()).T
        for cat in result['results']:
            summary.loc[cat, 'coef'] = (
                f"{result['results'][cat]['effect']:.3f}")
            summary.loc[cat, 'std err'] = (
                f"{result['results'][cat]['se']:.3f}")
            tval = result['results'][cat]['effect']/result[
                'results'][cat]['se']
            summary.loc[cat, 't'] = (
                f"{tval:.3f}")
            summary.loc[cat, 'P>|t|'] = (
                f"{result['results'][cat]['pval']:.3f}")
            summary.loc[cat, '2.5%'] = (
                f"{result['results'][cat]['ci'][0][0]:.3f}")
            summary.loc[cat, '97.5%'] = (
                f"{result['results'][cat]['ci'][0][1]:.3f}")
            summary.loc[cat, 'N'] = (
                f"{result['results'][cat]['N']}")
            summary.loc[cat, 'Ntrim'] = (
                f"{result['results'][cat]['Ntrim']:.0f}")
        print(summary)
        print("="*80)

    def compute_ATE(
            self, d1treat, d2treat, d1control, d2control, groupvar=None,
            name_groupvar=None, trim=0):
        """
        Compute (group) average treatment effect (ATE) from fitted scores of a
        particular sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual first period treatment of treatment group.
        d2treat : int or str
            Name of counterfactual second period treatment of treatment group.
        d1control : int or str
            Name of counterfactual first period treatment of control group.
        d2control : int or str
            Name of counterfactual second period treatment of control group.
        groupvar : array-like of shape (n_samples,) or None
            Discrete group variable. If provided, APO is computed within
            each group. If `None`, the APO is computed for the whole sample.
            The default is `None`.
        name_groupvar : str or None
            The name of the group variable. Needs to be provided only if
            ``groupvar`` is provided. The default is `None`.
        trim : int, float or array-like of shape (n_samples,)
            The trimming variable can be provided in these ways:

            * ``int`` or ``float``: A fixed trimming threshold, e.g. trim
              all propensity scores below 0.01.
            * ``array-like``: A bool array indicating which observations to
              trim (observations with ``True`` are trimmed)

            The default is 0 (no trimming).
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        self._check_vartype(d1control, (int, str), 'd1control')
        self._check_vartype(d2control, (int, str), 'd2control')
        # Check if both exist and fitted
        # Get sequence names
        seq_name_treat = str(d1treat) + '_' + str(d2treat)
        seq_name_control = str(d1control) + '_' + str(d2control)
        # Create effect name
        effect_name = seq_name_treat + "_vs_" + seq_name_control
        if (groupvar is not None) and (name_groupvar is None):
            raise ValueError(
                'If groupvar supplied, name_groupvar must be supplied as'
                ' well.')
        if (groupvar is None) and (name_groupvar is not None):
            name_groupvar = None
            print('Warning: name_groupvar set to None as no groupvar supplied')
        if name_groupvar is not None:
            self._check_vartype(name_groupvar, str, 'name_groupvar')
            effect_name = effect_name + '_' + name_groupvar
        # Check if sequence exists already
        if not (seq_name_treat in self.sequences):
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} does not exist.')
        if not (seq_name_control in self.sequences):
            raise ValueError(
                f'Control sequence {d1control} - {d2control} does not exist.')
        # Check if fitted
        if not self.sequences[seq_name_treat].is_fitted:
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} is not fitted.')
        if not self.sequences[seq_name_control].is_fitted:
            raise ValueError(
                f'Control sequence {d1control} - {d2control} is not fitted.')
        # Check if scores same length and same y
        if not (len(self.sequences[seq_name_treat].tscores) == len(
                self.sequences[seq_name_control].tscores)):
            raise ValueError(
                'Length of scores does not match. Use same observations to '
                'fit both sequences.')
        if not ((self.sequences[seq_name_treat].y2 == self.sequences[
                seq_name_control].y2).min()):
            raise ValueError('Sequences not fitted using same outcome.')
        # Check groups
        if groupvar is not None:
            z0a = check_array(groupvar, ensure_2d=False, dtype=str)
            # Check if same length as dml_score
            if len(z0a) != len(self.sequences[seq_name_treat].tscores):
                raise ValueError(
                    f'Group indicator must have length '
                    f'{len(self.sequences[seq_name_treat].tscores)}, '
                    f'got {len(z0a)}.')
        else:
            name = "ATE"
            z0a = np.array([name]*len(self.sequences[
                seq_name_treat].tscores))
        # Get unique values in z0
        z0a_unique = np.unique(z0a)
        # Check trim argument
        trim_str = self._check_trim(trim, n=len(
            self.sequences[seq_name_treat].y2))
        # Get inputs
        p1te_tre = self.sequences[seq_name_treat].psd1treat
        p2te_tre = self.sequences[seq_name_treat].psd2treat
        p1te_con = self.sequences[seq_name_control].psd1treat
        p2te_con = self.sequences[seq_name_control].psd2treat
        d1tre = self.sequences[seq_name_treat].d1tre
        d2tre = self.sequences[seq_name_treat].d2tre
        d1con = self.sequences[seq_name_control].d1tre
        d2con = self.sequences[seq_name_control].d2tre
        y2d1d2te_tre = self.sequences[seq_name_treat].y2d1d2treat
        y1d1te_tre = self.sequences[seq_name_treat].y1d1treat
        y2d1d2te_con = self.sequences[seq_name_control].y2d1d2treat
        y1d1te_con = self.sequences[seq_name_control].y1d1treat
        y2 = self.sequences[seq_name_treat].y2
        # Get indicator for trimmed observations
        if trim_str == 'array':
            _, trimmed = check_X_y(trim, trim, ensure_2d=False)
            trimmed = 1*trimmed
        else:
            trimmed_tre = 1*((p1te_tre*p2te_tre) < trim)
            trimmed_con = 1*((p1te_con*p2te_con) < trim)
            trimmed = 1*((trimmed_tre + trimmed_con) > 0)
        # Check for zero propensity scores
        trimmed, p1te_tre, p2te_tre, p1te_con, p2te_con, n_ps_0, \
            n_ps_0_notrim = self._check_zero_pscores(
                trimmed, p1te_tre, p2te_tre, p1te_con, p2te_con)
        # Compute treatment scores according to dml formula
        tscores1 = (d1tre*d2tre*(y2 - y2d1d2te_tre)/(p1te_tre*p2te_tre))
        tscores2 = (d1tre*(y2d1d2te_tre-y1d1te_tre)/(p1te_tre))
        tscores = tscores1 + tscores2 + y1d1te_tre
        cscores1 = (d1con*d2con*(y2 - y2d1d2te_con)/(p1te_con*p2te_con))
        cscores2 = (d1con*(y2d1d2te_con-y1d1te_con)/(p1te_con))
        cscores = cscores1 + cscores2 + y1d1te_con
        # Compute difference in scores per individual
        dml_scores = tscores - cscores
        # Create empty results dict
        gate = {}
        # Loop over unique categories
        for cat in z0a_unique:
            gate[cat] = {}
            scores = dml_scores[(z0a == cat) & (trimmed == 0)]
            # Compute means for trimmed observations
            effect = np.mean(scores)
            # Compute standard errors and p-value
            se = np.sqrt(np.mean(np.power((scores - effect), 2))/len(
                scores))
            pval = 2 * (norm.cdf(-np.abs(effect / se)))
            # Compute 95% confidence interval
            level = 0.95
            a = (1 - level)
            ab = np.array([a / 2, 1. - a / 2])
            fac = norm.ppf(ab)
            ci = np.vstack((effect + se * fac[0], effect + se * fac[
                1])).T
            # Alternative effects:
            # ICE: Iterated Conditional Expectations
            effect_ice = np.mean(y1d1te_tre[(z0a == cat) & (
                trimmed == 0)]) - np.mean(y1d1te_con[(z0a == cat) & (
                    trimmed == 0)])
            # IPW: Inverse Probability Weighting
            if self.dynamic_confounding:
                ipw_weight_treat = (d1tre*d2tre)/(p1te_tre*p2te_tre)
                ipw_weight_contr = (d1con*d2con)/(p1te_con*p2te_con)
            else:
                ipw_weight_treat = (d1tre)/(p1te_tre)
                ipw_weight_contr = (d1con)/(p1te_con)
            ipw_treat = np.sum(y2[(z0a == cat) & (
                trimmed == 0)]*ipw_weight_treat[(z0a == cat) & (
                    trimmed == 0)])/np.sum(ipw_weight_treat[(z0a == cat) & (
                        trimmed == 0)])
            ipw_contr = np.sum(y2[(z0a == cat) & (
                trimmed == 0)]*ipw_weight_contr[(z0a == cat) & (
                    trimmed == 0)])/np.sum(ipw_weight_contr[(z0a == cat) & (
                        trimmed == 0)])
            effect_ipw = ipw_treat - ipw_contr
            gate[cat]['effect'] = effect
            gate[cat]['effect_ice'] = effect_ice
            gate[cat]['effect_ipw'] = effect_ipw
            gate[cat]['se'] = se
            gate[cat]['pval'] = pval
            gate[cat]['ci'] = ci
            gate[cat]['N'] = np.sum((z0a == cat))
            gate[cat]['Ntrim'] = np.sum((z0a == cat) & (trimmed != 0))
        # Get indicator which is 1 if individual treated in periods 1 and
        # 2 and not trimmed
        if self.dynamic_confounding:
            idx_treat_without_trimmed = d1tre & d2tre & (trimmed == 0)
            # Same for controls
            idx_control_without_trimmed = d1con & d2con & (trimmed == 0)
        else:
            idx_treat_without_trimmed = d1tre & (trimmed == 0)
            idx_control_without_trimmed = d1con & (trimmed == 0)
        # Set NAN propensity scores back to 0
        if n_ps_0 > 0:
            p1te_tre[np.isnan(p1te_tre)] = 0
            p2te_tre[np.isnan(p2te_tre)] = 0
            p1te_con[np.isnan(p1te_con)] = 0
            p2te_con[np.isnan(p2te_con)] = 0
        # create dictionary of results
        self.ATE[effect_name] = {
            'd1treat': d1treat,
            'd1control': d1control,
            'd2treat': d2treat,
            'd2control': d2control,
            'trim': trim,
            'trim_str': trim_str,
            'p1te_tre': p1te_tre,
            'p2te_tre': p2te_tre,
            'p1te_con': p1te_con,
            'p2te_con': p2te_con,
            'd1tre': d1tre,
            'd2tre': d2tre,
            'd1con': d1con,
            'd2con': d2con,
            'y2d1d2te_tre': y2d1d2te_tre,
            'y1d1te_tre': y1d1te_tre,
            'y2d1d2te_con': y2d1d2te_con,
            'y1d1te_con': y1d1te_con,
            'y2': y2,
            'n_ps_0': n_ps_0,
            'n_ps_0_notrim': n_ps_0_notrim,
            'tscores': tscores,
            'cscores': tscores,
            'dml_scores': dml_scores,
            'idx_treat_without_trimmed': idx_treat_without_trimmed,
            'idx_control_without_trimmed': idx_control_without_trimmed,
            'groupvar': z0a,
            'name_groupvar': name_groupvar,
            'results': gate
            }
        self._summary_ATE(self.ATE[effect_name])
        return self

    def compute_GATEmATE(
            self, d1treat, d2treat, d1control, d2control, groupvar,
            name_groupvar, trim=0):
        """
        Compute difference between group average treatment effect and
        average treatment effect (GATEmATE) from fitted scores of a
        particular sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual first period treatment of treatment group.
        d2treat : int or str
            Name of counterfactual second period treatment of treatment group.
        d1control : int or str
            Name of counterfactual first period treatment of control group.
        d2control : int or str
            Name of counterfactual second period treatment of control group.
        groupvar : array-like of shape (n_samples,) or None
            Discrete group variable. If provided, APO is computed within
            each group. If `None`, the APO is computed for the whole sample.
            The default is `None`.
        name_groupvar : str or None
            The name of the group variable. Needs to be provided only if
            ``groupvar`` is provided. The default is `None`.
        trim : int, float or array-like of shape (n_samples,)
            The trimming variable can be provided in these ways:

            * ``int`` or ``float``: A fixed trimming threshold, e.g. trim
              all propensity scores below 0.01.
            * ``array-like``: A bool array indicating which observations to
              trim (observations with ``True`` are trimmed)

            The default is 0 (no trimming).
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        self._check_vartype(d1control, (int, str), 'd1control')
        self._check_vartype(d2control, (int, str), 'd2control')
        self._check_vartype(name_groupvar, str, 'name_groupvar')
        # Check if both exist and fitted
        # Get sequence names
        seq_name_treat = str(d1treat) + '_' + str(d2treat)
        seq_name_control = str(d1control) + '_' + str(d2control)
        # Create effect name
        effect_name = (
            seq_name_treat + "_vs_" + seq_name_control + '_' + name_groupvar)
        # Check if sequence exists already
        if not (seq_name_treat in self.sequences):
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} does not exist.')
        if not (seq_name_control in self.sequences):
            raise ValueError(
                f'Control sequence {d1control} - {d2control} does not exist.')
        # Check if fitted
        if not self.sequences[seq_name_treat].is_fitted:
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} is not fitted.')
        if not self.sequences[seq_name_control].is_fitted:
            raise ValueError(
                f'Control sequence {d1control} - {d2control} is not fitted.')
        # Check if scores same length and same y
        if not (len(self.sequences[seq_name_treat].tscores) == len(
                self.sequences[seq_name_control].tscores)):
            raise ValueError(
                'Length of scores does not match. Use same observations to '
                'fit both sequences.')
        if not ((self.sequences[seq_name_treat].y2 == self.sequences[
                seq_name_control].y2).min()):
            raise ValueError('Sequences not fitted using same outcome.')
        # Check groups
        z0a = check_array(groupvar, ensure_2d=False, dtype=str)
        # Check if same length as dml_score
        if len(z0a) != len(self.sequences[seq_name_treat].tscores):
            raise ValueError(
                f'Group indicator must have length '
                f'{len(self.sequences[seq_name_treat].tscores)}, '
                f'got {len(z0a)}.')
        # Get unique values in z0
        z0a_unique = np.unique(z0a)
        # Check trim argument
        trim_str = self._check_trim(trim, n=len(
            self.sequences[seq_name_treat].y2))
        # Get inputs
        p1te_tre = self.sequences[seq_name_treat].psd1treat
        p2te_tre = self.sequences[seq_name_treat].psd2treat
        p1te_con = self.sequences[seq_name_control].psd1treat
        p2te_con = self.sequences[seq_name_control].psd2treat
        d1tre = self.sequences[seq_name_treat].d1tre
        d2tre = self.sequences[seq_name_treat].d2tre
        d1con = self.sequences[seq_name_control].d1tre
        d2con = self.sequences[seq_name_control].d2tre
        y2d1d2te_tre = self.sequences[seq_name_treat].y2d1d2treat
        y1d1te_tre = self.sequences[seq_name_treat].y1d1treat
        y2d1d2te_con = self.sequences[seq_name_control].y2d1d2treat
        y1d1te_con = self.sequences[seq_name_control].y1d1treat
        y2 = self.sequences[seq_name_treat].y2
        # Get indicator for trimmed observations
        if trim_str == 'array':
            _, trimmed = check_X_y(trim, trim, ensure_2d=False)
            trimmed = 1*trimmed
        else:
            trimmed_tre = 1*((p1te_tre*p2te_tre) < trim)
            trimmed_con = 1*((p1te_con*p2te_con) < trim)
            trimmed = 1*((trimmed_tre + trimmed_con) > 0)
        # Check for zero propensity scores
        trimmed, p1te_tre, p2te_tre, p1te_con, p2te_con, n_ps_0, \
            n_ps_0_notrim = (self._check_zero_pscores(
                trimmed, p1te_tre, p2te_tre, p1te_con, p2te_con))
        # Compute treatment scores according to dml formula
        tscores1 = (d1tre*d2tre*(y2 - y2d1d2te_tre)/(p1te_tre*p2te_tre))
        tscores2 = (d1tre*(y2d1d2te_tre-y1d1te_tre)/(p1te_tre))
        tscores = tscores1 + tscores2 + y1d1te_tre
        cscores1 = (d1con*d2con*(y2 - y2d1d2te_con)/(p1te_con*p2te_con))
        cscores2 = (d1con*(y2d1d2te_con-y1d1te_con)/(p1te_con))
        cscores = cscores1 + cscores2 + y1d1te_con
        # Compute difference in scores per individual
        dml_scores = tscores - cscores
        scores_ate = dml_scores[(trimmed == 0)]
        ate_n = len(scores_ate)
        ate_effect = np.mean(scores_ate)
        ate_se = np.sqrt(np.mean(np.power(scores_ate-ate_effect, 2))/ate_n)
        # Create empty results dict
        gate = {}
        # Loop over unique categories
        for cat in z0a_unique:
            gate[cat] = {}
            scores_gate = dml_scores[(z0a == cat) & (trimmed == 0)]
            gate_n = len(scores_gate)
            gate_effect = np.mean(scores_gate)
            gate_se = np.sqrt(np.mean(np.power(
                scores_gate-gate_effect, 2))/gate_n)
            # Compute GATE-ATE
            gate_m_ate_effect = gate_effect - ate_effect
            gate_m_ate_var = np.power(ate_se, 2) + np.power(gate_se, 2) - (
                2*(gate_n/ate_n)*np.power(gate_se, 2))
            gate_m_ate_se = np.sqrt(gate_m_ate_var)
            pval = 2 * (norm.cdf(-np.abs(gate_m_ate_effect / gate_m_ate_se)))
            # Compute 95% confidence interval
            level = 0.95
            a = (1 - level)
            ab = np.array([a / 2, 1. - a / 2])
            fac = norm.ppf(ab)
            ci = np.vstack((gate_m_ate_effect + gate_m_ate_se * fac[
                0], gate_m_ate_effect + gate_m_ate_se * fac[1])).T
            gate[cat]['effect'] = gate_m_ate_effect
            gate[cat]['se'] = gate_m_ate_se
            gate[cat]['pval'] = pval
            gate[cat]['ci'] = ci
            gate[cat]['N'] = np.sum((z0a == cat))
            gate[cat]['Ntrim'] = np.sum((z0a == cat) & (trimmed != 0))
        # Get indicator which is 1 if individual treated in periods 1 and
        # 2 and not trimmed
        if self.dynamic_confounding:
            idx_treat_without_trimmed = d1tre & d2tre & (trimmed == 0)
            # Same for controls
            idx_control_without_trimmed = d1con & d2con & (trimmed == 0)
        else:
            idx_treat_without_trimmed = d1tre & (trimmed == 0)
            idx_control_without_trimmed = d1con & (trimmed == 0)
        # Set NAN propensity scores back to 0
        if n_ps_0 > 0:
            p1te_tre[np.isnan(p1te_tre)] = 0
            p2te_tre[np.isnan(p2te_tre)] = 0
            p1te_con[np.isnan(p1te_con)] = 0
            p2te_con[np.isnan(p2te_con)] = 0
        # create dictionary of results
        self.GATEmATE[effect_name] = {
            'd1treat': d1treat,
            'd1control': d1control,
            'd2treat': d2treat,
            'd2control': d2control,
            'trim': trim,
            'trim_str': trim_str,
            'p1te_tre': p1te_tre,
            'p2te_tre': p2te_tre,
            'p1te_con': p1te_con,
            'p2te_con': p2te_con,
            'd1tre': d1tre,
            'd2tre': d2tre,
            'd1con': d1con,
            'd2con': d2con,
            'y2d1d2te_tre': y2d1d2te_tre,
            'y1d1te_tre': y1d1te_tre,
            'y2d1d2te_con': y2d1d2te_con,
            'y1d1te_con': y1d1te_con,
            'y2': y2,
            'n_ps_0': n_ps_0,
            'n_ps_0_notrim': n_ps_0_notrim,
            'tscores': tscores,
            'cscores': tscores,
            'dml_scores': dml_scores,
            'idx_treat_without_trimmed': idx_treat_without_trimmed,
            'idx_control_without_trimmed': idx_control_without_trimmed,
            'groupvar': z0a,
            'name_groupvar': name_groupvar,
            'results': gate
            }
        self._summary_ATE(self.GATEmATE[effect_name])
        return self

    def summary(
            self, d1treat, d2treat, d1control=None, d2control=None,
            name_groupvar=None):
        """
        Print summary of a particular (group) average potential outcome or
        treatment effect (APO/GAPO/ATE/GATE) without recomputing it.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual first period treatment of treatment group.
        d2treat : int or str
            Name of counterfactual second period treatment of treatment group.
        d1control : int or str or None
            Name of counterfactual first period treatment of control group.
            `None` for (group) average potential outcome.
            The default is `None`.
        d2control : int or str or None
            Name of counterfactual second period treatment of control group.
            `None` for (group) average potential outcome.
            The default is `None`.
        name_groupvar : str or None
            The name of the group variable. Needs to be provided only if
            GAPO or GATE should be printed. The default is `None`.
        """
        self._check_vartype(d1treat, (int, str), 'd1treat')
        self._check_vartype(d2treat, (int, str), 'd2treat')
        if d1control is not None:
            self._check_vartype(d1control, (int, str), 'd1control')
        if d2control is not None:
            self._check_vartype(d2control, (int, str), 'd2control')
        if name_groupvar is not None:
            self._check_vartype(name_groupvar, str, 'name_groupvar')
        seq_name_treat = str(d1treat) + '_' + str(d2treat)
        groupinfo = ""
        if (d1control is not None) and (d2control is not None):
            seq_name_control = str(d1control) + '_' + str(d2control)
            effect_name = seq_name_treat + "_vs_" + seq_name_control
            if name_groupvar is not None:
                effect_name = effect_name + "_" + name_groupvar
                groupinfo = f" by {name_groupvar}"
            if not (effect_name in self.ATE):
                raise ValueError(
                    f"Effect {d1treat}-{d2treat} vs. {d1control}-{d2control}"
                    f"{groupinfo} does not exist. Run compute_ATE.")
            self._summary_ATE(self.ATE[effect_name])
        elif (d1control is None) and (d2control is None):
            if name_groupvar is not None:
                seq_name_treat = seq_name_treat + "_" + name_groupvar
                groupinfo = f" by {name_groupvar}"
            if not (seq_name_treat in self.APO):
                raise ValueError(
                    f"APO {d1treat}-{d2treat}{groupinfo} does not exist."
                    f"Run compute_APO.")
            self._summary_APO(self.APO[seq_name_treat])
        else:
            raise ValueError(
                "Provide d1control and d2control to get summary of ATE or "
                "set them both to `None` to get summary of APO.")

    def plot_pscores(
            self,
            d1treat,
            d2treat,
            trim=0,
            title="Propensity Score Density Plots",
            align_xlim=False,
            stat='density',
            common_bins=False,
            prog_dict=None):
        """
        Plot propensity score densities of a particular sequence.

        Parameters
        ----------
        d1treat : int or str
            Name of counterfactual treatment in first period.
        d2treat : int or str
            Name of counterfactual treatment in second period.
        trim : int, float or array-like of shape (n_samples,)
            The trimming variable can be provided in these ways:

            * ``int`` or ``float``: A fixed trimming threshold, e.g. trim
              all propensity scores below 0.01.
            * ``array-like``: A bool array indicating which observations to
              trim (observations with ``True`` are trimmed)

            The default is 0 (no trimming).
        title : str
            Title of the plot. The default is
            `"Propensity Score Density Plots"`.
        align_xlim : bool
            Whether to force the x-axis to range from 0 to 1. The default is
            `False`.
        stat : str
            The aggregate statistic to compute in each bin. One of ``count``,
            ``frequency``, ``probability``, ``percent`` or ``density``. See
            [`seaborn.histplot`](https://seaborn.pydata.org/generated/seaborn.histplot.html){:target="_blank"}.
            for details. The default is ``density``.
        common_bins : bool
            If `True`, use the same bins for both histograms. The default is
            `False`.
        prog_dict : dict or None
            Dictionary mapping program identifiers to names. The default is
            `None`.
        """
        # Check if both exist and fitted
        # Get sequence names
        seq_name_treat = str(d1treat) + '_' + str(d2treat)
        # Check if sequence exists already
        if not (seq_name_treat in self.sequences):
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} does not exist.')
        # Check if fitted
        if not self.sequences[seq_name_treat].is_fitted:
            raise ValueError(
                f'Treat sequence {d1treat} - {d2treat} is not fitted.')
        # Check trim argument
        trim_str = self._check_trim(trim, n=len(
            self.sequences[seq_name_treat].y2))
        # Get inputs
        psd1treat = self.sequences[seq_name_treat].psd1treat
        psd2treat = self.sequences[seq_name_treat].psd2treat
        d1tre = self.sequences[seq_name_treat].d1tre
        d2tre = self.sequences[seq_name_treat].d2tre
        # Get unique programs in second period
        g2t = self.sequences[seq_name_treat].g2t
        g2t_unique = np.unique(g2t)
        g2t_nunique = len(g2t_unique)
        # Check prog_dict argument
        if prog_dict is not None:
            self._check_vartype(prog_dict, (dict), 'prog_dict')
            check_isin = np.isin(g2t_unique, np.array(list(prog_dict.keys())))
            if not check_isin.all():
                raise ValueError(
                    f'Values {g2t_unique[~check_isin]} not present in keys of '
                    f'prog_dict.')
        else:
            prog_dict = dict(zip(g2t_unique, g2t_unique))
        # Get indicator for trimmed observations
        if trim_str == 'array':
            _, trimmed = check_X_y(trim, trim, ensure_2d=False)
            trimmed = 1*trimmed
        else:
            trimmed = 1*((psd1treat*psd2treat) < trim) if trim > 0 else (
                np.zeros_like(psd1treat))
        # Prepare data and indices
        ps = [psd1treat] + [psd2treat]*g2t_nunique
        idx_t = ([(d1tre == 1) & (trimmed == 0)] + [(((d1tre == 1) & (
            d2tre == 1)) & (trimmed == 0) & (
                g2t == xx)) for xx in g2t_unique])*2
        idx_c = ([~(d1tre == 1) & (trimmed == 0)] + [((~(d1tre == 1) | ~(
            d2tre == 1)) & (trimmed == 0) & (
                g2t == xx)) for xx in g2t_unique])*2
        # Titles for the plots
        if self.dynamic_confounding:
            titles = ([f"Pr($D_1$={d1treat}|$X_0$)"] + [
                f"Pr($D_2$={d2treat}|$X_0$, $X_1$, $D_1$={d1treat})"] *
                g2t_nunique)
            legend_t = ([f"$D_1=${d1treat}"] + [
                f"$D_1$={d1treat} and $D_2=${d2treat}"]*g2t_nunique)*2
            legend_c = ([f"$D_1 \\neq${d1treat}"] + [
                f"$D_1 \\neq${d1treat} or $D_2 \\neq${d2treat}"]*g2t_nunique)*2
            if g2t_nunique > 1:
                add1 = ([""] + ["("]*g2t_nunique)*2
                add2 = ([""] + [
                    f") and {d2treat}={prog_dict[xx]}" for xx in g2t_unique])*2
                legend_t = [x + y + z for x, y, z in zip(add1, legend_t, add2)]
                legend_c = [x + y + z for x, y, z in zip(add1, legend_c, add2)]
        else:
            titles = [f"Pr($D_1=${d1treat}, $D_2=${d2treat}|$X_0$)", "", ""]
            legend_t = [f"$D_1=${d1treat} and $D_2=${d2treat}", "", ""]*2
            legend_c = [f"$D_1$≠{d1treat} or $D_2\\neq${d2treat}", "", ""]*2
        # Set xlim
        if align_xlim is True:
            set_xlim = (-0.01, 1.01)
        else:
            set_xlim = None
        nobs = d1tre.shape[0]
        n_plots = (1 + g2t_nunique) if self.dynamic_confounding else 1
        plt.figure(figsize=(n_plots*4, 4))
        plt.suptitle(title, fontsize=18, y=0.98)
        for k in range(n_plots):
            plt.subplot(1, n_plots, k+1)
            cat = pd.Series([np.nan]*nobs, dtype='object')
            cat[idx_t[k] == 1] = legend_t[k]
            cat[idx_c[k] == 1] = legend_c[k]
            sns.histplot(
                x=ps[k], hue=cat.values, kde=True, element='step',
                stat=stat, common_norm=False, common_bins=common_bins,
                palette={
                    legend_t[k]: sns.color_palette()[0],
                    legend_c[k]: sns.color_palette()[1]}).set(
                        title=titles[k], xlim=set_xlim,
                        xlabel="Propensity score")
            plt.tight_layout()
        plt.show()

    def joint_trimming(
            self, seq_list, trim, trim_product=False):
        """
        Apply joint trimming across several sequences

        Parameters
        ----------
        seq_list : list of str
            The list of sequences to be considered in trimming. The elements
            of the list should be strings of the type ``'d1treat_d2treat'``.
        trim : tuple of integers of length 2
            A tuple inidating the quantiles used for joint trimming. For
            example, ``(0, 1)`` would implement min-max trimming, i.e. all
            observations with propensity score smaller than the largest
            minimum propensity score are deleted (and vice versa).
        trim_product : bool
            Whether to only trim with respect to the first-period and
            second-period propensity scores individually or also consider the
            product of the propensity scores across periods. The default is
            `False`.
        """
        self._check_vartype(seq_list, list, 'seq_list')
        seq_ready = []
        seq_notfit = []
        seq_notexist = []
        for seq in seq_list:
            # Check if sequence exists already
            if not (seq in self.sequences):
                seq_notexist.append(seq)
            # Check if fitted
            elif not self.sequences[seq].is_fitted:
                seq_notfit.append(seq)
            else:
                seq_ready.append(seq)
        if len(seq_notexist) > 0:
            raise ValueError(
                f'Cannot trim as the following sequences do not exist:'
                f' {seq_notexist}')
        if len(seq_notfit) > 0:
            raise ValueError(
                f'Cannot trim as the following sequences are not fitted:'
                f' {seq_notfit}')
        # Get number of propensity scores
        n_scores = len(self.sequences[seq_ready[0]].psd1treat)
        # Check trim argument
        self._check_vartype(trim, tuple, 'trim')
        _ = self._check_trim(trim, n=len(
            self.sequences[seq].y2))
        # Get the quantile
        fmin = self._quantile_fixed(trim[0])
        fmax = self._quantile_fixed(trim[1])
        # Loop over sequences
        # Implementation: treatment group vs. not the treatment group
        trimmed = np.zeros((n_scores,))
        for seq in seq_ready:
            # Get propensity scores
            p1te_tre = self.sequences[seq].psd1treat
            d1tre = self.sequences[seq].d1tre
            p2te_tre = self.sequences[seq].psd2treat
            d2tre = self.sequences[seq].d2tre
            g2t = self.sequences[seq].g2t
            # Procedure:
            # 1. Determine smallest p1te_tre for each treatment group
            # 2. Take the max of the two
            # 3. Drop all individuals with smaller (or equal) p1te_tre
            # Do the reverse for largest p1te_tre
            # I do <= in case the minimum is zero for both groups
            # trim p1 scores
            trimmed += 1*(p1te_tre <= np.max([
                fmin(p1te_tre[d1tre == 1]),
                fmin(p1te_tre[d1tre == 0])]))
            trimmed += 1*(p1te_tre >= np.min([
                fmax(p1te_tre[d1tre == 1]),
                fmax(p1te_tre[d1tre == 0])]))
            # trim p2 scores
            if self.dynamic_confounding:
                # Get unique g2t
                g2t_unique = np.unique(g2t)
                for dd in g2t_unique:
                    trimmed += 1*((p2te_tre) <= np.max([
                        fmin((p2te_tre)[((d1tre == 1) & (d2tre == 1)) & (
                            g2t == dd)]),
                        fmin((p2te_tre)[((d1tre != 1) | (d2tre != 1)) & (
                            g2t == dd)])
                        ]))*(g2t == dd)
                    trimmed += 1*((p2te_tre) >= np.min([
                        fmax((p2te_tre)[((d1tre == 1) & (d2tre == 1)) & (
                            g2t == dd)]),
                        fmax((p2te_tre)[((d1tre != 1) | (d2tre != 1)) & (
                            g2t == dd)])
                        ]))*(g2t == dd)
            if trim_product and self.dynamic_confounding:
                # Get unique g2t
                g2t_unique = np.unique(g2t)
                for dd in g2t_unique:
                    # trim p1*p2 scores
                    trimmed += 1*((p1te_tre*p2te_tre) <= np.max([
                        fmin((p1te_tre*p2te_tre)[((d1tre == 1) & (
                            d2tre == 1)) & (g2t == dd)]),
                        fmin((p1te_tre*p2te_tre)[((d1tre != 1) | (
                            d2tre != 1)) & (g2t == dd)])
                        ]))*(g2t == dd)
                    trimmed += 1*((p1te_tre*p2te_tre) >= np.min([
                        fmax((p1te_tre*p2te_tre)[((d1tre == 1) & (
                            d2tre == 1)) & (g2t == dd)]),
                        fmax((p1te_tre*p2te_tre)[((d1tre != 1) | (
                            d2tre != 1)) & (g2t == dd)])
                        ]))*(g2t == dd)
        trimmed = (trimmed > 0)
        return trimmed
