import numpy as np
from scipy.special import expit


def dyn_data_example(
        n=1000,
        random_state=None,
        r_0=3):
    """
    Generate data according to linear dgp from Bradic et al. (2024)[^Bradic].

    Parameters
    ----------
    n : int
        The number of observations to be generated. The defualt is 1000.
    random_state : int, RandomState instance or None
        Controls randomness of the samples generated. The default is `None`.
    r_0 : int
        The number of covariates per period. Only the first three covariates
        will have an influence on future treatment assignment, covariates
        and outcomes. The default is 3.

    Returns
    -------
    return_dict : dict
        Dictionary of generated variables:

        * ``X0``: pre-treatment covariates
        * ``p1``: first period propensity scores
        * ``D1``: first period treatment
        * ``X1_d10``: potential time-varying covariates if ``d1=0``
        * ``X1_d11``: potential time-varying covariates if ``d1=1``
        * ``X1``: time-varying covariates
        * ``p2_d10``: second period propensity scores if ``d1=0``
        * ``p2_d11``: second period propensity scores if ``d1=1``
        * ``p2``: second period propensity scores
        * ``D2``: second period treatment
        * ``Y``: final outcome
        * ``Y00``: potential outcome if ``d1=0`` and ``d2=0``
        * ``Y10``: potential outcome if ``d1=1`` and ``d2=0``
        * ``Y01``: potential outcome if ``d1=0`` and ``d2=1``
        * ``Y11``: potential outcome if ``d1=1`` and ``d2=1``

    Details
    -------
    The true effect of sequence 11 vs. sequence 00 equals
    $$ATE_{\\text{11-00}} = \\alpha_{Y_{11}} + \\sum_{j=1}^{r_0}
    \\gamma_{Y_{11}, j} - \\alpha_{Y_{00}} = -3$$

    References
    ----------
    [^Bradic]:
        Bradic, J., Ji, W., & Zhang, Y. (2024). High-dimensional inference
        for dynamic treatment effects. The Annals of Statistics, 52(2),
        415–440.
    """

    # Params
    alpha_p1 = 0
    beta_p1_signal = np.array([1, 1, 1])
    alpha_p2_d10 = 0
    beta_p2_d10_signal = np.array([.5, 0, -.5])
    gamma_p2_d10_signal = np.array([.5, 0, .5])
    alpha_p2_d11 = 0
    beta_p2_d11_signal = np.array([1, 1, 0])
    gamma_p2_d11_signal = np.array([1, -1, 0])
    alpha_Y00 = 1
    beta_Y00_signal = np.array([1, 1, -1])
    gamma_Y00_signal = np.array([1, -1, 0])
    alpha_Y11 = -1
    beta_Y11_signal = np.array([-1, 1, -1])
    gamma_Y11_signal = np.array([-1, -1, 1])

    # Random number generator
    rng = np.random.default_rng(random_state)

    # X0
    X0 = rng.multivariate_normal(np.zeros(r_0), np.identity(r_0), size=n)

    # D1
    # Coefficients for p1
    beta_p1 = np.concatenate([beta_p1_signal[:r_0], np.zeros(
        r_0-len(beta_p1_signal[:r_0]))]).reshape(r_0, -1)
    # Compute p1
    p1 = expit(alpha_p1 + X0 @ beta_p1)
    # Draw treatment
    D1 = rng.binomial(1, p1)

    # X1
    r_1 = r_0
    U = 1 + rng.standard_normal(n).reshape(n, -1)
    V = rng.multivariate_normal(np.zeros(r_1), np.identity(r_1), size=n)
    X1 = X0 + V + D1 * U * np.ones_like(X0)
    # potential covariates
    X1_d10 = X0 + V
    X1_d11 = X1_d10 + U * np.ones_like(X0)

    # D2
    # Coefficients for p2
    # if d1=0
    beta_p2_d10 = np.concatenate([beta_p2_d10_signal[:r_1], np.zeros(
        r_0-len(beta_p2_d10_signal[:r_1]))]).reshape(r_0, -1)
    gamma_p2_d10 = np.concatenate([gamma_p2_d10_signal[:r_1], np.zeros(
        r_0-len(gamma_p2_d10_signal[:r_1]))]).reshape(r_0, -1)
    # if d1=1
    beta_p2_d11 = np.concatenate([beta_p2_d11_signal[:r_1], np.zeros(
        r_0-len(beta_p2_d11_signal[:r_1]))]).reshape(r_0, -1)
    gamma_p2_d11 = np.concatenate([gamma_p2_d11_signal[:r_1], np.zeros(
        r_0-len(gamma_p2_d11_signal[:r_1]))]).reshape(r_0, -1)
    # compute p2
    p2_d10 = expit(alpha_p2_d10 + X0 @ beta_p2_d10 + X1_d10 @ gamma_p2_d10)
    p2_d11 = expit(alpha_p2_d11 + X0 @ beta_p2_d11 + X1_d11 @ gamma_p2_d11)
    p2 = D1 * p2_d11 + (1 - D1) * p2_d10
    # Draw treatment
    D2 = rng.binomial(1, p2)

    # Y
    # Coefficients for potential outcomes
    beta_Y00 = np.concatenate([beta_Y00_signal[:r_1], np.zeros(
        r_0-len(beta_Y00_signal[:r_1]))]).reshape(r_0, -1)
    gamma_Y00 = np.concatenate([gamma_Y00_signal[:r_1], np.zeros(
        r_0-len(gamma_Y00_signal[:r_1]))]).reshape(r_0, -1)
    beta_Y11 = np.concatenate([beta_Y11_signal[:r_1], np.zeros(
        r_0-len(beta_Y11_signal[:r_1]))]).reshape(r_0, -1)
    gamma_Y11 = np.concatenate([gamma_Y11_signal[:r_1], np.zeros(
        r_0-len(gamma_Y11_signal[:r_1]))]).reshape(r_0, -1)
    # Noise
    zeta = rng.standard_normal(n).reshape(n, -1)
    # compute Y, important that potential outcomes depend on POTENTIAL
    # covariates!
    Y00 = alpha_Y00 + X0 @ beta_Y00 + X1_d10 @ gamma_Y00
    Y11 = alpha_Y11 + X0 @ beta_Y11 + X1_d11 @ gamma_Y11
    Y = ((1 + D1 + D2) == 1)*Y00 + D1*D2*Y11 + zeta
    # Hence, the other two potential outcomes are just zeta
    Y01 = zeta
    Y10 = zeta

    # Collect generated data
    colnames = [
        'X0', 'p1', 'D1', 'X1_d10', 'X1_d11', 'X1', 'p2_d10', 'p2_d11', 'p2',
        'D2', 'Y', 'Y00', 'Y10', 'Y01', 'Y11']
    variables = [
        X0, p1.reshape(-1), D1.reshape(-1), X1_d10, X1_d11, X1,
        p2_d10.reshape(-1), p2_d11.reshape(-1), p2.reshape(-1),
        D2.reshape(-1), Y.reshape(-1), Y00.reshape(-1), Y10.reshape(-1),
        Y01.reshape(-1), Y11.reshape(-1)]
    return_dict = dict(zip(colnames, variables))
    return return_dict
