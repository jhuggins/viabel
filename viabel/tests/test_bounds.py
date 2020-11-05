import viabel

import numpy as np
from scipy.stats import norm

MC_SAMPLES = 10000000
MC_TOL = 5/np.sqrt(MC_SAMPLES)


def _gaussian_alpha_divergence(alpha, var1, var2):
    """Compute D_alpha(N(0, var1) | N(0, var2))"""
    tmp = alpha*var2 - (alpha - 1)*var1
    print('tmp =', tmp)
    if tmp < 0: # pragma: no cover
        return np.inf
    return -0.5 / (alpha - 1) * np.log(tmp) + .5*alpha/(alpha - 1)*np.log(var2)  - .5*np.log(var1)

def _gaussian_kl_divergence(var1, var2):
    return .5*(var1/var2 + np.log(var2/var1) - 1)

def test_divergence_bound():
    np.random.seed(846)
    var1 = 4
    var2 = 16
    p1 = norm(scale=np.sqrt(var1))
    p2 = norm(scale=np.sqrt(var2))
    samples = p2.rvs(MC_SAMPLES)
    log_weights = p1.logpdf(samples) - p2.logpdf(samples)
    for alpha in [1.5, 2, 3]:
        print('alpha =', alpha)
        for elbo in [None, 0]:
            expected_dalpha = _gaussian_alpha_divergence(alpha, var1, var2)
            if elbo is None:
                expected_dalpha += alpha/(alpha - 1)*_gaussian_kl_divergence(var2, var1)
            np.testing.assert_allclose(
                viabel.divergence_bound(log_weights, alpha, elbo),
                expected_dalpha,
                atol=MC_TOL, rtol=MC_TOL, err_msg='incorrect d2 value')


def test_wasserstein_bounds():
    np.random.seed(341)
    d2 = 5.0
    stdev = 3.5
    samples = norm.rvs(scale=stdev, size=MC_SAMPLES)
    res = viabel.wasserstein_bounds(d2, samples)
    np.testing.assert_allclose(res['W1'], 2*stdev*np.sqrt(np.expm1(d2)),
                               rtol=MC_TOL, err_msg='incorrect W1 value')
    np.testing.assert_allclose(res['W2'], 2*stdev*(3*np.expm1(d2))**0.25,
                               rtol=MC_TOL, err_msg='incorrect W2 value')


def test_all_bounds():
    np.random.seed(1639)
    var1 = 2.5
    var2 = 9.3
    p1 = norm(scale=np.sqrt(var1))
    p2 = norm(scale=np.sqrt(var2))
    samples = p2.rvs(MC_SAMPLES)
    log_weights = p1.logpdf(samples) - p2.logpdf(samples)
    res = viabel.all_bounds(log_weights, samples, q_var=var2, log_norm_bound=None)
    print('KL =', _gaussian_kl_divergence(var2, var1))
    expected_d2 = _gaussian_alpha_divergence(2, var1, var2) + 2*_gaussian_kl_divergence(var2, var1)
    np.testing.assert_allclose(res['d2'], expected_d2,
                               rtol=MC_TOL, err_msg='incorrect d2 value')
    stdev2 = np.sqrt(var2)
    np.testing.assert_allclose(res['W1'], 2*stdev2*np.sqrt(np.expm1(res['d2'])),
                               rtol=MC_TOL, err_msg='incorrect W1 value')
    np.testing.assert_allclose(res['W2'], 2*stdev2*(3*np.expm1(res['d2']))**0.25,
                               rtol=MC_TOL, err_msg='incorrect W2 value')
