from viabel import convenience
from viabel import Model, MFGaussian

import autograd.numpy as anp
import numpy as np
from autograd.scipy.stats import norm

import pytest


def test_bbvi():
    np.random.seed(851)
    mean = np.array([3.,-4.])[np.newaxis,:]
    stdev = np.array([2.,5.])[np.newaxis,:]
    log_p = lambda x: anp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)
    # large number of MC samples and smaller epsilon and learning rate to ensure accuracy
    results = convenience.bbvi(2, log_density=log_p, num_mc_samples=100)
    # iterate averaging introduces some bias, so use last iterate
    est_mean, est_cov = results['objective'].approx.mean_and_cov(results['var_param']) #results['var_param_history'][-1])
    est_stdev = np.sqrt(np.diag(est_cov))
    np.testing.assert_almost_equal(mean.squeeze(), est_mean, decimal=2)
    np.testing.assert_almost_equal(stdev.squeeze(), est_stdev, decimal=2)

    with pytest.raises(ValueError):
        convenience.bbvi(2)
    with pytest.raises(ValueError):
        convenience.bbvi(2, objective=True, fit=True)
    with pytest.raises(ValueError):
        convenience.bbvi(2, log_density=True, fit=True)
    with pytest.raises(ValueError):
        convenience.bbvi(2, objective=True, log_density=True)


def test_vi_diagnostics():
    np.random.seed(153)
    log_p = lambda x: anp.sum(norm.logpdf(x), axis=1)
    results = convenience.bbvi(2, log_density=log_p, num_mc_samples=100)
    diagnostics = convenience.vi_diagnostics(results['var_param'],
                                             objective=results['objective'])
    assert diagnostics['khat'] < .1
    assert diagnostics['d2'] < 0.1

    log_p2 = lambda x: anp.sum(norm.logpdf(x, scale=3), axis=1)
    model2 = Model(log_p2)
    diagnostics2 = convenience.vi_diagnostics(results['var_param'],
                                              approx=results['objective'].approx,
                                              model=model2)
    assert diagnostics2['khat'] > 0.7
    assert 'd2' not in diagnostics2

    log_p3 = lambda x: anp.sum(norm.logpdf(x, scale=.5), axis=1)
    model3 = Model(log_p3)
    diagnostics3 = convenience.vi_diagnostics(results['var_param'],
                                              approx=results['objective'].approx,
                                              model=model3)
    print(diagnostics3)
    assert diagnostics3['khat'] < 0  # weights are bounded
    assert diagnostics3['d2'] > 2
