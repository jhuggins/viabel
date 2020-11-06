from viabel import convenience

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
    results = convenience.bbvi(2, log_density=log_p, num_mc_samples=100,
                               epsilon=1e-8, learning_rate_end=.0001)
    # iterate averaging introduces some bias, so use last iterate
    est_mean, est_cov = results['objective'].approx.mean_and_cov(results['var_param_history'][-1])
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
        convenience.bbvi(2, log_density=True, objective=True)
