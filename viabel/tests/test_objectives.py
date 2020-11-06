from viabel.approximations import MFStudentT
from viabel.optimization import adagrad_optimize
from viabel.objectives import ExclusiveKL, AlphaDivergence

import autograd.numpy as anp
import numpy as np
from autograd.scipy.stats import norm


def _test_vi(objective_cls, num_mc_samples, **kwargs):
    np.random.seed(851)
    # mean = np.random.randn(1,dimension)
    # stdev = np.exp(np.random.randn(1,dimension))
    mean = np.array([1.,-1.])[np.newaxis,:]
    stdev = np.array([2.,5.])[np.newaxis,:]
    log_p = lambda x: anp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)
    approx = MFStudentT(2, 100)
    objective = objective_cls(approx, log_p, num_mc_samples, **kwargs)
    # large number of MC samples and smaller epsilon and learning rate to ensure accuracy
    init_param = np.array([0, 0, 1, 1], dtype=np.float32)
    var_param, var_param_history, _, _ = adagrad_optimize(5000, objective, init_param, epsilon=1e-8, learning_rate_end=.0001)
    # iterate averaging introduces some bias, so use last iterate
    est_mean, est_cov = approx.mean_and_cov(var_param_history[-1])
    est_stdev = np.sqrt(np.diag(est_cov))
    print(est_stdev, stdev)
    np.testing.assert_almost_equal(mean.squeeze(), est_mean, decimal=2)
    np.testing.assert_almost_equal(stdev.squeeze(), est_stdev, decimal=2)


def test_objective():
     _test_vi(ExclusiveKL, 100)


def test_AlphaDivergence():
    _test_vi(AlphaDivergence, 100, alpha=2)
