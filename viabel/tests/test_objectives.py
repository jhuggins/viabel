from viabel import black_box_klvi, black_box_chivi, mean_field_t_variational_family, adagrad_optimize

import autograd.numpy as anp
import numpy as np
from autograd.scipy.stats import norm


def _test_vi(objective, n_samples, **kwargs):
    np.random.seed(851)
    # mean = np.random.randn(1,dimension)
    # stdev = np.exp(np.random.randn(1,dimension))
    mean = np.array([1.,-1.])[np.newaxis,:]
    stdev = np.array([2.,5.])[np.newaxis,:]
    log_p = lambda x: anp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)
    family = mean_field_t_variational_family(2, 100)
    objective_and_grad = objective(var_family=family, logdensity=log_p,
                                   n_samples=n_samples, **kwargs)
    # large number of MC samples and smaller epsilon and learning rate to ensure accuracy
    init_param = np.array([0., 0., 1., 1.])
    var_param, var_param_history, _, _ = adagrad_optimize(5000, objective_and_grad, init_param, epsilon=1e-8, learning_rate_end=.0001)
    # iterate averaging introduces some bias, so use last iterate
    est_mean, est_cov = family.mean_and_cov(var_param_history[-1])
    est_stdev = np.sqrt(np.diag(est_cov))
    np.testing.assert_almost_equal(mean.squeeze(), est_mean, decimal=2)
    np.testing.assert_almost_equal(stdev.squeeze(), est_stdev, decimal=2)


def test_klvi():
     _test_vi(black_box_klvi, 100)


def test_chivi():
    _test_vi(black_box_chivi, 100, alpha=2)


if __name__ == '__main__':
    test_klvi()
