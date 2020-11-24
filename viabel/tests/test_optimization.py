from viabel.optimization import adagrad_optimize
from viabel.objectives import VariationalObjective

import autograd.numpy as anp
import numpy as np
from autograd import grad


class DummyObjective:
    """Simple quadratic dummy objective with artifical Gaussian gradiet noise"""
    def __init__(self, noise=1, scales=1):
        self._noise = noise
        self.objective_fun = lambda x: .5*anp.sum((x/scales)**2)
        self.grad_objective_fun = grad(self.objective_fun)

    def __call__(self, x):
        noisy_grad = self.grad_objective_fun(x) + self._noise*np.random.randn(x.size)
        return self.objective_fun(x), noisy_grad


def _test_optimizer(opt, objective, true_value, n_iters, **kwargs):
    np.random.seed(851)
    dim = true_value.size
    init_param = true_value + np.random.randn(dim) / np.sqrt(dim)
    var_param, var_param_history, _, _ = opt(n_iters, objective, init_param, **kwargs)
    # var_param_history[-1]
    np.testing.assert_almost_equal(var_param, true_value, decimal=2)


def test_adagrad_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        objective = DummyObjective(noise=.2, scales=scales)
        true_value = np.zeros_like(scales)
        _test_optimizer(adagrad_optimize, objective, true_value,
                        40000, epsilon=1e-8,
                        learning_rate_end=.0001)
