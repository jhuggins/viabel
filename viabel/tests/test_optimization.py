from viabel.optimization import FASO, RMSProp, AdaGrad, WindowedAdaGrad
from viabel.objectives import VariationalObjective

import autograd.numpy as anp
import numpy as np
from autograd import grad


class DummyApproximationFamily:
    def __init__(self):
        self.supports_kl = True

    def kl(self, param1, param2):
        return np.mean((param1-param2)**2)


class DummyObjective:
    """Simple quadratic dummy objective with artifical Gaussian gradient noise"""
    def __init__(self, target, noise=1, scales=1):
        self._noise = noise
        self.objective_fun = lambda x: .5*anp.sum(((x-target)/scales)**2)
        self.grad_objective_fun = grad(self.objective_fun)
        self.approx = DummyApproximationFamily()

    def __call__(self, x):
        noisy_grad = self.grad_objective_fun(x) + self._noise*np.random.randn(x.size)
        return self.objective_fun(x), noisy_grad


def _test_optimizer(opt_class, objective, true_value, n_iters, **kwargs):
    np.random.seed(851)
    dim = true_value.size
    init_param = true_value + np.random.randn(dim) / np.sqrt(dim)
    results = opt_class.optimize(n_iters, objective, init_param)
    np.testing.assert_almost_equal(results['opt_param'], true_value, decimal=2)


def test_rmsprop_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = RMSProp(0.01)
        _test_optimizer(sgd, objective, true_value, 20000)


def test_adagrad_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = AdaGrad(0.1)
        _test_optimizer(sgd, objective, true_value, 20000)


def test_windowed_adagrad_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = WindowedAdaGrad(0.01)
        _test_optimizer(sgd, objective, true_value, 20000)


def test_faso_rmsprop_optimize():
    for scales in [np.ones(2), np.ones(4), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = FASO(RMSProp(0.01), mcse_threshold=.002)
        _test_optimizer(sgd, objective, true_value, 20000)
