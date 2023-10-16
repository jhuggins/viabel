import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad

from viabel.optimization import (
    RAABBVI, FASO, Adagrad, RMSProp, Adam,
    AveragedAdam, AveragedRMSProp,
    StochasticGradientOptimizer, WindowedAdagrad)


class DummyApproximationFamily:
    def __init__(self):
        self.supports_kl = True

    def kl(self, param1, param2):
        return np.mean((param1 - param2)**2)


class DummyObjective:
    """Simple quadratic dummy objective with artifical Gaussian gradient noise"""

    def __init__(self, target, noise=1, scales=1):
        self._noise = noise
        self.objective_fun = lambda x: .5 * jnp.sum(((x - target) / scales)**2)
        self.grad_objective_fun = grad(self.objective_fun)
        self.approx = DummyApproximationFamily()
        self.update = lambda x,y: x - y

    def __call__(self, x):
        noisy_grad = self.grad_objective_fun(x) + self._noise * np.random.randn(x.size)
        return self.objective_fun(x), noisy_grad


def _test_optimizer(opt_class, objective, true_value, n_iters, **kwargs):
    np.random.seed(851)
    dim = true_value.size
    init_param = true_value + np.random.randn(dim) / np.sqrt(dim)
    results = opt_class.optimize(n_iters, objective, init_param)
    jnp.allclose(results['opt_param'], true_value)


def test_sgo_optimize():
    for scales in [np.ones(1), np.ones(3)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = StochasticGradientOptimizer(0.01, diagnostics=True)
        _test_optimizer(sgd, objective, true_value, 1000)


def test_sgo_error_checks():
    with pytest.raises(ValueError):
        StochasticGradientOptimizer(0.01, iterate_avg_prop=0)
    with pytest.raises(ValueError):
        StochasticGradientOptimizer(0.01, iterate_avg_prop=1.01)


def test_rmsprop_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = RMSProp(0.01)
        _test_optimizer(sgd, objective, true_value, 1000)

def test_adam_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = Adam(0.01)
        _test_optimizer(sgd, objective, true_value, 1000)

def test_adagrad_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = Adagrad(0.1)
        _test_optimizer(sgd, objective, true_value, 1000)


def test_windowed_adagrad_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = WindowedAdagrad(0.01)
        _test_optimizer(sgd, objective, true_value, 1000)


def test_avgrmsprop_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = AveragedRMSProp(0.01)
        _test_optimizer(sgd, objective, true_value, 1000)


def test_avgadam_optimize():
    for scales in [np.ones(1), np.ones(3), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = AveragedAdam(0.01)
        _test_optimizer(sgd, objective, true_value, 1000)


def test_faso_rmsprop_optimize():
    for scales in [np.ones(2), np.ones(4), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = FASO(RMSProp(0.01, diagnostics=True), mcse_threshold=.002)
        _test_optimizer(sgd, objective, true_value, 1000)


def test_raabbvi_avgrmsprop_optimize():
    for scales in [np.ones(2), np.ones(4), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = RAABBVI(AveragedRMSProp(0.01, diagnostics=True), rho=0.5, mcse_threshold=.002, 
                inefficiency_threshold=1.0, accuracy_threshold=0.002)
        _test_optimizer(sgd, objective, true_value, 1000)
  
        
def test_raabbvi_avgadam_optimize():
    for scales in [np.ones(2), np.ones(4), np.geomspace(.1, 1, 4)]:
        true_value = np.arange(scales.size)
        objective = DummyObjective(true_value, noise=.2, scales=scales)
        sgd = RAABBVI(AveragedAdam(0.01, diagnostics=True), rho=0.5, mcse_threshold=.002, 
                inefficiency_threshold=1.0, accuracy_threshold=0.002)
        _test_optimizer(sgd, objective, true_value, 1000)
        
        
def test_faso_error_checks():
    with pytest.raises(ValueError):
        FASO(FASO(RMSProp(0.01)))
    with pytest.raises(ValueError):
        FASO(RMSProp(0.01), mcse_threshold=0)
    with pytest.raises(ValueError):
        FASO(RMSProp(0.01), W_min=0)
    with pytest.raises(ValueError):
        FASO(RMSProp(0.01), k_check=0)
    with pytest.raises(ValueError):
        FASO(RMSProp(0.01), ESS_min=0)
        
