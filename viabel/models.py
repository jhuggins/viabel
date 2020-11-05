from autograd.extend import primitive, defvjp
import autograd.numpy as np

__all__ = [
    'Model',
    'StanModel'
]


class Model(object):
    def __init__(self, log_density):
        self._log_density = log_density

    def __call__(self, model_param):
        return self._log_density(model_param)

    def constrain(self, model_param):
        raise NotImplementedError()

    @property
    def supports_tempering(self):
        return False

    def set_inverse_temperature(self, inverse_temp):
        raise NotImplementedError()


def _vectorize_if_needed(f, a, axis=-1):
    if a.ndim > 1:
        return np.apply_along_axis(f, axis, a)
    else:
        return f(a)


def _ensure_2d(a):
    if a.ndim == 0:
        return a
    while a.ndim < 2:
        a = a[:,np.newaxis]
    return a


def _make_stan_log_density(fitobj):
    @primitive
    def log_density(x):
        return _vectorize_if_needed(fitobj.log_prob, x)
    def log_density_vjp(ans, x):
        return lambda g: _ensure_2d(g) * _vectorize_if_needed(fitobj.grad_log_prob, x)
    defvjp(log_density, log_density_vjp)
    return log_density


class StanModel(Model):
    def __init__(self, fit):
        self._fit = fit
        super().__init__(_make_stan_log_density(fit))

    def constrain(self, model_param):
        return self._fit.constrain_pars(model_param)
