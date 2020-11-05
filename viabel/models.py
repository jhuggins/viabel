from autograd.extend import primitive, defvjp
import autograd.numpy as np

__all__ = [
    'Model',
    'StanModel'
]


class Model(object):
    """Base class for representing a model.

    Does not support tempering. It can be overridden in part or in whole by
    classes that inherit it. See ``StanModel`` for an example."""
    def __init__(self, log_density):
        """
        Parameters
        ----------
        log_density : `function`
            Function for computing the (unnormalized) log density of the model.
            Must support automatic differentiation with ``autograd``.
        """
        self._log_density = log_density

    def __call__(self, model_param):
        """Compute (unnormalized) log density of the model.

        Parameters
        ----------
        model_param : `numpy.ndarray`, shape (dim,)
            Model parameter value

        Returns
        -------
        log_density : `float`
        """
        return self._log_density(model_param)

    def constrain(self, model_param):
        """Construct dictionary of constrained parameters.

        Parameters
        ----------
        model_param : `numpy.ndarray`, shape (dim,)
            Model parameter value

        Returns
        -------
        constrained_params : `dict`

        Raises
        ------
        NotImplementedError
            If constrained parameterization is not supported.
        """
        raise NotImplementedError()

    @property
    def supports_tempering(self):
        """Whether the model supports tempering."""
        return False

    def set_inverse_temperature(self, inverse_temp):
        """If tempering supported, set inverse temperature.

        Parameters
        ----------
        inverse_temp : `float`

        Raises
        ------
        NotImplementedError
            If tempering is not supported.
        """
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
    """Class that encapsulates a PyStan model."""
    def __init__(self, fit):
        """
        Parameters
        ----------
        fit : `StanFit4model` object
        """
        self._fit = fit
        super().__init__(_make_stan_log_density(fit))

    def constrain(self, model_param):
        return self._fit.constrain_pars(model_param)
