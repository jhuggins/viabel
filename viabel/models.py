import jax
import numpy as np

from ._utils import ensure_2d, vectorize_if_needed

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


def _make_stan_log_density(fitobj):
    @jax.custom_vjp
    def log_density(x):
        return vectorize_if_needed(fitobj.log_density, x)

    def log_density_fwd(x):
        x = np.asarray(x)
        value, grad = vectorize_if_needed(fitobj.log_density_gradient, x)
        return log_density(x), grad

    def log_density_bwd(res, g):
        grad = res
        g = np.asarray(g, dtype=object)
        return ensure_2d(g) * grad,

    log_density.defvjp(log_density_fwd, log_density_bwd)
    return log_density

class StanModel(Model):
    """Class that encapsulates a BridgeStan model."""

    def __init__(self, fit):
        """
        Parameters
        ----------
        fit : `StanFit4model` object
        """
        self._fit = fit
        super().__init__(_make_stan_log_density(fit))

    def constrain(self, model_param):
        return self._fit.param_constrain(model_param)
        return self._fit.constrain_pars(model_param)
