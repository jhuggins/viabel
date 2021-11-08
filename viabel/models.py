from autograd.extend import defvjp, primitive

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


def _make_stan_log_density(model):
    @primitive
    def log_density(x):
        return vectorize_if_needed(lambda t: model.log_prob(t.tolist()), x)

    def log_density_vjp(ans, x):
        return lambda g: ensure_2d(
            g) * vectorize_if_needed(lambda t: model.grad_log_prob(t.tolist()), x)
    defvjp(log_density, log_density_vjp)
    return log_density


class StanModel(Model):
    """Class that encapsulates a PyStan model."""

    def __init__(self, model):
        """
        Parameters
        ----------
        model : `stan.Model` object
        """
        self._model = model
        super().__init__(_make_stan_log_density(model))

    def constrain(self, model_param):
        return self._model.constrain_pars(model_param.tolist())
