from abc import ABC, abstractmethod

from autograd import value_and_grad, vector_jacobian_product
from autograd.core import getval

import autograd.numpy as np
import autograd.numpy.random as npr

__all__ = [
    'VariationalObjective',
    'StochasticVariationalObjective',
    'ExclusiveKL',
    'DISInclusiveKL',
    'AlphaDivergence'
]


class VariationalObjective(ABC):
    """A class representing a variational objective to minimize"""
    def __init__(self, approx, model):
        """
        Parameters
        ----------
        approx : `ApproximationFamily` object
        model : `Model` object
        """
        self._approx = approx
        self._model = model
        self._objective_and_grad = None
        self._update_objective_and_grad()


    def __call__(self, var_param):
        """Evaluate objective and its gradient.

        May produce an (un)biased estimate of both.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        """
        if self._objective_and_grad is None:
            raise RuntimeError("no objective and gradient available")
        return self._objective_and_grad(var_param)

    @abstractmethod
    def _update_objective_and_grad(self):
        """Update the objective and gradient function.

        Should be called whenever a parameter that the objective depends on
        (e.g., `approx` or `model`) is updated."""
        pass

    @property
    def approx(self):
        """The approximation family."""
        return self._approx

    @approx.setter
    def approx(self, value):
        self._approx = value
        self._update_objective_and_grad()

    @property
    def model(self):
        """The model."""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._update_objective_and_grad()


class StochasticVariationalObjective(VariationalObjective):
    """A class representing a variational objective approximated using Monte Carlo."""
    def __init__(self, approx, model, num_mc_samples):
        """
        Parameters
        ----------
        approx : `ApproximationFamily` object
        model : `Model` object
        num_mc_sample : `int`
            Number of Monte Carlo samples to use to approximate the objective.
        """
        self._num_mc_samples = num_mc_samples
        super().__init__(approx, model)

    @property
    def num_mc_samples(self):
        """Number of Monte Carlo samples to use to approximate the objective."""
        return self._num_mc_samples

    @num_mc_samples.setter
    def num_mc_samples(self, value):
        self._num_mc_samples = value
        self._update_objective_and_grad()


class ExclusiveKL(StochasticVariationalObjective):
    """Exclusive Kullback-Leibler divergence.

    Equivalent to using the canonical evidence lower bound (ELBO)
    """
    def __init__(self, approx, model, num_mc_samples, use_path_deriv=False):
            """
            Parameters
            ----------
            approx : `ApproximationFamily` object
            model : `Model` object
            num_mc_sample : `int`
                Number of Monte Carlo samples to use to approximate the objective.
            use_path_deriv : `bool`
                Use path derivative (for "sticking the landing") gradient estimator
            """
            self._use_path_deriv = use_path_deriv
            super().__init__(approx, model, num_mc_samples)

    def _update_objective_and_grad(self):
        approx = self.approx
        def variational_objective(var_param):
            samples = approx.sample(var_param, self.num_mc_samples)
            if self._use_path_deriv:
                var_param_stopped = getval(var_param)
                lower_bound = np.mean(self.model(samples) - approx.log_density(var_param_stopped, samples))
            elif approx.supports_entropy:
                lower_bound = np.mean(self.model(samples)) + approx.entropy(var_param)
            else:
                lower_bound = np.mean(self.model(samples) - approx.log_density(samples))
            return -lower_bound
        self._objective_and_grad = value_and_grad(variational_objective)


class DISInclusiveKL(StochasticVariationalObjective):
    """Inclusive Kullback-Leibler divergence using Distilled Importance Sampling."""
    def __init__(self, approx, model, num_mc_samples, use_path_deriv=False, w_clip_threshold=10):
            """
            Parameters
            ----------
            approx : `ApproximationFamily` object
            model : `Model` object
            num_mc_sample : `int`
                Number of Monte Carlo samples to use to approximate the objective.
            use_path_deriv : `bool`
                Use path derivative (for "sticking the landing") gradient estimator
            """
            self._use_path_deriv = use_path_deriv
            self._w_clip_threshold = w_clip_threshold
            super().__init__(approx, model, num_mc_samples)

    def _update_objective_and_grad(self):
        approx = self.approx
        def variational_objective(var_param):
            samples = approx.sample(var_param, self.num_mc_samples)
            if self._use_path_deriv:
                var_param_stopped = getval(var_param)
                log_density_q = approx.log_density(var_param_stopped, samples)
            else:
                log_density_q = approx.log_density(samples)

            log_density_p_unnormalized = self.model(samples)
            log_w = log_density_p_unnormalized - log_density_q
            w_clipped = np.exp(log_w).clip(0, self._w_clip_threshold)
            obj = np.inner(w_clipped, log_density_q) / self.num_mc_samples
            return obj

        self._objective_and_grad = value_and_grad(variational_objective)


class AlphaDivergence(StochasticVariationalObjective):
    """Log of the alpha-divergence."""
    def __init__(self, approx, model, num_mc_samples, alpha):
        """
        Parameters
        ----------
        approx : `ApproximationFamily` object
        model : `Model` object
        num_mc_sample : `int`
            Number of Monte Carlo samples to use to approximate the objective.
        alpha : `float`
        """
        self._alpha = alpha
        super().__init__(approx, model, num_mc_samples)

    @property
    def alpha(self):
        """Alpha parameter of the divergence."""
        return self._alpha

    def _update_objective_and_grad(self):
        """Provides a stochastic estimate of the variational lower bound."""
        def compute_log_weights(var_param, seed):
            samples = self.approx.sample(var_param, self.num_mc_samples, seed)
            log_weights = self.model(samples) - self.approx.log_density(var_param, samples)
            return log_weights

        log_weights_vjp = vector_jacobian_product(compute_log_weights)
        alpha = self.alpha
        # manually compute objective and gradient
        def objective_grad_and_log_norm(var_param):
            # must create a shared seed!
            seed = npr.randint(2**32)
            log_weights = compute_log_weights(var_param, seed)
            log_norm = np.max(log_weights)
            scaled_values = np.exp(log_weights - log_norm)**alpha
            obj_value = np.log(np.mean(scaled_values))/alpha + log_norm
            obj_grad = alpha*log_weights_vjp(var_param, seed, scaled_values) / scaled_values.size
            return (obj_value, obj_grad)

        self._objective_and_grad = objective_grad_and_log_norm
