from abc import ABC, abstractmethod

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad, vector_jacobian_product, make_hvp
from autograd.core import getval

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
    
    def _hessian_vector_product(self, var_param, x):
        """Compute hessian vector product at given variaitonal parameter point x. """
        pass
    
    def update(self, var_param, direction):
        """Update the variational parameter in optimization."""
        return var_param - direction
        
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
                lower_bound = np.mean(
                    self.model(samples) - approx.log_density(var_param_stopped, samples))
            elif approx.supports_entropy:
                lower_bound = np.mean(self.model(samples)) + approx.entropy(var_param)
            else:
                lower_bound = np.mean(self.model(samples) - approx.log_density(samples))
            return -lower_bound
        self._hvp = make_hvp(variational_objective)
        self._objective_and_grad = value_and_grad(variational_objective)

    def _hessian_vector_product(self, var_param, x):
        hvp_fun = self._hvp(var_param)[0]
        return hvp_fun(x)


class DISInclusiveKL(StochasticVariationalObjective):
    """Inclusive Kullback-Leibler divergence using Distilled Importance Sampling."""

    def __init__(self, approx, model, num_mc_samples, ess_target,
                 temper_prior, temper_prior_params, use_resampling=True,
                 num_resampling_batches=1, w_clip_threshold=10):
        """
        Parameters
        ----------
        approx : `ApproximationFamily` object
        model : `Model` object
        num_mc_sample : `int`
            Number of Monte Carlo samples to use to approximate the KL divergence.
            (N in the paper)
        ess_target: `int`
            The ess target to adjust epsilon (M in the paper). It is also the number of
            samples in resampling.
        temper_prior: `Model` object
            A prior distribution to temper the model. Typically multivariate normal.
        temper_prior_params: `numpy.ndarray` object
            Parameters for the temper prior. Typically mean 0 and variance 1.
        use_resampling: `bool`
            Whether to use resampling.
        num_resampling_batches: `int`
            Number of resampling batches. The resampling batch is `max(1, ess_target / num_resampling_batches)`.
        w_clip_threshold: `float`
            The maximum weight.
        """
        self._ess_target = ess_target
        self._w_clip_threshold = w_clip_threshold
        self._max_bisection_its = 50
        self._max_eps = self._eps = 1
        self._use_resampling = use_resampling
        self._num_resampling_batches = num_resampling_batches
        self._resampling_batch_size = max(1, self._ess_target // num_resampling_batches)
        self._objective_step = 0

        self._tempered_model_log_pdf = lambda eps, samples, log_p_unnormalized: (
            eps * temper_prior.log_density(temper_prior_params, samples)
            + (1 - eps) * log_p_unnormalized)
        super().__init__(approx, model, num_mc_samples)

    def _get_weights(self, eps, samples, log_p_unnormalized, log_q):
        """Calculates normalised importance sampling weights"""
        logw = self._tempered_model_log_pdf(eps, samples, log_p_unnormalized) - log_q
        max_logw = np.max(logw)
        if max_logw == -np.inf:
            raise ValueError('All weights zero! '
                             + 'Suggests overflow in importance density.')

        w = np.exp(logw)
        return w

    def _get_ess(self, w):
        """Calculates effective sample size of normalised importance sampling weights"""
        ess = (np.sum(w) ** 2.0) / np.sum(w ** 2.0)
        return ess

    def _get_eps_and_weights(self, eps_guess, samples, log_p_unnormalized, log_q):
        """Find new epsilon value

        Uses bisection to find epsilon < eps_guess giving required ESS.
        If none exists, returns eps_guess.

        Returns new epsilon value and corresponding ESS and normalised importance sampling weights.
        """

        lower = 0.
        upper = eps_guess
        eps_guess = (lower + upper) / 2.
        for i in range(self._max_bisection_its):
            w = self._get_weights(eps_guess, samples, log_p_unnormalized, log_q)
            ess = self._get_ess(w)
            if ess > self._ess_target:
                upper = eps_guess
            else:
                lower = eps_guess
            eps_guess = (lower + upper) / 2.

        w = self._get_weights(eps_guess, samples, log_p_unnormalized, log_q)
        ess = self._get_ess(w)

        # Consider returning extreme epsilon values if they are still endpoints
        if lower == 0.:
            eps_guess = 0.
        if upper == self._max_eps:
            eps_guess = self._max_eps

        return eps_guess, ess, w

    def _clip_weights(self, w):
        """Clip weights to `self._w_clip_threshold`
        Other weights are scaled up proportionately to keep sum equal to 1"""
        S = np.sum(w)
        if not any(w > S * self._w_clip_threshold):
            return w

        to_clip = (w >= S * self._w_clip_threshold)  # nb clip those equal to max_weight
        # so we don't push them over it!
        n_to_clip = np.sum(to_clip)
        to_not_clip = np.logical_not(to_clip)
        sum_unclipped = np.sum(w[to_not_clip])
        if sum_unclipped == 0:
            # Impossible to clip further!
            return w
        w[to_clip] = self._w_clip_threshold * sum_unclipped \
            / (1. - self._w_clip_threshold * n_to_clip)
        return self._clip_weights(w)

    def _update_objective_and_grad(self):
        approx = self.approx

        def variational_objective(var_param):
            if not self._use_resampling or self._objective_step % self._num_resampling_batches == 0:
                self._state_samples = getval(approx.sample(var_param, self.num_mc_samples))
                self._state_log_q = approx.log_density(var_param, self._state_samples)
                self._state_log_p_unnormalized = self.model(self._state_samples)

                self._eps, ess, w = self._get_eps_and_weights(
                    self._eps, self._state_samples, self._state_log_p_unnormalized, self._state_log_q)
                self._state_w_clipped = self._clip_weights(w)
                self._state_w_sum = np.sum(self._state_w_clipped)
                self._state_w_normalized = self._state_w_clipped / self._state_w_sum

            self._objective_step += 1

            if not self._use_resampling:
                return -np.inner(getval(self._state_w_clipped), self._state_log_q) / self.num_mc_samples
            else:
                indices = np.random.choice(self.num_mc_samples,
                                           size=self._resampling_batch_size, p=getval(self._state_w_normalized))
                samples_resampled = self._state_samples[indices]

                obj = np.mean(-approx.log_density(var_param, getval(samples_resampled)))

                return obj * getval(self._state_w_sum) / self.num_mc_samples

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
            obj_value = np.log(np.mean(scaled_values)) / alpha + log_norm
            obj_grad = alpha * log_weights_vjp(var_param, seed, scaled_values) / scaled_values.size
            return (obj_value, obj_grad)

        self._objective_and_grad = objective_grad_and_log_norm
