from abc import ABC, abstractmethod

import jax.numpy as np
import numpy.random as npr
from jax import value_and_grad, vjp, grad, random, hessian, jacobian, vmap, jit
from functools import partial
from jax import device_get

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

    with reparameterized gradient estimator and control variate

    This implementation of reparameterization and control variate is based on:

    "Reducing Reparameterization Gradient Variance" by Andrew C. Miller, Nicholas J. Foti , Alexander D'Amour ,
    and Ryan P. Adams, Code based on the implementation by Andrew C. Miller:
    https://github.com/andymiller/ReducedVarianceReparamGradients

    """

    def __init__(self, approx, model, num_mc_samples, use_path_deriv=False, hessian_approx_method=None):
        """
        Parameters
        ----------
        approx : `ApproximationFamily` object
        model : `Model` object
        num_mc_sample : `int`
            Number of Monte Carlo samples to use to approximate the objective.
        use_path_deriv : `bool`
            Use path derivative (for "sticking the landing") gradient estimator
        hessian_approx_method : 'string'
            Select from different methods for approximating the hessian:
                'full' : use the full hessian matrix provided by BridgeStan
                'mean_only' : use control variate only for mean estimator to avoid calculation of full hessian
                'loo_diag_approx' : using "leave one out" method with hessian vector product value at other samples to
                    estimate the diagonal values of hessian
                'loo_direct_approx;: the same method as 'loo_diag_approx' but use the scaled approximation to the
                    gradient of scale to do the "loo" estimation
        """

        self._use_path_deriv = use_path_deriv
        if hessian_approx_method in [None, 'full', 'mean_only', 'loo_diag_approx', 'loo_direct_approx']:
            self.hessian_approx_method = hessian_approx_method
        else:
            raise ValueError("Name of approximation must be one of 'full', 'mean_only', 'loo_diag_approx', 'loo_direct_approx' or None object.")
        super().__init__(approx, model, num_mc_samples)

    def _update_objective_and_grad(self):
        approx = self.approx

        if self.hessian_approx_method is None:
            def variational_objective(var_param):
                samples = approx.sample(var_param, self.num_mc_samples)
                if self._use_path_deriv:
                    var_param_stopped = var_param.primal
                    var_param_stopped = device_get(var_param_stopped)
                    lower_bound = np.mean(
                        self.model(samples) - approx.log_density(var_param_stopped, samples))
                elif approx.supports_entropy:
                    lower_bound = np.mean(self.model(samples)) + approx.entropy(var_param)
                else:
                    lower_bound = np.mean(self.model(samples) - approx.log_density(samples))
                return -lower_bound

            def hessian_vector_product(f, x, v):
                return grad(lambda x: np.vdot(grad(f)(x), v))(x)
            
            def make_hvp(f,x):
                def hvp_for_v(v):
                    return hessian_vector_product(f, x, v)
                return hvp_for_v
            self._hvp = partial(make_hvp, variational_objective)
            self._objective_and_grad = value_and_grad(variational_objective)
            return

        def RGE(var_param):
            z_samples = approx.sample(var_param, self.num_mc_samples)
            m_mean, cov = approx.mean_and_cov(var_param)
            s_scale = np.sqrt(np.diag(cov))
            epsilon_sample = (z_samples - m_mean) / s_scale
            # elbo = np.mean(self._model(z_samples) - approx.log_density(var_param, z_samples))
            if self._use_path_deriv:
                var_param_stopped = device_get(var_param)
                lower_bound = np.mean(
                    self.model(z_samples) - approx.log_density(var_param_stopped, z_samples))
            elif approx.supports_entropy:
                lower_bound = np.mean(self.model(z_samples)) + approx.entropy(var_param)
            else:
                lower_bound = np.mean(self.model(z_samples) - approx.log_density(z_samples))

            # self.model takes in one single parameter to calculate grad and hessian
            def f_model(x):
                x = np.atleast_2d(x)
                return self._model(x)

            def make_hvp(f):
                hessian_f = jit(jacobian(jacobian(f)))

                def compute_hvp(x, a):
                    # Obtain the Hessian by evaluating the double Jacobian at x
                    hessian_at_x = hessian_f(x)
                    # Compute the HVP
                    return np.dot(hessian_at_x, a)

                return compute_hvp

            # estimate grad and hessian
            grad_f = jacobian(self.model)
            grad_f = vmap(grad_f)
            grad_f_single = jacobian(f_model)
            dLdm = grad_f(z_samples)
            dLdm = dLdm.reshape(z_samples.shape[0], z_samples.shape[1])
            # log-std
            # dLds = dLdm * epsilon_sample + 1 / s_scale
            dLdlns = dLdm * epsilon_sample * s_scale + 1
            # var_param MC gradient
            g_hat_rprm_grad = np.column_stack([dLdm, dLdlns])
            # These implementation of using reparameterization and control variate to reduce variation
            if self.hessian_approx_method == "full":
                hessian_f = hessian(f_model)
                # Miller's implementation
                gmu = grad_f(m_mean)
                gmu = gmu.squeeze()
                #gmu = gmu.reshape(z_samples.shape[0], z_samples.shape[1])
                H = hessian_f(m_mean).squeeze()
                Hdiag = np.diag(H)
                # construct normal approx samples of data term
                dLdz = gmu + np.dot(H, (s_scale * epsilon_sample).T).T
                # dLds  = (dLdz*eps + 1/s_lam[None,:]) * s_lam
                dLds = dLdz * epsilon_sample * s_scale + 1.
                elbo_gsamps_tilde = np.column_stack([dLdz, dLds])
                # characterize the mean of the dLds component (and z comp)
                dLds_mu = (Hdiag * s_scale + 1 / s_scale) * s_scale
                gsamps_tilde_mean = np.concatenate([gmu, dLds_mu])
                # subtract mean to compute control variate
                elbo_gsamps_cv = g_hat_rprm_grad - (elbo_gsamps_tilde - gsamps_tilde_mean)
                g_hat_rv = np.mean(elbo_gsamps_cv, axis=0)
            elif self.hessian_approx_method == "mean_only":
                # linear approximation of gradient: mean
                scaled_samples = np.multiply(s_scale, epsilon_sample)
                a = grad_f(m_mean * np.ones_like(z_samples))
                a = a.reshape(z_samples.shape[0], z_samples.shape[1])
                hvp = make_hvp(f_model)
                b = np.array([hvp(m_mean, s) for s in scaled_samples])
                b = b.reshape(z_samples.shape[0], z_samples.shape[1])
                g_tilde_mean_approx = a + b
                # linear approximation of gradient: log-scale
                g_tilde_scale_approx_ln = np.zeros_like(g_tilde_mean_approx)
                # Expectation of linear approximation of gradient: mean
                E_g_tilde_mean = grad_f_single(m_mean)
                E_g_tilde_mean = E_g_tilde_mean.squeeze()
                # Expectation of linear approximation of gradient: log-scale
                E_g_tilde_scale_ln = np.zeros_like(E_g_tilde_mean)
                g_tilde = np.column_stack([g_tilde_mean_approx, g_tilde_scale_approx_ln])
                E_g_tilde = np.concatenate([E_g_tilde_mean, E_g_tilde_scale_ln])
                E_g_tilde = np.multiply(E_g_tilde, np.ones_like(g_tilde))
                g_hat_rv = np.mean(g_hat_rprm_grad - (g_tilde - E_g_tilde), axis=0)
            elif self.hessian_approx_method == "loo_diag_approx":
                """ use other samples to estimate a per-sample diagonal
                expectation
                """
                # assert ns > 1, "loo approximations require more than 1 sample"
                # compute hessian vector products and save them for both parts
                hvp_lam = make_hvp(f_model)
                hvps = np.array([hvp_lam(m_mean, s_scale * e) for e in epsilon_sample])
                hvps = hvps.reshape(z_samples.shape[0], z_samples.shape[1])
                gmu = grad_f(m_mean * np.ones_like(z_samples))
                gmu = gmu.reshape(z_samples.shape[0], z_samples.shape[1])
                # construct normal approx samples of data term
                dLdz = gmu + hvps
                dLds = dLdz * (epsilon_sample * s_scale) + 1
                # compute Leave One Out approximate diagonal (per-sample mean of dLds)
                Hdiag_sum = np.sum(epsilon_sample * hvps, axis=0)
                Hdiag_s = (Hdiag_sum[None, :] - epsilon_sample * hvps) / float(np.shape(z_samples)[0] - 1)
                dLds_mu = (Hdiag_s + 1 / s_scale[None, :]) * s_scale
                # compute gsamps_cv - mean(gsamps_cv), and finally the var reduced
                D = int(0.5 * np.shape(g_hat_rprm_grad)[1])
                g_hat_rv = g_hat_rprm_grad.copy()
                g_hat_rv = g_hat_rv.at[:, :D].set(g_hat_rv[:, :D] - hvps)
                g_hat_rv = g_hat_rv.at[:, D:].set(g_hat_rv[:, D:] - (dLds - dLds_mu))
                g_hat_rv = np.mean(g_hat_rv, axis=0)
            elif self.hessian_approx_method == "loo_direct_approx":
                hvp_lam = make_hvp(f_model)
                gmu = grad_f(m_mean * np.ones_like(z_samples))
                gmu = gmu.reshape(z_samples.shape[0], z_samples.shape[1])
                hvps = np.array([hvp_lam(m_mean, s_scale * e)[0] for e in epsilon_sample])
                # construct normal approx samples of data term
                dLdz = gmu + hvps
                dLds = (dLdz * epsilon_sample + 1 / s_scale[None, :]) * s_scale
                # compute Leave One Out approximate diagonal (per-sample mean of dLds)
                dLds_sum = np.sum(dLds, axis=0)
                dLds_mu = (dLds_sum[None, :] - dLds) / float(np.shape(z_samples)[0] - 1)
                # compute gsamps_cv - mean(gsamps_cv), and finally the var reduced
                elbo_gsamps_tilde_centered = np.column_stack([hvps, dLds - dLds_mu])
                g_hat_rv = np.mean(g_hat_rprm_grad - elbo_gsamps_tilde_centered, axis=0)
            else:
                raise RuntimeError("Invalid hessian approximation method!")
            return -lower_bound, -g_hat_rv
        self._objective_and_grad = RGE

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
        w[to_clip] = self._w_clip_threshold * sum_unclipped(1. - self._w_clip_threshold * n_to_clip)
        return self._clip_weights(w)

    def _update_objective_and_grad(self):
        approx = self.approx
        
        def choice(key, a, p, size=None):
            """Sample from a with probabilities p using JAX"""
            cdf = np.cumsum(p)
            if isinstance(a, int):
                a = np.arange(a)
                if p is None:
                    p = np.ones(a.shape) / a.size
            if size is None:
                random_values = random.uniform(key)
            else:
                random_values = random.uniform(key, shape=(size,))
            indices = np.searchsorted(cdf, random_values)
            return a[indices]

        def variational_objective(var_param):
            if not self._use_resampling or self._objective_step % self._num_resampling_batches == 0:
                state_samples = approx.sample(var_param, self.num_mc_samples).primal
                self._state_samples = device_get(state_samples)
                self._state_log_q = approx.log_density(var_param, self._state_samples)
                self._state_log_p_unnormalized = self.model(self._state_samples)

                self._eps, ess, w = self._get_eps_and_weights(
                    self._eps, self._state_samples, self._state_log_p_unnormalized, self._state_log_q)
                self._state_w_clipped = self._clip_weights(w)
                self._state_w_sum = np.sum(self._state_w_clipped)
                self._state_w_normalized = self._state_w_clipped / self._state_w_sum

            self._objective_step += 1

            if not self._use_resampling:
                state_w = self._state_w_clipped.primal
                return -np.inner(evice_get(state_w), self._state_log_q) / self.num_mc_samples
            else:
                rng = random.PRNGKey(0)
                indices = choice(key=rng, a=self.num_mc_samples,
                                           size=self._resampling_batch_size, p=self._state_w_normalized)
                samples_resampled = self._state_samples[indices]
                
                obj = np.mean(-approx.log_density(var_param, samples_resampled))
                state_w_sum = self._state_w_sum.primal

                return obj * device_get(state_w_sum) / self.num_mc_samples

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

        def compute_log_weights(var_param):
            samples = self.approx.sample(var_param, self.num_mc_samples)
            log_weights = self.model(samples) - self.approx.log_density(var_param, samples)
            return log_weights

        #log_weights_vjp = vector_jacobian_product(compute_log_weights)
        alpha = self.alpha

        # manually compute objective and gradient

        def objective_grad_and_log_norm(var_param):
            log_weights = compute_log_weights(var_param)

            unary_compute_log_weights = partial(compute_log_weights)
            _, log_weights_vjp_fun = vjp(unary_compute_log_weights, var_param)
            log_norm = np.max(log_weights)
            scaled_values = np.exp(log_weights - log_norm) ** alpha
            log_weights_gradient = log_weights_vjp_fun(scaled_values)[0]
            obj_value = np.log(np.mean(scaled_values)) / alpha + log_norm
            obj_grad = alpha * log_weights_gradient / scaled_values.size
            return (obj_value, obj_grad)

        self._objective_and_grad = objective_grad_and_log_norm
