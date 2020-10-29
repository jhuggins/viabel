from autograd import value_and_grad, vector_jacobian_product

import autograd.numpy as np
import autograd.numpy.random as npr

__all__ = [
    'black_box_klvi',
    'black_box_chivi',
]


def black_box_klvi(var_family, logdensity, n_samples):
    def variational_objective(var_param):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples)
        lower_bound = var_family.entropy(var_param) + np.mean(logdensity(samples))
        return -lower_bound

    objective_and_grad = value_and_grad(variational_objective)

    return objective_and_grad


def black_box_chivi(alpha, var_family, logdensity, n_samples):
    def compute_log_weights(var_param, seed):
        """Provides a stochastic estimate of the variational lower bound."""
        samples = var_family.sample(var_param, n_samples, seed)
        log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
        return log_weights

    log_weights_vjp = vector_jacobian_product(compute_log_weights)

    def objective_grad_and_log_norm(var_param):
        seed = npr.randint(2**32)
        log_weights = compute_log_weights(var_param, seed)
        log_norm = np.max(log_weights)
        scaled_values = np.exp(log_weights - log_norm)**alpha
        obj_value = np.log(np.mean(scaled_values))/alpha + log_norm
        obj_grad = alpha*log_weights_vjp(var_param, seed, scaled_values) / scaled_values.size
        return (obj_value, obj_grad)

    return objective_grad_and_log_norm
