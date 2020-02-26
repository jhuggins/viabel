from collections import namedtuple

from autograd import value_and_grad, vector_jacobian_product
from autograd.extend import primitive, defvjp

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist
from autograd.scipy.linalg import sqrtm
from scipy.linalg import eigvalsh

from paragami import (PatternDict,
                      NumericVectorPattern,
                      PSDSymmetricMatrixPattern,
                      FlattenFunctionInput)

import tqdm

from ._distributions import multivariate_t_logpdf

__all__ = [
    'mean_field_gaussian_variational_family',
    'mean_field_t_variational_family',
    't_variational_family',
    'black_box_klvi',
    'black_box_chivi',
    'make_stan_log_density',
    'adagrad_optimize',
]

VariationalFamily = namedtuple('VariationalFamily',
                               ['sample', 'entropy',
                                'logdensity', 'mean_and_cov',
                                'pth_moment', 'var_param_dim'])


def mean_field_gaussian_variational_family(dim):
    rs = npr.RandomState(0)
    def unpack_params(var_param):
        mean, log_std = var_param[:dim], var_param[dim:]
        return mean, log_std

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        mean, log_std = unpack_params(var_param)
        return my_rs.randn(n_samples, dim) * np.exp(log_std) + mean

    def entropy(var_param):
        mean, log_std = unpack_params(var_param)
        return 0.5 * dim * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    def logdensity(x, var_param):
        mean, log_std = unpack_params(var_param)
        return mvn.logpdf(x, mean, np.diag(np.exp(2*log_std)))

    def mean_and_cov(var_param):
        mean, log_std = unpack_params(var_param)
        return mean, np.diag(np.exp(2*log_std))

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        _, log_std = unpack_params(var_param)
        vars = np.exp(2*log_std)
        if p == 2:
            return np.sum(vars)
        else:  # p == 4
            return 2*np.sum(vars**2) + np.sum(vars)**2

    return VariationalFamily(sample, entropy, logdensity,
                             mean_and_cov, pth_moment, 2*dim)


def mean_field_t_variational_family(dim, df):
    if df <= 2:
        raise ValueError('df must be greater than 2')
    rs = npr.RandomState(0)
    def unpack_params(var_param):
        mean, log_scale = var_param[:dim], var_param[dim:]
        return mean, log_scale

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        mean, log_scale = unpack_params(var_param)
        return mean + np.exp(log_scale)*my_rs.standard_t(df, size=(n_samples, dim))

    def entropy(var_param):
        # ignore terms that depend only on df
        mean, log_scale = unpack_params(var_param)
        return np.sum(log_scale)

    def logdensity(x, var_param):
        mean, log_scale = unpack_params(var_param)
        if x.ndim == 1:
            x = x[np.newaxis,:]
        return np.sum(t_dist.logpdf(x, df, mean, np.exp(log_scale)), axis=-1)

    def mean_and_cov(var_param):
        mean, log_scale = unpack_params(var_param)
        return mean, df / (df - 2) * np.diag(np.exp(2*log_scale))

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        if df <= p:
            raise ValueError('df must be greater than p')
        _, log_scale = unpack_params(var_param)
        scales = np.exp(log_scale)
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(scales**2)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(scales**4) + np.sum(scales**2)**2)

    return VariationalFamily(sample, entropy, logdensity,
                             mean_and_cov, pth_moment, 2*dim)


def _get_mu_sigma_pattern(dim):
    ms_pattern = PatternDict(free_default=True)
    ms_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_pattern['Sigma'] = PSDSymmetricMatrixPattern(size=dim)
    return ms_pattern


def t_variational_family(dim, df):
    if df <= 2:
        raise ValueError('df must be greater than 2')
    rs = npr.RandomState(0)
    ms_pattern = _get_mu_sigma_pattern(dim)

    logdensity = FlattenFunctionInput(
        lambda x, ms_dict: multivariate_t_logpdf(x, ms_dict['mu'], ms_dict['Sigma'], df),
        patterns=ms_pattern, free=True, argnums=1)

    def sample(var_param, n_samples, seed=None):
        my_rs = rs if seed is None else npr.RandomState(seed)
        s = np.sqrt(my_rs.chisquare(df, n_samples) / df)
        param_dict = ms_pattern.fold(var_param)
        z = my_rs.randn(n_samples, dim)
        sqrtSigma = sqrtm(param_dict['Sigma'])
        return param_dict['mu'] + np.dot(z, sqrtSigma)/s[:,np.newaxis]

    def entropy(var_param):
        # ignore terms that depend only on df
        param_dict = ms_pattern.fold(var_param)
        return .5*np.log(np.linalg.det(param_dict['Sigma']))

    def mean_and_cov(var_param):
        param_dict = ms_pattern.fold(var_param)
        return param_dict['mu'], df / (df - 2.) * param_dict['Sigma']

    def pth_moment(p, var_param):
        if p not in [2,4]:
            raise ValueError('only p = 2 or 4 supported')
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = ms_pattern.fold(var_param)
        sq_scales = np.linalg.eigvalsh(param_dict['Sigma'])
        c = df / (df - 2)
        if p == 2:
            return c*np.sum(sq_scales)
        else:  # p == 4
            return c**2*(2*(df-1)/(df-4)*np.sum(sq_scales**2) + np.sum(sq_scales)**2)

    return VariationalFamily(sample, entropy, logdensity, mean_and_cov,
                             pth_moment, ms_pattern.flat_length(True))


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


def _vectorize_if_needed(f, a, axis=-1):
    if a.ndim > 1:
        return np.apply_along_axis(f, axis, a)
    else:
        return f(a)


def _ensure_2d(a):
    while a.ndim < 2:
        a = a[:,np.newaxis]
    return a


def make_stan_log_density(fitobj):
    @primitive
    def log_density(x):
        return _vectorize_if_needed(fitobj.log_prob, x)
    def log_density_vjp(ans, x):
        return lambda g: _ensure_2d(g) * _vectorize_if_needed(fitobj.grad_log_prob, x)
    defvjp(log_density, log_density_vjp)
    return log_density


def learning_rate_schedule(n_iters, learning_rate, learning_rate_end):
    if learning_rate <= 0:
        raise ValueError('learning rate must be positive')
    if learning_rate_end is not None:
        if learning_rate <= learning_rate_end:
            raise ValueError('initial learning rate must be greater than final learning rate')
        # constant learning rate for first quarter, then decay like a/(b + i)
        # for middle half, then constant for last quarter
        b = n_iters*learning_rate_end/(2*(learning_rate - learning_rate_end))
        a = learning_rate*b
        start_decrease_at = n_iters//4
        end_decrease_at = 3*n_iters//4
    for i in range(n_iters):
        if learning_rate_end is None or i < start_decrease_at:
            yield learning_rate
        elif i < end_decrease_at:
            yield a / (b + i - start_decrease_at + 1)
        else:
            yield learning_rate_end


def adagrad_optimize(n_iters, objective_and_grad, init_param,
                     has_log_norm=False, window=10,learning_rate=.01,
                     epsilon=.1, learning_rate_end=None):
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    with tqdm.trange(n_iters) as progress:
        try:
            schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
            for i, curr_learning_rate in zip(progress, schedule):
                prev_variational_param = variational_param
                if has_log_norm:
                    obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                else:
                    obj_val, obj_grad = objective_and_grad(variational_param)
                    log_norm = 0
                value_history.append(obj_val)
                local_grad_history.append(obj_grad)
                local_log_norm_history.append(log_norm)
                log_norm_history.append(log_norm)
                if len(local_grad_history) > window:
                    local_grad_history.pop(0)
                    local_log_norm_history.pop(0)
                grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                scaled_grads = grad_scale[:,np.newaxis]*np.array(local_grad_history)
                accum_sum = np.sum(scaled_grads**2, axis=0)
                variational_param = variational_param - curr_learning_rate*obj_grad/np.sqrt(epsilon + accum_sum)
                if i >= 3*n_iters // 4:
                    variational_param_history.append(variational_param.copy())
                if i % 10 == 0:
                    avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                    progress.set_description(
                        'Average Loss = {:,.5g}'.format(avg_loss))
        except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
            # do not print log on the same line
            progress.close()
        finally:
            progress.close()
    variational_param_history = np.array(variational_param_history)
    smoothed_opt_param = np.mean(variational_param_history, axis=0)
    return (smoothed_opt_param, variational_param_history,
            np.array(value_history), np.array(log_norm_history))
