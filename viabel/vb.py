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
    'make_stan_log_density'
]

VariationalFamily = namedtuple('VariationalFamily',
                               ['sample', 'entropy', 'kl',
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

    def kl(var_param0, var_param1):
        mean0, log_std0 = unpack_params(var_param0)
        mean1, log_std1 = unpack_params(var_param1)
        mean_diff = mean0 - mean1
        log_std_diff = log_std0 - log_std1
        return .5 * np.sum(  np.exp(2*log_std_diff)
                           + mean_diff**2 / np.exp(2*log_std1)
                           - 2*log_std_diff
                           - 1)

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

    return VariationalFamily(sample, entropy, kl, logdensity,
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

    return VariationalFamily(sample, entropy, None, logdensity,
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

    return VariationalFamily(sample, entropy, None, logdensity, mean_and_cov,
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
