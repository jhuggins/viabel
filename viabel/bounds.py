from warnings import warn

import numpy as np

__all__ = [
    'all_bounds',
    'error_bounds',
    'wasserstein_bounds',
    'divergence_bound'
]


def all_bounds(log_weights, samples=None, moment_bound_fn=None,
               q_var=None, p_var=None, log_norm_bound=None):
    """Compute all error and distance bounds.

    Compute error and distance bounds between distribution `p` and `q` using
    samples from `q`. The distributions need not be normalized.

    Parameters
    ----------
    log_weights : array-like of integers, shape=(n_samples,)
        log weights `log p(x_i) - log q(x_i)`, where `x_i` is sampled from `q`
        and `p` may be an unnormalized distribution

    samples : array-like matrix, shape=(n_samples, n_dimensions)
        samples `x_i` associated with log weights

    moment_bound_fn : function
        `moment_bound_fn(p)` should return a bound on `min_y E[(x_i - y)^p]`.
        It must be provided if `samples` is `None` and it must support `p = 2`
        and `p = 4`.

    q_var : float or array-like matrix
        (Bound on) the (co)variance of `q`.

    p_var : float or array-like matrix
        (Bound on) the (co)variance of `p`.

    log_norm_bound : float
        Bound on the overall log normalization constant (the log marginal
        likelihood when `p` is the unnormalized log posterior)


    Returns
    -------
    results : dict
        contains the following bounds: `mean_error`, `var_error`, `std_error`,
        `d2`, `W1`, `W2`."""
    d2, log_norm_bound = divergence_bound(log_weights,
                                          log_norm_bound=log_norm_bound,
                                          return_log_norm_bound=True)
    results = wasserstein_bounds(d2, samples, moment_bound_fn)

    if q_var is None and samples is not None:
        q_var = np.cov(samples.T)

    results.update(error_bounds(q_var=q_var, p_var=p_var, **results))
    results['d2'] = d2
    results['log_norm_bound'] = log_norm_bound
    return results


def _compute_norm_if_needed(var):
    if np.asarray(var).ndim == 2:
        return np.linalg.norm(var, ord=2)
    return var


def error_bounds(W1=np.inf, W2=np.inf, q_var=np.inf, p_var=np.inf):
    """Compute error bounds.

    Compute bounds on differences in the means, standard deviations, and
    covariances of `p` and `q` using (bounds on) the 1- and 2-Wasserstein
    distance.

    Parameters
    ----------
    W1 : float
        (Bound on) the 1-Wasserstein distance between `p` and `q`.

    W2 : float
        (Bound on) the 2-Wasserstein distance between `p` and `q`.

    q_var : float or array-like matrix
        (Bound on) the (co)variance of `q`.

    p_var : float or array-like matrix
        (Bound on) the (co)variance of `p`.

    Returns
    -------
    results : dict
        contains the following bounds: `mean_error`, `var_error`, `std_error`."""
    results = dict()
    results['mean_error'] = mean_bound(min(W1, W2))
    results['std_error'] = std_bound(W2)
    results['cov_error'] = var_bound(W2, _compute_norm_if_needed(q_var),
                                     _compute_norm_if_needed(p_var))
    return results


def wasserstein_bounds(d2, samples=None, moment_bound_fn=None):
    """Compute all bounds.

    Compute 1- and 2-Wasserstein distance bounds between distribution `p` and
    `q` using a bound on the 2-divergence and moment bounds.

    Parameters
    ----------
    d2 : float
        (Bound on) the 2-divergence between `p` and `q`.

    samples : array-like matrix, shape=(n_samples, n_dimensions)
        samples from `q`.

    moment_bound_fn : array-like matrix, shape=(n_variant_types, n_signatures)
        `moment_bound_fn(a)` should return a bound on `min_y E[(x_i - y)^a]`.
        It must be provided if `samples` is `None`. Must support `a = 2`
        and `a = 4`.

    Returns
    -------
    results : dict
        contains the following bounds: `W1`, `W2`."""
    results = dict()
    if moment_bound_fn is None:
        if samples is None:
            raise ValueError('must provides samples if moment_bound_fn not given')
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples[:,np.newaxis]
        sample_mean = np.mean(samples, axis=0, keepdims=True)
        centered_samples = samples - sample_mean
        moment_bound_fn = lambda p: np.mean(np.sum(centered_samples**p, axis=1))
    for p in [1, 2]:
        Cp = moment_bound_fn(2*p)
        results['W{}'.format(p)] = 2 * Cp**(.5/p) * np.expm1(d2)**(.5/p)
    return results


def divergence_bound(log_weights, alpha=2., log_norm_bound=None,
                     return_log_norm_bound=False):
    """Compute a bound on the alpha-divergence.

    Compute error and distance bounds between distribution `p` and `q` using
    samples from `q`.

    Parameters
    ----------
    log_weights : array-like of integers, shape=(n_samples,)
        log weights `log p(x_i) - log q(x_i)`, where `x_i` is sampled from `q`
        and `p` may be an unnormalized distribution.

    alpha : float
        order of the Renyi divergence. Must be greater than 1

    log_norm_bound : float
        Bound on the log normalization constant for `p` (the log marginal
        likelihood when `p` is the unnormalized log posterior).

    Returns
    -------
    dalpha : float
        Bound on the alpha-divergence."""
    if alpha <= 1:
        raise ValueError('alpha must be greater than 1')
    log_weights = np.asarray(log_weights)
    log_rescale = np.max(log_weights)
    rescaled_weights = np.exp(log_weights - log_rescale)**alpha
    mean_rescaled_weight = mean_and_check_mc_error(rescaled_weights,
                                                   quantity_name='CUBO')
    cubo = np.log(mean_rescaled_weight)/alpha + log_rescale
    if log_norm_bound is None:
        log_norm_bound = mean_and_check_mc_error(log_weights,
                                                 quantity_name='ELBO')
    dalpha = alpha / (alpha - 1) * (cubo - log_norm_bound)
    if return_log_norm_bound:
        return dalpha, log_norm_bound
    return dalpha


def mean_and_check_mc_error(a, atol=0.01, rtol=0.0, quantity_name=None):
    m = np.mean(a)
    s = np.std(a)/np.sqrt(a.size)
    if s > rtol*np.abs(m) + atol:
        msg = 'significant Monte Carlo error'
        if quantity_name is not None:
            msg += ' when computing ' + quantity_name
        msg += ' (mean = {}, standard deviation = {})'.format(m, s)
        warn(msg)
    return m


_var_bound_const_1 = 2*np.sqrt(2)
_var_bound_const_2 = 1 + 3*np.sqrt(2)


def mean_bound(Wp):
    return Wp


def std_bound(W2):
    return W2


def var_bound(W2, var1, var2=None):
    if var2 is not None:
        min_var = np.min([var1, var2], axis=0)
    else:
        min_var = var1
    min_std = np.sqrt(min_var)
    return 2 * (min_std * W2 + W2**2)
