import warnings

import autograd.numpy as np
from scipy.fftpack import next_fast_len


def autocov(samples, axis=-1):
    """Compute autocovariance estimates for every lag for the input array.
    Parameters
    ----------
    samples : `numpy.ndarray(n_chains, n_iters)`
        An array containing samples
    Returns
    -------
    acov: `numpy.ndarray`
        Autocovariance of samples that has same size as the input array
    """
    axis = axis if axis > 0 else len(samples.shape) + axis
    n = samples.shape[axis]
    m = next_fast_len(2 * n)

    samples = samples - samples.mean(axis, keepdims=True)

    # added to silence tuple warning for a submodule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ifft_samp = np.fft.rfft(samples, n=m, axis=axis)
        ifft_samp *= np.conjugate(ifft_samp)

        shape = tuple(
            slice(None) if dim_len != axis else slice(0, n)
            for dim_len, _ in enumerate(samples.shape)
        )
        cov = np.fft.irfft(ifft_samp, n=m, axis=axis)[shape]
        cov /= n
        return cov


def ess(samples):
    """
    Computing Effective Sample Size

    Parameters
    ----------
     samples : `numpy.ndarray(n_chains, n_iters)`
        An array containing samples

    Returns
    -------
    ess : `real`
       effective sample size of the given sample

    """
    n_chain, n_draw = samples.shape
    if (n_chain > 1):
        ValueError("Number of chains must be 1")
    acov = autocov(samples, axis=1)
    # chain_mean = samples.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    # if n_chain > 1:
    #   var_plus += _numba_var(svar, np.var, chain_mean, axis=None, ddof=1)

    rho_hat_t = np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    t = 1
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    # improve estimation
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even
    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess = n_chain * n_draw
    tau_hat = -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + np.sum(rho_hat_t[max_t + 1: max_t + 2])
    tau_hat = max(tau_hat, 1 / np.log10(ess))
    ess = ess / tau_hat
    if np.isnan(rho_hat_t).any():
        ess = np.nan
    return ess


def MCSE(sample):
    """
    Compute the Monte Carlo standard error (MCSE)

    Parameters
    ----------
    samples : `numpy.ndarray(n_iters, 2*dim)`
        An array containing variational samples

    Returns
    -------
    mcse : `numpy.ndarray(2*dim)`
        MCSE for each variational parameter

    """
    n_iters, d = sample.shape
    sd_dev = np.sqrt(np.var(sample, ddof=1, axis=0))
    eff_samp = [ess(sample[:, i].reshape(1, n_iters)) for i in range(d)]
    mcse = sd_dev / np.sqrt(eff_samp)
    return eff_samp, mcse


def compute_R_hat(chains, warmup=0, jitter=1e-8):
    """
    Computing R hat values using split R hat approach

    Parameters
    ----------
    chains : `numpy_ndarray(n_iters, dimensions)`
        Sample of parameter estimates that has one chain
    warmup : `int`, optional
        Number of iterations needed for warmup. The default is 0.
    jitter : `float`, optional
        Smoothing term that avoids division by zero. The default is 1e-8.

    Returns
    -------
    R_hat : `numpy_ndarray(dimenstions,)`
        Computed R hat values for each parameter

    """
    n_chains = 1
    chains = chains[warmup:, :]
    n_iters, d = chains.shape
    if n_iters % 2 == 1:
        n_iters = int(n_iters - 1)
        chains = chains[:n_iters, :]
    n_iters = n_iters // 2
    n_chains2 = n_chains * 2
    psi = np.reshape(chains, (n_chains2, n_iters, d))
    psi_dot_j = np.mean(psi, axis=1)
    psi_dot_dot = np.mean(psi_dot_j, axis=0)
    s_j_2 = np.sum((psi - np.expand_dims(psi_dot_j, axis=1)) ** 2, axis=1) / (n_iters - 1)
    B = n_iters * np.sum((psi_dot_j - psi_dot_dot) ** 2, axis=0) / (n_chains2 - 1)
    W = np.nanmean(s_j_2, axis=0)
    W = W + jitter
    var_hat = (n_iters - 1) / n_iters + (B / (n_iters * W))
    R_hat = np.sqrt(var_hat)
    return R_hat


def R_hat_convergence_check(samples, windows, Rhat_threshold=1.1):
    """
    Convergence check of samples using R_hat

    Parameters
    ----------
    samples : `list`
        Samples to check the convergence
    windows : `numpy_array`
        Windows that should be used to do the check

    Returns
    -------
    success: `bool`
        Whether the convergence criterion is met
    best_W: `int`
        Best window size
    """
    R_hat_array = [np.max(compute_R_hat(np.array(samples[-window:]), 0)) for window in windows]
    best_R_hat_ind = np.argmin(R_hat_array)
    success = R_hat_array[best_R_hat_ind] <= Rhat_threshold
    return success, windows[best_R_hat_ind]
