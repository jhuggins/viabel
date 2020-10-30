import autograd.numpy as np
from autograd.extend import primitive


def compute_R_hat(chains, warmup=0.5):
    """
    Compute the split-R-hat for multiple chains,
    all the chains are split into two and the R-hat is computed over them
    before removing the 'warmup' iterates if desired.

    Parameters
    ----------
    chains : multi-dimensional array, shape=(n_chains, n_iters, n_var_params)

    Returns
    -------
    var_hat : var-hat computed in BDA

    R_hat: the split R-hat for multiple chains
    """

    jitter = 1e-8
    n_iters = chains.shape[1]
    n_chains = chains.shape[0]
    if warmup <1:
        warmup = int(warmup*n_iters)

    if warmup > n_iters-2:
        raise ValueError('Warmup should be less than number of iterates ..')

    if (n_iters -warmup) %2 :
        warmup = int(warmup +1)

    chains = chains[:, warmup:, :]

    K = chains.shape[2]
    n_iters = chains.shape[1]
    n_iters = int(n_iters//2)
    psi = np.reshape(chains, (n_chains * 2, n_iters, K))
    n_chains2 = n_chains*2
    psi_dot_j = np.mean(psi, axis=1)
    psi_dot_dot = np.mean(psi_dot_j, axis=0)
    s_j_2 = np.sum((psi - np.expand_dims(psi_dot_j, axis=1)) ** 2, axis=1) / (n_iters - 1)
    B = n_iters * np.sum((psi_dot_j - psi_dot_dot) ** 2, axis=0) / (n_chains2 - 1)
    W = np.nanmean(s_j_2, axis=0)
    W = W + jitter
    var_hat = (n_iters - 1) / n_iters + (B / (n_iters*W))
    R_hat = np.sqrt(var_hat)
    return var_hat, R_hat

def compute_R_hat_window(chains, interval=100, warmup=0.5):
    """
    Compute the split-R-hat for multiple chains over increasingly bigger
    windows. Say you have 2000 iterates and the interval size is 100,
    then this will compute the Rhat for first 1000, 1100, 1200, 1300
    iterates and so on ...

    Parameters
    ----------
    chains : multi-dimensional array, shape=(n_chains, n_iters, n_dimensions)

    interval : size by which window size increases

    start : iterate index at which the computation begins.

    Returns
    -------
    R_hat_array : multi-dimensional of split R-hat statistics for increasingly bigger windows.
    """
    n_chains, n_iters, K= chains.shape
    n_subchains = n_iters //interval
    r_hats_halfway = list()

    if warmup <1:
        warmup = int(warmup*n_iters)

    if warmup > n_iters-1:
        raise ValueError('Warmup should be less than number of iterates ..')

    if (n_iters -warmup) %2 :
        warmup = int(warmup +1)

    for i in range(n_subchains):
        sub_chains = chains[:, :warmup+(i+1)*interval,:]
        n_sub_chains, n_sub_iters, K = sub_chains.shape
        r_hat_current = compute_R_hat(sub_chains, warmup=n_sub_iters//2)[1]
        r_hats_halfway.append(r_hat_current)

    return np.array(r_hats_halfway)


# compute mcmcse for a chain/array
def monte_carlo_se_moving(chains, warmup=0.5, param_idx=0):
    """
    Compute the monte carlo standard error for a variational parameter
    at each iterate using all iterates before that iterate.
    The MCSE is computed using eq (5) of https://arxiv.org/pdf/1903.08008.pdf

    Here, MCSE(\lambda_i)=  sqrt(V(\lambda_i)/Seff)
    where ESS is the effective sample size computed using eq(11).
    MCSE is from 100th to the last iterate using all the chains.

    Parameters
    ----------
    iterate_chains : multi-dimensional array, shape=(n_chains, n_iters, n_var_params)

    warmup : warmup iterates

    param_idx : index of the variational parameter

    Returns
    -------
    mcse_combined_list : array of mcse values for variational parameter with param_idx
    """

    n_chains, N_iters = chains.shape[0], chains.shape[1]

    if warmup <1:
        warmup = int(warmup*N_iters)

    if warmup > N_iters-1:
        raise ValueError('Warmup should be less than number of iterates ..')

    if (N_iters -warmup) %2 :
        warmup = int(warmup +1)

    chains = chains[:, warmup:, param_idx]
    mcse_combined_list = np.zeros(N_iters)
    Neff, _, _, _ = autocorrelation(chains, warmup=0, param_idx=param_idx)

    for i in range(101, N_iters):
        chains_sub = chains[:, :i]
        n_chains, n_iters = chains_sub.shape[0], chains_sub.shape[1]
        chains_flat = np.reshape(chains_sub, (n_chains*i, 1))
        variances_combined = np.var(chains_flat, ddof=1, axis=0)
        Neff , _, _, _ = autocorrelation(chains[:,:i,:], warmup=0, param_idx=param_idx)
        mcse_combined = np.sqrt(variances_combined/Neff)
        mcse_combined_list[i] = mcse_combined
    return  np.array(mcse_combined_list)


def monte_carlo_se(iterate_chains, warmup=0.5):
    """
    compute monte carlo standard error using all chains and all variational parameters at once.
    Parameters
    ----------
    iterate_chains : multi-dimensional array, shape=(n_chains, n_iters, n_var_params)

    Returns
    -------
    mcse_combined : Monte Carlo Standard Error

    """
    n_chains, n_iters, K = iterate_chains.shape[0], iterate_chains.shape[1], iterate_chains.shape[2]

    if warmup <1:
        warmup = int(warmup*n_iters)

    if warmup > n_iters-2:
        raise ValueError('Warmup should be less than number of iterates ..')

    if (n_iters -warmup) %2 :
        warmup = int(warmup +1)
    chains_flat = np.reshape(iterate_chains, (n_chains * n_iters, K))
    variances_combined = np.var(chains_flat, ddof=1, axis=0)
    #mcse_combined = np.sqrt(variances_combined / Neff)

    Neff = np.zeros(K)
    for pmx in range(K):
        chains = iterate_chains[:, warmup:, pmx]
        a, _, _, _ = autocorrelation(iterate_chains, warmup=0, param_idx=pmx)
        Neff[pmx] = a

    mcse_combined = np.sqrt(variances_combined / Neff)
    return mcse_combined


def autocorrelation(iterate_chains, warmup=0.5, param_idx=0, lag_max=100):
    """
    Compute the autocorrelation and ESS for a variational parameter using FFT.
    where ESS is the effective sample size computed using eq(10) and (11) of https://arxiv.org/pdf/1903.08008.pdf
    MCSE is from 100th to the last iterate using all the chains.

    Parameters
    ----------
    iterate_chains : multi-dimensional array, shape=(n_chains, n_iters, n_var_params)

    warmup : warmup iterates

    param_idx : index of the variational parameter

    lag_max: lag value

    Returns
    -------
    neff : Effective sample size

    rho_t: autocorrelation at last lag

    autocov: auto covariance using FFT

    a: array of autocorrelation from lag t=0 to lag t=lag_max
    """
    n_iters = iterate_chains.shape[1]
    n_chains = iterate_chains.shape[0]
    if warmup <1:
        warmup = int(warmup*n_iters)

    if warmup > n_iters-2:
        raise ValueError('Warmup should be less than number of iterates ..')

    if (n_iters -warmup) %2 :
        warmup = int(warmup +1)

    chains = iterate_chains[:, warmup:, param_idx]
    means = np.mean(chains, axis=1)
    variances = np.var(chains, ddof=1, axis=1)
    if n_chains == 1:
        var_between = 0
    else:
        var_between = n_iters * np.var(means, ddof=1)

    var_chains = np.mean(variances, axis=0)
    var_pooled = ((n_iters - 1.) * var_chains + var_between) /n_iters
    n_pad = int(2**np.ceil(1. + np.log2(n_iters)))
    freqs =   np.fft.rfft(chains - np.expand_dims(means, axis=1), n_pad)
    #print(freqs)
    autocov = np.fft.irfft(np.abs(freqs)**2)[:,:n_iters].real
    autocov= autocov / np.arange(n_iters, 0, -1)
    rho_t = 0
    lag = 1
    a = []
    neff_array = []
    for lag in range(lag_max):
        val =  1. - (var_chains - np.mean(autocov[:,lag])) / var_pooled
        a.append(val)
        if val >= 0:
            rho_t = rho_t + val
        else:
            #break
            rho_t =rho_t

    neff = n_iters *n_chains /(1 + 2*rho_t)
    return neff, rho_t, autocov, np.asarray(a)


def compute_khat_iterates(iterate_chains, warmup=0.85, param_idx=0, increasing=True):
    """
    Compute the khat over iterates for a variational parameter after removing warmup.
    Parameters
    ----------
    iterate_chains : multi-dimensional array, shape=(n_chains, n_iters, n_var_params)

    warmup : warmup iterates

    param_idx : index of the variational parameter

    increasing : boolean sort array in increasing order: TRUE or decreasing order:FALSE

    fraction: the fraction of iterates
    Returns
    -------
    maximum of khat over all chains for the variational parameter param_idx

    """
    chains = iterate_chains[:, :, param_idx]
    n_iters = chains.shape[1]
    n_chains = chains.shape[0]

    k_hat_values = np.zeros(n_chains)
    for i in range(n_chains):
        if increasing:
            sorted_chain = np.sort(chains[i,:])
        else:
            sorted_chain = np.sort(-chains[i,:])

        ind_last = int(n_iters * warmup)
        filtered_chain = sorted_chain[ind_last:]
        if increasing:
            filtered_chain = filtered_chain -np.min(filtered_chain)
        else:
            filtered_chain = filtered_chain -np.min(filtered_chain)
        k_post, _ = gpdfit(filtered_chain)
        k_hat_values[i] = k_post

    return np.nanmax(k_hat_values)


# taken from arviz ...
def gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).
    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.
    Parameters
    ----------
    ary : array
        sorted 1D data array
    Returns
    -------
    k : float
        estimated shape parameter
    sigma : float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n ** 0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)  # pylint: disable=no-member
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    # remove negligible weights
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    # normalise weights
    weights /= weights.sum()

    # posterior mean for b
    b_post = np.sum(b_ary * weights)
    # estimate for k
    k_post = np.log1p(-b_post * ary).mean()  # pylint: disable=invalid-unary-operand-type,no-member
    # add prior for k_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    sigma = -k_post / b_post

    return k_post, sigma
