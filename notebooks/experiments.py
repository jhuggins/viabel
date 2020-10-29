import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from viabel import all_bounds
from viabel.objectives import black_box_klvi, black_box_chivi
from viabel.optimization import adagrad_optimize
from utils import Timer
from psis import psislw


## Display bounds information ##

def print_bounds(results):
    print('Bounds on...')
    print('  2-Wasserstein   {:.3g}'.format(results['W2']))
    print('  2-divergence    {:.3g}'.format(results['d2']))
    print('  mean error      {:.3g}'.format(results['mean_error']))
    print('  stdev error     {:.3g}'.format(results['std_error']))
    print('  sqrt cov error  {:.3g}'.format(np.sqrt(results['cov_error'])))
    print('  cov error       {:.3g}'.format(results['cov_error']))


## Check approximation accuracy ##

def check_accuracy(true_mean, true_cov, approx_mean, approx_cov, verbose=False,
                   method=None):
    true_std = np.sqrt(np.diag(true_cov))
    approx_std = np.sqrt(np.diag(approx_cov))
    results = dict(mean_error=np.linalg.norm(true_mean - approx_mean),
                   cov_error_2=np.linalg.norm(true_cov - approx_cov, ord=2),
                   cov_norm_2=np.linalg.norm(true_cov, ord=2),
                   cov_error_nuc=np.linalg.norm(true_cov - approx_cov, ord='nuc'),
                   cov_norm_nuc=np.linalg.norm(true_cov, ord='nuc'),
                   std_error=np.linalg.norm(true_std - approx_std),
                   rel_std_error=np.linalg.norm(approx_std/true_std - 1),
                  )
    if method is not None:
        results['method'] = method
    if verbose:
        print('mean   =', approx_mean)
        print('stdevs =', approx_std)
        print()
        print('mean error             = {:.3g}'.format(results['mean_error']))
        print('stdev error            = {:.3g}'.format(results['std_error']))
        print('||cov error||_2^{{1/2}}  = {:.3g}'.format(np.sqrt(results['cov_error_2'])))
        print('||true cov||_2^{{1/2}}   = {:.3g}'.format(np.sqrt(results['cov_norm_2'])))
    return results


def check_approx_accuracy(var_family, var_param, true_mean, true_cov,
                          verbose=False, name=None):
    return check_accuracy(true_mean, true_cov,
                          *var_family.mean_and_cov(var_param),
                          verbose, name)


## Convenience functions and PSIS ##

def get_samples_and_log_weights(logdensity, var_family, var_param, n_samples):
    samples = var_family.sample(var_param, n_samples)
    log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
    return samples, log_weights


def psis_correction(logdensity, var_family, var_param, n_samples):
    samples, log_weights = get_samples_and_log_weights(logdensity, var_family,
                                                       var_param, n_samples)
    smoothed_log_weights, khat = psislw(log_weights)
    return samples.T, smoothed_log_weights, khat


def improve_with_psis(logdensity, var_family, var_param, n_samples,
                      true_mean, true_cov, transform=None, verbose=False):
    samples, slw, khat = psis_correction(logdensity, var_family,
                                         var_param, n_samples)
    if verbose:
        print('khat = {:.3g}'.format(khat))
        print()
    if transform is not None:
        samples = transform(samples)
    slw -= np.max(slw)
    wts = np.exp(slw)
    wts /= np.sum(wts)
    approx_mean = np.sum(wts[np.newaxis,:]*samples, axis=1)
    approx_cov = np.cov(samples, aweights=wts, ddof=0)
    res = check_accuracy(true_mean, true_cov, approx_mean, approx_cov, verbose)
    res['khat'] = khat
    return res, approx_mean, approx_cov


## Plotting ##

def plot_approx_and_exact_contours(logdensity, var_family, var_param,
                                   xlim=[-10,10], ylim=[-3, 3],
                                   cmap2='Reds', savepath=None):
    xlist = np.linspace(*xlim, 100)
    ylist = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(xlist, ylist)
    XY = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    zs = np.exp(logdensity(XY))
    Z = zs.reshape(X.shape)
    zsapprox = np.exp(var_family.logdensity(XY, var_param))
    Zapprox = zsapprox.reshape(X.shape)
    plt.contour(X, Y, Z, cmap='Greys', linestyles='solid')
    plt.contour(X, Y, Zapprox, cmap=cmap2, linestyles='solid')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def plot_history(history, B=None, ylabel=None):
    if B is None:
        B = min(500, history.size//10)
    window = np.ones(B)/B
    smoothed_history = np.convolve(history, window, 'valid')
    plt.plot(smoothed_history)
    yscale = 'log' if np.all(smoothed_history > 0) else 'linear'
    plt.yscale(yscale)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xlabel('iteration')
    plt.show()


def plot_dist_to_opt_param(var_param_history, opt_param):
    plt.plot(np.linalg.norm(var_param_history - opt_param[np.newaxis,:], axis=1))
    plt.title('iteration vs distance to optimal parameter')
    plt.xlabel('iteration')
    plt.ylabel('distance')
    sns.despine()
    plt.show()


## Run experiment with both KLVI and CHIVI ##

def _optimize_and_check_results(logdensity, var_family, objective_and_grad,
                                init_var_param, true_mean, true_cov,
                                plot_contours, ylabel, contour_kws=dict(),
                                elbo=None, n_iters=5000,
                                bound_w2=True, verbose=False, use_psis=True,
                                n_psis_samples=1000000, **kwargs):
    opt_param, var_param_history, value_history, _ = \
        adagrad_optimize(n_iters, objective_and_grad, init_var_param, **kwargs)
    plot_dist_to_opt_param(var_param_history, opt_param)
    accuracy_results = check_approx_accuracy(var_family, opt_param,
                                             true_mean, true_cov, verbose);
    other_results = dict(opt_param=opt_param,
                         var_param_history=var_param_history,
                         value_history=value_history)
    if bound_w2 not in [False, None]:
        if bound_w2 is True:
            n_samples = 1000000
        else:
            n_samples = bound_w2
        print()
        with Timer('Computing CUBO and ELBO with {} samples'.format(n_samples)):
            _, log_weights = get_samples_and_log_weights(
                logdensity, var_family, opt_param, n_samples)
            var_dist_cov = var_family.mean_and_cov(opt_param)[1]
            moment_bound_fn = lambda p: var_family.pth_moment(p, opt_param)
            other_results.update(all_bounds(log_weights,
                                            q_var=var_dist_cov,
                                            moment_bound_fn=moment_bound_fn,
                                            log_norm_bound=elbo))
        if verbose:
            print()
            print_bounds(other_results)
    if plot_contours:
        plot_approx_and_exact_contours(logdensity, var_family, opt_param,
                                       **contour_kws)
    if use_psis:
        print()
        print('Results with PSIS correction')
        print('----------------------------')
        other_results['psis_results'], _, _ = \
            improve_with_psis(logdensity, var_family, opt_param, n_psis_samples,
                              true_mean, true_cov, verbose=verbose)
    return accuracy_results, other_results


def run_experiment(logdensity, var_family, init_param, true_mean, true_cov,
                   kl_n_samples=100, chivi_n_samples=500,
                   alpha=2, **kwargs):
    klvi = black_box_klvi(var_family, logdensity, kl_n_samples)
    chivi = black_box_chivi(alpha, var_family, logdensity, chivi_n_samples)
    dim = true_mean.size
    plot_contours = dim == 2
    if plot_contours:
        plot_approx_and_exact_contours(logdensity, var_family, init_param,
                                       **kwargs.get('contour_kws', dict()))

    print('|--------------|')
    print('|     KLVI     |')
    print('|--------------|', flush=True)
    kl_results, other_kl_results = _optimize_and_check_results(
        logdensity, var_family, klvi, init_param,
        true_mean, true_cov, plot_contours, '-ELBO', **kwargs)
    kl_results['method'] = 'KLVI'
    print()
    print('|---------------|')
    print('|     CHIVI     |')
    print('|---------------|', flush=True)
    elbo = other_kl_results['log_norm_bound']
    chivi_results, other_chivi_results = _optimize_and_check_results(
        logdensity, var_family, chivi, init_param, true_mean, true_cov,
        plot_contours, 'CUBO', elbo=elbo, **kwargs)
    chivi_results['method'] = 'CHIVI'
    return klvi, chivi, kl_results, chivi_results, other_kl_results, other_chivi_results
