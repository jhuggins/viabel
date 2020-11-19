import numpy as np

from viabel.approximations import MFGaussian
from viabel.diagnostics import all_diagnostics
from viabel.models import Model, StanModel
from viabel.objectives import ExclusiveKL
from viabel.optimization import adagrad_optimize
from viabel._psis import psislw

all = [
    'bbvi',
    'vi_diagnostics',
]


def bbvi(dimension, *, n_iters=10000, num_mc_samples=10, log_density=None, approx=None, objective=None, fit=None,  **kwargs):
    """Fit a model using black-box variational inference.

    Currently the objective is optimized using ``viabel.optimization.adagrad_optimize``.

    Parameters
    ----------
    dimension : `int`
        Dimension of the model parameter.
    n_iters : `int`
        Number of iterations of the optimization.
    num_mc_samples : `int`
        Number of Monte Carlo samples to use for estimating the gradient of
        the objective.
    log_density : `function`
        (Unnormalized) log density of the model. Must support automatic
        differentiation with ``autograd``. Either ``log_density`` or ``fit``
        must be provided.
    approx : `ApproximationFamily` object
        The approximation family. The default is to use ``viabel.approximations.MFGaussian``.
    objective : `VariationalObjective` class
        The default is to use ``viabel.objectives.ExclusiveKL``.
    fit : `StanFit4model` object
        If provided, a ``StanModel`` will be used. Both ``fit`` and
        ``log_density`` cannot be given.
    **kwargs
        Keyword arguments to pass to ``adagrad_optimize``.

    Returns
    -------
    results : `dict`
        Contains the following entries: `var_param`, `var_param_history`,
        `objective`
    """
    if log_density is None:
        if fit is None:
            raise ValueError('either log_density or fit must be specified')
        if objective is not None:
            raise ValueError('objective can only be specified if log_density is too')
        model = StanModel(fit)
    elif fit is None:
        model = Model(log_density)
    else:
        raise ValueError('log_density and fit cannot both be specified')

    if approx is None:
        if objective is not None:
            raise ValueError('objective can only be specified if approx is too')
        approx = MFGaussian(dimension)
    if objective is None:
        objective = ExclusiveKL(approx, log_density, num_mc_samples)
    init_param = np.zeros(approx.var_param_dim)
    var_param, var_param_history, _, _ = adagrad_optimize(n_iters, objective, init_param, **kwargs)
    results = dict(var_param=var_param,
                   var_param_history=var_param_history,
                   objective=objective)
    return results


def vi_diagnostics(var_param, *, objective=None, model=None, approx=None, n_samples=100000):
    """Check variational inference diagnostics.

    Check Pareto k and 2-divergence diagnostics. Return additional diagnostics
    with mean, standard deviation, and covariance error bounds.

    Parameters
    ----------
    var_param : `numpy.ndarray`, shape (var_param_dim,)
        The variational parameter.
    objective : `function`
    model : `Model` object
    approx : `ApproximationFamily` object
    n_samples : `int`
        The number of samples to use for the diagnostics.

    Returns
    -------
    diagnostics : `dict`
        Also includes samples and smoothed log weights.

    See Also
    --------
    diagostics.all_diagnostics : Compute all diagnostics.
    """
    if objective is None:
        if model is None or approx is None:
            raise ValueError('either objective or both model and approx must be specified')
    elif model is not None or approx is not None:
        raise ValueError('model and/or approx cannot be specified if objective is')
    else:
        model = objective.model
        approx = objective.approx
    if n_samples <= 0:
        raise ValueError('n_samples must be positive')

    return _vi_diagnostics(var_param, model, approx, n_samples)


def _vi_diagnostics(var_param, model, approx, n_samples):
    # first check Pareto k-hat
    samples, smoothed_log_weights, khat = psis_correction(var_param, model, approx, n_samples)
    results = dict(samples=samples,
                   smoothed_log_weights=smoothed_log_weights,
                   khat=khat)
    print('Pareto k is estimated to be khat = {:.2f}'.format(results['khat']))
    if results['khat'] > 0.7:
        print('WARNING: khat > 0.7 means importance sampling is not feasible.')
        print('WARNING: not running further diagnostics')
        return results
    print()
    # if k-hat looks good, check other diagnostics
    if approx.supports_pth_moment(2) and approx.supports_pth_moment(4):
        moment_bound_fn = lambda p: approx.pth_moment(var_param, p)
    else:
        moment_bound_fn = None
    _, q_var = approx.mean_and_cov(var_param)
    results.update(all_diagnostics(smoothed_log_weights,
                                   samples=samples,
                                   moment_bound_fn=moment_bound_fn,
                                   q_var=q_var))
    print('The 2-divergence is estimated to be d2 = {:.2g}'.format(results['d2']))
    if results['d2'] > 4.6: # pragma: no cover
        print('WARNING: d2 > 4.6 means the approximation is very inaccurate')
    elif results['d2'] > 0.1:
        print('WARNING: 0.1 > d2 < 4.6 means the approximation is somewhat '
              'inaccurate. Use importance sampling to decrease error.')
    else:
        print('\nAll diagnostics pass.')
    return results


def psis_correction(var_param, model, approx, n_samples):
    samples, log_weights = samples_and_log_weights(var_param, model, approx, n_samples)
    smoothed_log_weights, khat = psislw(log_weights, overwrite_lw=True)
    return samples.T, smoothed_log_weights, khat


def samples_and_log_weights(var_param, model, approx, n_samples):
    samples = approx.sample(var_param, n_samples)
    log_weights = model(samples) - approx.log_density(var_param, samples)
    return samples, log_weights
