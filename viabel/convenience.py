import numpy as np

from viabel.family import mean_field_gaussian_variational_family
from viabel.models import make_stan_log_density
from viabel.objectives import black_box_klvi
from viabel.optimization import adagrad_optimize


def bbvi(dimension, n_iters=10000, n_samples=10, log_density=None, family=None, objective_and_grad=None, fit=None,  **kwargs):
    """Fit a model using black-box variational inference.

    Currently the objective is optimized using ``viabel.optimization.adagrad_optimize``.

    Parameters
    ----------
    dimension : int
        Dimension of the model parameter.
    n_iters : int
        Number of iterations of the optimization.
    n_samples : int
        Number of Monte Carlo samples to use for estimating the gradient of
        the objective.
    log_density : function
        Log density of the model. It must be provided unless ``fit`` is specified.
    family : object
        Variational family object. The default is to use ``viabel.family.mean_field_gaussian_variational_family``.
    objective : function
        Function for constructing the objective and gradient function. The default is
        to use ``viabel.objectives.black_box_klvi``.
    fit : StanFit4model object
        If provided, ``log_density`` will be constructed using ``viabel.models.make_stan_log_density``.
        Both ``fit`` and ``log_density`` cannot be given.
    **kwargs
        Keyword arguments to pass to ``adagrad_optimize``.

    Returns
    -------
    results : dict
        Dictionary containing the results.
    """
    if log_density is None:
        if fit is None:
            raise ValueError('either log_density or fit must be specified')
        if objective_and_grad is not None:
            raise ValueError('objective_and_grad can only be specified if log_density is too')
        log_density = make_stan_log_density(fit)
    elif fit is not None:
        raise ValueError('log_density and fit cannot both be specified')
    if family is None:
        if objective_and_grad is not None:
            raise ValueError('objective_and_grad can only be specified if family is too')
        family = mean_field_gaussian_variational_family(dimension)
    if objective_and_grad is None:
        objective_and_grad = black_box_klvi(family, log_density, n_samples)
    init_param = np.zeros(family.var_param_dim)
    var_param, var_param_history, _, _ = adagrad_optimize(n_iters, objective_and_grad, init_param, **kwargs)
    results = dict(var_param=var_param,
                   var_param_history=var_param_history,
                   log_density=log_density,
                   family=family,
                   objective_and_grad=objective_and_grad)
    return results
