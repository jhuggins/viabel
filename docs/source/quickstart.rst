===========
Quick Start
===========


Variational inference
---------------------

VIABEL currently supports both standard KL-based variational inference (KLVI)
and chi-squared variational inference (CHIVI).
Models are provided as Autograd-compatible log densities or can be constructed
from PyStan fit objects.
As a simple example, we consider Neal's funnel distribution in 2 dimensions so
that we can visualize the results.

>>> import autograd.numpy as np
>>> import autograd.scipy.stats.norm as norm
>>> D = 2  # number of dimensions
>>> log_sigma_stdev = 1.35
>>> def log_density(x):
>>>    mu, log_sigma = x[:, 0], x[:, 1]
>>>    sigma_density = norm.logpdf(log_sigma, 0, log_sigma_stdev)
>>>    mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
>>>    return sigma_density + mu_density

We will use a product of *t*-distributions as the variational family:
>>> from viabel.family import mean_field_t_variational_family
>>> var_family = mean_field_t_variational_family(D, 40)

The variational objective is (standard) exclusive KL-divergence (i.e., the ELBO)
with unbiased reparameterization gradients:

>>> from viabel.objectives import black_box_klvi
>>> # number of Monte Carlo samples for estimating gradients
>>> num_mc_samples = 100
>>> # function that returns an unbiased estimate of the objective and its gradient
>>> vi_objective_and_grad = black_box_klvi(var_family, log_density, num_mc_samples)

The variational objective can be optimized using a windowed version of adagrad:
>>> init_var_param = np.zeros(var_family.var_param_dim)
>>> n_iters = 2500
>>> # var_param is the estimated optimal variational parameter using iterate averaging
>>> var_param, _, _, _ = adagrad_optimize(n_iters, vi_objective_and_grad, init_var_param, learning_rate=.01)

In this case, the resulting variational approximation (red) of the
funnel distribution (black) is not particularly good.

.. image:: funnel.png


Error Bounds
------------

The error bounds are based on samples from the approximation *Q* and evaluations
of the (maybe unnormalized) log densities of *Q* and the target distribution *P*.
In particular, you can compute bounds on:

* the :math:`\alpha`-divergence between *P* and *Q*
* the *p*\ -Wasserstein distance between *P* and *Q*
* the differences between the means, standard deviations, and variances of *P* and *Q*
