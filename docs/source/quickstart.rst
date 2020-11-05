===========
Quick Start
===========


Variational inference
---------------------

VIABEL currently supports both standard KL-based variational inference (KLVI)
and chi-squared variational inference (CHIVI). Models are provided as
Autograd-compatible log densities or can be constructed from PyStan fit objects.

As a simple example, we consider Neal's funnel distribution in 2 dimensions so that we can visualize the results.
>>> import autograd.numpy as np
>>> import autograd.scipy.stats.norm as norm
>>> D = 2  # number of dimensions
>>> log_sigma_stdev = 1.35
>>> def log_density(x):
>>>    mu, log_sigma = x[:, 0], x[:, 1]
>>>    sigma_density = norm.logpdf(log_sigma, 0, log_sigma_stdev)
>>>    mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
>>>    return sigma_density + mu_density

VIABEL's `bbvi` function provides reasonable defaults: the objective is the ELBO
(i.e., the including Kullback-Leibler divergence), a mean-field Gaussian
approximation family, and windowed version of adagrad for stochastic optimization:

>>> from viabel import bbvi
>>> results = bbvi(D, n_iters=5000, log_density=log_density)

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
