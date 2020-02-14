#  viabel: Variational Inference Approximation Bounds that are Efficient and Lightweight

## Description

This package computes bounds errors of the mean, standard deviation, and variances
estimates produced by a continuous approximation to a (unnormalized) distribution.
A canonical application is a variational approximation to a Bayesian posterior
distribution.
In particular, using samples from the approximation *Q* and evaluations of the
(maybe unnormalized) log densities of *Q* and (target distribution) *P*,
the package provides functionality to compute bounds on:

* the &alpha;-divergence between *P* and *Q*
* the *p*-Wasserstein distance between *P* and *Q*
* the differences between the means, standard deviations, and variances of *P* and *Q*

There is also an (optional) variational Bayes functionality (`viabel.vb`), which
supports both standard KL-based variational inference (KLVI) and chi-squared
variational inference (CHIVI).
Models are provided as `autograd`-compatible log densities or can be constructed
from `pystan` fit objects.
The variational objective is optimized using a windowed version of adagrad
and unbiased reparameterization gradients.
By default there is support for mean-field Gaussian, mean-field Student's t,
and full-rank Student's t variational families.

If you use this package, please cite:

[Practical posterior error bounds from variational objectives](https://arxiv.org/abs/1910.04102).
Jonathan H. Huggins,
Miko&#0322;aj Kasprzak,
Trevor Campbell,
Tamara Broderick.
In *Proc. of the 23rd International Conference on Artificial Intelligence and
Statistics* (AISTATS), Palermo, Italy. PMLR: Volume 108, 2020.

## Compilation and testing

After cloning the repository, testing and installation is easy.
If you just want to compute bounds, you can install using the command
```bash
pip install .
```
The only dependency is `numpy`. If you want to use the basic variational Bayes
functionality, use the command
```bash
pip install .[vb]
```
This will install some additional dependencies.
If in addition to the above, you want to run all of the [example notebooks](notebooks),
use the command
```bash
pip install .[examples]
```
This will install even more dependencies.

To test the package:
```bash
nosetests tests/
```
Currently there is only coverage for `viabel.bounds`.

## Usage Examples

Basic usage examples of the bounds are provided in
[normal-mixture-example.ipynb](notebooks/normal-mixture-example.ipynb).

A more involved example that demonstrates how to use the variational Bayes functionality
and then compute bounds is provided in [robust-regression-model-example.ipynb](notebooks/robust-regression-model-example.ipynb).
