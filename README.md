#  viabel: Variational Inference Approximation Bounds that are Efficient and Lightweight

## Description

This package computes bounds errors of the mean, standard deviation, and variances
estimates produced by a continuous approximation to a (unnormalized) distribution.
A canonical application is a variational approximation to a Bayesian posterior
distribution.
In particular, using samples from the approximation Q and evaluations of the
(maybe unnormalized) log densities of Q and (target distribution) P,
the package provides functionality to compute bounds on:

* the alpha-divergence between P and Q
* the p-Wasserstein distance between P and Q
* the differences between the means, standard deviations, and variances of P and Q

If you use this package, please cite

[Practical posterior error bounds from variational objectives](https://arxiv.org/abs/1910.04102).
Jonathan H. Huggins,
Miko&#0322;aj Kasprzak,
Trevor Campbell,
Tamara Broderick.
*arXiv:1910.04102 [stat.ML]*, 2019.
