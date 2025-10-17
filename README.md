#  VIABEL: *V*ariational *I*nference and *A*pproximation *B*ounds that are *E*fficient and *L*ightweight
[![Build Status](https://travis-ci.org/jhuggins/viabel.svg?branch=master)](https://travis-ci.org/jhuggins/viabel) [![Code Coverage](https://codecov.io/gh/jhuggins/viabel/branch/master/graph/badge.svg)](https://codecov.io/gh/jhuggins/viabel) [![Documentation Status](https://readthedocs.org/projects/viabel/badge/?version=latest)](https://viabel.readthedocs.io/en/latest/?badge=latest)


VIABEL is a library (still in early development) that provides two types of
functionality:

1. A lightweight, flexible set of methods for variational inference that is
agnostic to how the model is constructed. All that is required is a
log density and its gradient.
2. Methods for computing bounds on the errors of the mean, standard deviation,
and variance estimates produced by a continuous approximation to an
(unnormalized) distribution. A canonical application is a variational
approximation to a Bayesian posterior distribution.


## Documentation

For examples and API documentation, see
[readthedocs](https://viabel.readthedocs.io).

## Installation

You can install the latest stable version using `pip install viabel`.
Alternatively, you can clone the repository and use the master branch to
get the most up-to-date version.

## Citing VIABEL

If you use this package for diagnostics, please cite:

[Validated Variational Inference via Practical Posterior Error Bounds](https://arxiv.org/abs/1910.04102).
Jonathan H. Huggins,
Miko&#0322;aj Kasprzak,
Trevor Campbell,
Tamara Broderick.
In *Proc. of the 23rd International Conference on Artificial Intelligence and
Statistics* (AISTATS), Palermo, Italy. PMLR: Volume 108, 2020.

The equivalent BibTeX entry is:
```
@inproceedings{Huggins:2020:VI,
  author = {Huggins, Jonathan H and Kasprzak, Miko{\l}aj and Campbell, Trevor and Broderick, Tamara},
  title = {{Validated Variational Inference via Practical Posterior Error Bounds}},
  booktitle = {Proc. of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year = {2020}
}
```

If you use this package for variational inference, please cite:

[A Framework for Improving the Reliability of Black-box Variational Inference](https://jmlr.org/papers/v25/22-0327.html).
Manushi Welandawe,
Michael Riis Andersen,
Aki Vehtari,
Jonathan H. Huggins (2024).
Journal of Machine Learning Research, 25(219):1−71.

The equivalent BibTeX entry is:
```
@article{Welandawe:2024:BBVI,
  author  = {Manushi Welandawe and Michael Riis Andersen and Aki Vehtari and Jonathan H. Huggins},
  title   = {A Framework for Improving the Reliability of Black-box Variational Inference},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {219},
  pages   = {1--71},
  url     = {http://jmlr.org/papers/v25/22-0327.html}
}
```
