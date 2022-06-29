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

[Robust, Automated, and Accurate Black-box Variational Inference](https://arxiv.org/abs/2203.15945).
Manushi Welandawe,
Michael Riis Andersen,
Aki Vehtari,
Jonathan H. Huggins.
arXiv preprint arXiv:2203.15945 (2022).

The equivalent BibTeX entry is:
```
@article{Welandawe:2022:BBVI,
  author={Welandawe, Manushi and Andersen, Michael Riis and Vehtari, Aki and Huggins, Jonathan H},
  title={Robust, Automated, and Accurate Black-box Variational Inference},
  journal={arXiv preprint arXiv:2203.15945},
  year={2022}
}
```