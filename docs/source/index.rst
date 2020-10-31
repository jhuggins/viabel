.. VIABEL documentation master file, created by
   sphinx-quickstart on Sat Oct 31 13:12:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VIABEL
======

VIABEL (**V**\ ariational **I**\ nference and **A**\ pproximation **B**\ ounds
that are **E**\ fficient and **L**\ ightweight) is a library (still in early
development) that provides two types of functionality:

#. A lightweight, flexible set of methods for variational inference that is agnostic to how the model is constructed. All that is required is a log density and its gradient.
#. Methods for computing bounds on the errors of the mean, standard deviation, and variance estimates produced by a continuous approximation to an (unnormalized) distribution. A canonical application is a variational approximation to a Bayesian posterior distribution.

.. toctree::
   :maxdepth: 1

   installation
   quickstart

.. Indices and tables
   ==================
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
