API
====
.. currentmodule:: viabel

Convenience Methods
-------------------

.. autosummary::
   :toctree:

   bbvi
   vi_diagnostics

Approximation Families
----------------------

.. autosummary::
   :template: myclass.rst
   :toctree:

   ApproximationFamily
   MFGaussian
   MFStudentT
   MultivariateT
   NeuralNet
   NVPFlow

Models
------

.. autosummary::
   :template: myclass.rst
   :toctree:

   Model
   StanModel

Variational Objectives
----------------------

.. autosummary::
   :template: myclass.rst
   :toctree:

   VariationalObjective
   ExclusiveKL
   DISInclusiveKL
   AlphaDivergence

Diagnostics
-----------

.. autosummary::
   :toctree:

   all_diagnostics
   divergence_bound
   error_bounds
   wasserstein_bounds

Optimization
------------

.. autosummary::
   :template: myclass.rst
   :toctree:

   Optimizer
   StochasticGradientOptimizer
   Adam
   AveragedAdam
   RMSProp
   AveragedRMSProp
   Adagrad
   WindowedAdagrad
   FASO
   RAABBVI
