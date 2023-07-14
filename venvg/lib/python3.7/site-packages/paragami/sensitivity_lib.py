##########################################################################
# Functions for evaluating the sensitivity of optima to hyperparameters. #
##########################################################################

import autograd
import autograd.numpy as np
from copy import deepcopy
from math import factorial
import scipy as sp
import scipy.sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix
import warnings

from .function_patterns import FlattenFunctionInput

#############################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
# This will soon be moved to vittles.       #
# Do not develop in paragami!               #
#############################################

class HessianSolver:
    """A class to provide a common interface for solving :math:`H^{-1} g`.
    """
    def __init__(self, h, method):
        """
        Parameters
        -------------
        h : `numpy.ndarray` or `scipy.sparse` matrix
            The "Hessian" matrix for sensitivity analysis.
        method : {'factorization', 'cg'}
            How to solve the system.  `factorization` uses a Cholesky decomposition,
            and `cg` uses conjugate gradient.
        """
        self.__valid_methods = [ 'factorization', 'cg' ]
        if method not in self.__valid_methods:
            raise ValueError('method must be one of {}'.format(self.__valid_methods))
        self._method = method
        self.set_h(h)
        self.set_cg_options({})

    def set_h(self, h):
        """Set the Hessian matrix.
        """
        self._h = h
        self._sparse = sp.sparse.issparse(h)
        if self._method == 'factorization':
            if self._sparse:
                self._solve_h = sp.sparse.linalg.factorized(self._h)
            else:
                self._h_chol = sp.linalg.cho_factor(self._h)
        elif self._method == 'cg':
            self._linop = sp.sparse.linalg.aslinearoperator(self._h)
        else:
            raise ValueError('Unknown method {}'.format(self._method))

    def set_cg_options(self, cg_opts):
        """Set the cg options as a dictionary.

        Parameters
        -------------
        cg_opts : `dict`
            A dictionary of keyword options to be passed to
            `scipy.sparse.linalg.cg`.  If ``method`` is not ``cg``, these will be
            ignored.
        """
        self._cg_opts = cg_opts

    def solve(self, v):
        """Solve the linear system :math:`H{-1} v`.

        Parameters
        ------------
        v : `numpy.ndarray`
            A numpy array.

        Returns
        --------
        h_inv_v : `numpy.ndarray`
            The value of :math:`H{-1} v`.
        """
        if self._method == 'factorization':
            if self._sparse:
                return self._solve_h(v)
            else:
                return sp.linalg.cho_solve(self._h_chol, v)
        elif self._method == 'cg':
            cg_result = sp.sparse.linalg.cg(self._linop, v, **self._cg_opts)
            if cg_result[1] != 0:
                warnings.warn('CG exited with error code {}'.format(cg_result[1]))
            return cg_result[0]

        else:
            raise ValueError('Unknown method {}'.format(self._method))



##############
# LRVB class #
##############

class LinearResponseCovariances:
    """
    Calculate linear response covariances of a variational distribution.

    Let :math:`q(\\theta | \\eta)` be a class of probability distribtions on
    :math:`\\theta` where the class is parameterized by the real-valued vector
    :math:`\\eta`.  Suppose that we wish to approximate a distribution
    :math:`q(\\theta | \\eta^*) \\approx p(\\theta)` by solving an optimization
    problem :math:`\\eta^* = \\mathrm{argmin} f(\\eta)`.  For example, :math:`f`
    might be a measure of distance between :math:`q(\\theta | \\eta)` and
    :math:`p(\\theta)`.  This class uses the sensitivity of the optimal
    :math:`\\eta^*` to estimate the covariance
    :math:`\\mathrm{Cov}_p(g(\\theta))`. This covariance estimate is called the
    "linear response covariance".

    In this notation, the arguments to the class mathods are as follows.
    :math:`f` is ``objective_fun``, :math:`\\eta^*` is ``opt_par_value``, and
    the function ``calculate_moments`` evaluates :math:`\\mathbb{E}_{q(\\theta |
    \\eta)}[g(\\theta)]` as a function of :math:`\\eta`.

    Methods
    ------------
    set_base_values:
        Set the base values, :math:`\\eta^*` that optimizes the
        objective function.
    get_hessian_at_opt:
        Return the Hessian of the objective function evaluated at the optimum.
    get_hessian_cholesky_at_opt:
        Return the Cholesky decomposition of the Hessian of the objective
        function evaluated at the optimum.
    get_lr_covariance:
        Return the linear response covariance of a given moment.
    """
    def __init__(
        self,
        objective_fun,
        opt_par_value,
        validate_optimum=False,
        hessian_at_opt=None,
        factorize_hessian=True,
        grad_tol=1e-8):
        """
        Parameters
        --------------
        objective_fun: Callable function
            A callable function whose optimum parameterizes an approximate
            Bayesian posterior.  The function must take as a single
            argument a numeric vector, ``opt_par``.
        opt_par_value:
            The value of ``opt_par`` at which ``objective_fun`` is optimized.
        validate_optimum: Boolean
            When setting the values of ``opt_par``, check
            that ``opt_par`` is, in fact, a critical point of
            ``objective_fun``.
        hessian_at_opt: Numeric matrix (optional)
            The Hessian of ``objective_fun`` at the optimum.  If not specified,
            it is calculated using automatic differentiation.
        factorize_hessian: Boolean
            If ``True``, solve the required linear system using a Cholesky
            factorization.  If ``False``, use the conjugate gradient algorithm
            to avoid forming or inverting the Hessian.
        grad_tol: Float
            The tolerance used to check that the gradient is approximately
            zero at the optimum.
        """

        warnings.warn(
            'This class is being moved to the vittles package.',
            DeprecationWarning)
        self._obj_fun = objective_fun
        self._obj_fun_grad = autograd.grad(self._obj_fun, argnum=0)
        self._obj_fun_hessian = autograd.hessian(self._obj_fun, argnum=0)
        self._obj_fun_hvp = autograd.hessian_vector_product(
            self._obj_fun, argnum=0)

        self._grad_tol = grad_tol

        self.set_base_values(
            opt_par_value, hessian_at_opt,
            factorize_hessian, validate=validate_optimum)

    def set_base_values(self,
                        opt_par_value,
                        hessian_at_opt,
                        factorize_hessian=True,
                        validate=True,
                        grad_tol=None):
        if grad_tol is None:
            grad_tol = self._grad_tol

        # Set the values of the optimal parameters.
        self._opt0 = deepcopy(opt_par_value)

        # Set the values of the Hessian at the optimum.
        if hessian_at_opt is None:
            self._hess0 = self._obj_fun_hessian(self._opt0)
        else:
            self._hess0 = hessian_at_opt

        method = 'factorization' if factorize_hessian else 'cg'
        self.hess_solver = HessianSolver(self._hess0, method)

        if validate:
            # Check that the gradient of the objective is zero at the optimum.
            grad0 = self._obj_fun_grad(self._opt0)
            newton_step = -1 * self.hess_solver.solve(grad0)

            newton_step_norm = np.linalg.norm(newton_step)
            if newton_step_norm > grad_tol:
                err_msg = \
                    'The gradient is not zero at the putatively optimal ' + \
                    'values.  ||newton_step|| = {} > {} = grad_tol'.format(
                        newton_step_norm, grad_tol)
                raise ValueError(err_msg)

    # Methods:
    def get_hessian_at_opt(self):
        return self._hess0

    def get_lr_covariance_from_jacobians(self,
                                         moment_jacobian1,
                                         moment_jacobian2):
        """
        Get the linear response covariance between two vectors of moments.

        Parameters
        ------------
        moment_jacobian1: 2d numeric array.
            The Jacobian matrix of a map from a value of
            ``opt_par`` to a vector of moments of interest.  Following
            standard notation for Jacobian matrices, the rows should
            correspond to moments and the columns to elements of
            a flattened ``opt_par``.
        moment_jacobian2: 2d numeric array.
            Like ``moment_jacobian1`` but for the second vector of moments.

        Returns
        ------------
        Numeric matrix
            If ``moment_jacobian1(opt_par)`` is the Jacobian
            of :math:`\mathbb{E}_q[g_1(\\theta)]` and
            ``moment_jacobian2(opt_par)``
            is the Jacobian of  :math:`\mathbb{E}_q[g_2(\\theta)]` then this
            returns the linear response estimate of
            :math:`\\mathrm{Cov}_p(g_1(\\theta), g_2(\\theta))`.
        """

        if moment_jacobian1.ndim != 2:
            raise ValueError('moment_jacobian1 must be a 2d array.')

        if moment_jacobian2.ndim != 2:
            raise ValueError('moment_jacobian2 must be a 2d array.')

        if moment_jacobian1.shape[1] != len(self._opt0):
            err_msg = ('The number of rows of moment_jacobian1 must match' +
                       'the dimension of the optimization parameter. ' +
                       'Expected {} rows, but got shape = {}').format(
                         len(self._opt0), moment_jacobian1.shape)
            raise ValueError(err_msg)

        if moment_jacobian2.shape[1] != len(self._opt0):
            err_msg = ('The number of rows of moment_jacobian2 must match' +
                       'the dimension of the optimization parameter. ' +
                       'Expected {} rows, but got shape = {}').format(
                         len(self._opt0), moment_jacobian2.shape)
            raise ValueError(err_msg)

        # return moment_jacobian1 @ cho_solve(
        #     self._hess0_chol, moment_jacobian2.T)
        return moment_jacobian1 @ self.hess_solver.solve(moment_jacobian2.T)

    def get_moment_jacobian(self, calculate_moments):
        """
        The Jacobian matrix of a map from ``opt_par`` to a vector of
        moments of interest.

        Parameters
        ------------
        calculate_moments: Callable function
            A function that takes the folded ``opt_par`` as a single argument
            and returns a numeric vector containing posterior moments of
            interest.

        Returns
        ----------
        Numeric matrix
            The Jacobian of the moments.
        """
        calculate_moments_jacobian = autograd.jacobian(calculate_moments)
        return calculate_moments_jacobian(self._opt0)

    def get_lr_covariance(self, calculate_moments):
        """
        Get the linear response covariance of a vector of moments.

        Parameters
        ------------
        calculate_moments: Callable function
            A function that takes the folded ``opt_par`` as a single argument
            and returns a numeric vector containing posterior moments of
            interest.

        Returns
        ------------
        Numeric matrix
            If ``calculate_moments(opt_par)`` returns
            :math:`\\mathbb{E}_q[g(\\theta)]`
            then this returns the linear response estimate of
            :math:`\\mathrm{Cov}_p(g(\\theta))`.
        """

        moment_jacobian = self.get_moment_jacobian(calculate_moments)
        return self.get_lr_covariance_from_jacobians(
            moment_jacobian, moment_jacobian)


class HyperparameterSensitivityLinearApproximation:
    """
    Linearly approximate dependence of an optimum on a hyperparameter.

    Suppose we have an optimization problem in which the objective
    depends on a hyperparameter:

    .. math::

        \hat{\\theta} = \mathrm{argmin}_{\\theta} f(\\theta, \\lambda).

    The optimal parameter, :math:`\hat{\\theta}`, is a function of
    :math:`\\lambda` through the optimization problem.  In general, this
    dependence is complex and nonlinear.  To approximate this dependence,
    this class uses the linear approximation:

    .. math::

        \hat{\\theta}(\\lambda) \\approx \hat{\\theta}(\\lambda_0) +
            \\frac{d\hat{\\theta}}{d\\lambda}|_{\\lambda_0}
                (\\lambda - \\lambda_0).

    In terms of the arguments to this function,
    :math:`\\theta` corresponds to ``opt_par``,
    :math:`\\lambda` corresponds to ``hyper_par``,
    and :math:`f` corresponds to ``objective_fun``.

    Methods
    ------------
    set_base_values:
        Set the base values, :math:`\\lambda_0` and
        :math:`\\theta_0 := \hat\\theta(\\lambda_0)`, at which the linear
        approximation is evaluated.
    get_dopt_dhyper:
        Return the Jacobian matrix
        :math:`\\frac{d\hat{\\theta}}{d\\lambda}|_{\\lambda_0}` in flattened
        space.
    get_hessian_at_opt:
        Return the Hessian of the objective function in the
        flattened space.
    predict_opt_par_from_hyper_par:
        Use the linear approximation to predict
        the value of ``opt_par`` from a value of ``hyper_par``.
    """
    def __init__(
        self,
        objective_fun,
        opt_par_value, hyper_par_value,
        validate_optimum=False,
        hessian_at_opt=None,
        cross_hess_at_opt=None,
        factorize_hessian=True,
        hyper_par_objective_fun=None,
        grad_tol=1e-8):
        """
        Parameters
        --------------
        objective_fun : `callable`
            The objective function taking two positional arguments,
            - ``opt_par``: The parameter to be optimized (`numpy.ndarray` (N,))
            - ``hyper_par``: A hyperparameter (`numpy.ndarray` (N,))
            and returning a real value to be minimized.
        opt_par_value :  `numpy.ndarray` (N,)
            The value of ``opt_par`` at which ``objective_fun`` is
            optimized for the given value of ``hyper_par_value``.
        hyper_par_value : `numpy.ndarray` (M,)
            The value of ``hyper_par`` at which ``opt_par`` optimizes
            ``objective_fun``.
        validate_optimum : `bool`, optional
            When setting the values of ``opt_par`` and ``hyper_par``, check
            that ``opt_par`` is, in fact, a critical point of
            ``objective_fun``.
        hessian_at_opt : `numpy.ndarray` (N,N), optional
            The Hessian of ``objective_fun`` at the optimum.  If not specified,
            it is calculated using automatic differentiation.
        cross_hess_at_opt : `numpy.ndarray`  (N, M)
            Optional.  The second derivative of the objective with respect to
            ``input_val`` then ``hyper_val``.
            If not specified it is calculated at initialization.
        factorize_hessian : `bool`, optional
            If ``True``, solve the required linear system using a Cholesky
            factorization.  If ``False``, use the conjugate gradient algorithm
            to avoid forming or inverting the Hessian.
        hyper_par_objective_fun : `callable`, optional
            The part of ``objective_fun`` depending on both ``opt_par`` and
            ``hyper_par``.  The arguments must be the same as
            ``objective_fun``:
            - ``opt_par``: The parameter to be optimized (`numpy.ndarray` (N,))
            - ``hyper_par``: A hyperparameter (`numpy.ndarray` (N,))
            This can be useful if only a small part of the objective function
            depends on both ``opt_par`` and ``hyper_par``.  If not specified,
            ``objective_fun`` is used.
        grad_tol : `float`, optional
            The tolerance used to check that the gradient is approximately
            zero at the optimum.
        """

        warnings.warn(
            'This class is being moved to the vittles package.',
            DeprecationWarning)

        self._objective_fun = objective_fun
        self._obj_fun_grad = autograd.grad(self._objective_fun, argnum=0)
        self._obj_fun_hessian = autograd.hessian(self._objective_fun, argnum=0)
        self._obj_fun_hvp = autograd.hessian_vector_product(
            self._objective_fun, argnum=0)

        if hyper_par_objective_fun is None:
            self._hyper_par_objective_fun = self._objective_fun
            self._hyper_obj_fun = self._objective_fun
        else:
            self._hyper_par_objective_fun = hyper_par_objective_fun

        # TODO: is this the right default order?  Make this flexible.
        self._hyper_obj_fun_grad = \
            autograd.grad(self._hyper_par_objective_fun, argnum=0)
        self._hyper_obj_cross_hess = autograd.jacobian(
            self._hyper_obj_fun_grad, argnum=1)

        self._grad_tol = grad_tol

        self.set_base_values(
            opt_par_value, hyper_par_value,
            hessian_at_opt, cross_hess_at_opt,
            factorize_hessian,
            validate_optimum=validate_optimum,
            grad_tol=self._grad_tol)

    def set_base_values(self,
                        opt_par_value, hyper_par_value,
                        hessian_at_opt, cross_hess_at_opt,
                        factorize_hessian,
                        validate_optimum=True, grad_tol=None):

        # Set the values of the optimal parameters.
        self._opt0 = deepcopy(opt_par_value)
        self._hyper0 = deepcopy(hyper_par_value)

        # Set the values of the Hessian at the optimum.
        if hessian_at_opt is None:
            self._hess0 = self._obj_fun_hessian(self._opt0, self._hyper0)
        else:
            self._hess0 = hessian_at_opt
        if self._hess0.shape != (len(self._opt0), len(self._opt0)):
            raise ValueError('``hessian_at_opt`` is the wrong shape.')

        method = 'factorization' if factorize_hessian else 'cg'
        self.hess_solver = HessianSolver(self._hess0, method)

        if validate_optimum:
            if grad_tol is None:
                grad_tol = self._grad_tol

            # Check that the gradient of the objective is zero at the optimum.
            grad0 = self._obj_fun_grad(self._opt0, self._hyper0)
            newton_step = -1 * self.hess_solver.solve(grad0)

            newton_step_norm = np.linalg.norm(newton_step)
            if newton_step_norm > grad_tol:
                err_msg = \
                    'The gradient is not zero at the putatively optimal ' + \
                    'values.  ||newton_step|| = {} > {} = grad_tol'.format(
                        newton_step_norm, grad_tol)
                raise ValueError(err_msg)

        if cross_hess_at_opt is None:
            self._cross_hess = self._hyper_obj_cross_hess(self._opt0, self._hyper0)
        else:
            self._cross_hess = cross_hess_at_opt
        if self._cross_hess.shape != (len(self._opt0), len(self._hyper0)):
            raise ValueError('``cross_hess_at_opt`` is the wrong shape.')

        self._sens_mat = -1 * self.hess_solver.solve(self._cross_hess)


    # Methods:
    def get_dopt_dhyper(self):
        return self._sens_mat

    def get_hessian_at_opt(self):
        return self._hess0

    def predict_opt_par_from_hyper_par(self, new_hyper_par_value):
        """
        Predict ``opt_par`` using the linear approximation.

        Parameters
        ------------
        new_hyper_par_value: `numpy.ndarray` (M,)
            The value of ``hyper_par`` at which to approximate ``opt_par``.
        """
        return \
            self._opt0 + \
            self._sens_mat @ (new_hyper_par_value - self._hyper0)


################################
# Higher-order approximations. #
################################

class ParametricSensitivityTaylorExpansion(object):
    def __init__(self, objective_function,
                 input_val0, hyper_val0, order,
                 hess0=None,
                 hyper_par_objective_function=None):
        raise NotImplementedError(
            'This class is now implemented in the ``vittles`` package.')



class SparseBlockHessian():
    def __init__(self, objective_function, sparsity_array):
        raise NotImplementedError(
            'This class is now implemented in the ``vittles`` package.')
