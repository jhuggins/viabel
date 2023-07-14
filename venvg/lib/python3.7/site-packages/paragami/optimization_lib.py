import autograd
import autograd.numpy as np
import copy
import scipy as osp
import scipy.sparse
import warnings

from .autograd_supplement_lib import \
    get_sparse_product, get_differentiable_solver

###############################
# Preconditioned objectives.  #
###############################


def truncate_eigenvalues(evals, ev_min=None, ev_max=None):
    """Truncate the vector ``evals`` so that values lower than
    ``ev_min`` are replaced with ``ev_min`` and values larger than
    ``ev_max`` are replaced with ``ev_max``.

    Parameters
    ------------
    evals: `np.ndarray` (N, )
    ev_min: `float`, optional
        If ``None``, no lower truncation is done.
    ev_max: `float`, optional
        If ``None``, no upper truncation is done.

    Returns
    ---------
    eig_val_trunc
        A truncated version of ``evals``.
    """
    eig_val_trunc = copy.deepcopy(evals)
    if not ev_min is None:
        if not np.isreal(ev_min):
            raise ValueError('ev_min must be real-valued.')
        ev_min = float(ev_min)
        eig_val_trunc[np.real(eig_val_trunc) <= ev_min] = ev_min
    if not ev_max is None:
        if not np.isreal(ev_max):
            raise ValueError('ev_max must be real-valued.')
        ev_max = float(ev_max)
        eig_val_trunc[np.real(eig_val_trunc) >= ev_max] = ev_max
    return eig_val_trunc


def transform_eigenspace(eigvecs, eigvals, transform_function):
    """Return a function that multiplies a vector by a matrix with
    transformed eigenvalues.

    Let ``eigvecs`` and ``eigvals`` be selected eigenvectors and
    eigenvalues.  The columns of ``eigvecs``
    are the eigenvectors associated with the corresponding entry of
    ``eigvals``.  A function, ``a_mult``, is returned.  This function
    is the identity for all directions orthogonal to ``eigvecs``,
    and has eigenvalues ``transform_function(eigvals)`` in the
    space spanned by ``eigvecs``.

    Parameters
    ------------
    eigvecs: `numpy.ndarray` (N, K)
        The eigenvectors.
    eigvals: `numpy.ndarray` (K,)
        The eigenvalues
    transform_function: callable
        A function from ``eigvals`` to a vector of the same length.
        The output of ``transform_function(eigvals)`` will be the new
        eigenvalues.

    Returns
    -----------
    a_mult: callable
        A linear function from a length-``K`` numpy vector to another
        length-``K`` vector with the above-described eigendecomposition.
    """

    if eigvecs.ndim != 2:
        raise ValueError('``eigvecs`` must be 2d.')
    if eigvals.ndim != 1:
        raise ValueError('``eigvals`` must be 1d.')
    if eigvecs.shape[1] != len(eigvals):
        raise ValueError(
            'The columns of ``eigvecs`` and length of ``eigvals`` must match.')

    new_eigvals = transform_function(eigvals)

    def a_mult(vec):
        vec_loadings = eigvecs.T @ vec
        # Equivalent to the more transparent:
        # vec_perp = vec - eigvecs @ vec_loadings
        # return vec_perp + eigvecs @ (new_eigvals * vec_loadings)
        return vec + eigvecs @ ((new_eigvals - 1) * vec_loadings)

    return a_mult


def _get_sym_matrix_inv_sqrt_funcs(mat, ev_min=None, ev_max=None):
    """
    Get the inverse square root of a symmetric matrix with thresholds for the
    eigenvalues.

    This is useful for calculating preconditioners.
    """
    mat = np.atleast_2d(mat)

    # Symmetrize for numerical stability.
    mat_sym = 0.5 * (mat + mat.T)
    eig_val, eig_vec = np.linalg.eigh(mat_sym)

    eig_val_trunc = truncate_eigenvalues(eig_val, ev_min=ev_min, ev_max=ev_max)

    mult_mat_sqrt = \
        transform_eigenspace(eig_vec, eig_val_trunc, np.sqrt)

    mult_mat_inv_sqrt = \
        transform_eigenspace(eig_vec, eig_val_trunc, lambda x: 1. / np.sqrt(x))

    return mult_mat_sqrt, mult_mat_inv_sqrt


def _get_matrix_from_operator(mult_fun, dim):
    """Get a matrix representation of a linear operator on vectors.

    Parameters
    -------------
    mult_fun: `callable`
        A function linearly that maps vectors to vectors.
    dim: `int`
        The dimension of the input to ``mult_fun``.

    Returns
    ----------
    A matrix representation of the operator ``mult_fun``.
    """
    mat = []
    for i in range(dim):
        # Re-instantiate to make sure that mult_fun is not returning a reference.
        vec = np.zeros(dim)
        vec[i] = 1.0
        mat.append(mult_fun(vec))
    return np.vstack(mat).T


class PreconditionedFunction():
    """
    Get a function whose input has been preconditioned.

    Throughout, the subscript ``_c`` will denote quantiites or
    funcitons in the preconditioned space.  For example, ``x`` will
    refer to a variable in the original space and ``x_c`` to the same
    variable after preconditioning.

    Preconditioning means transforming :math:`x \\rightarrow x_c = A^{-1} x`,
    where the matrix :math:`A` is the "preconditioner".  If :math:`f` operates
    on :math:`x`, then the preconditioned function operates on :math:`x_c` and
    is defined by :math:`f_c(x_c) := f(A x_c) = f(x)`. Gradients of the
    preconditioned function are defined with respect to its argument in the
    preconditioned space, e.g., :math:`f'_c = \\frac{df_c}{dx_c}`.

    A typical value of the preconditioner is an inverse square root of the
    Hessian of :math:`f`, because then the Hessian of :math:`f_c` is
    the identity when the gradient is zero.  This can help speed up the
    convergence of optimization algorithms.

    Methods
    ----------
    set_preconditioner:
        Set the preconditioner to a specified value.
    set_preconditioner_with_hessian:
        Set the preconditioner based on the Hessian of the objective
        at a point in the orginal domain.
    precondition:
        Convert from the original domain to the preconditioned domain.
    unprecondition:
        Convert from the preconditioned domain to the original domain.
    """
    def __init__(self, original_fun):
        """
        Parameters
        -------------
        original_fun:
            callable function of a single argument
        preconditioner:
            The initial preconditioner.
        preconditioner_inv:
            The inverse of the initial preconditioner.
        """
        self._original_fun = original_fun

        # Initialize to the identity preconditioner.
        self.set_identitity_preconditioner()

    def set_identitity_preconditioner(self):
        self.set_preconditioner_functions(lambda x: x, lambda x: x)

    def set_preconditioner_matrix(self, a, a_inv=None):
        """Set the preconditioner with a matrix.
        """
        if osp.sparse.issparse(a):
            if a_inv is None:
                a_inv = osp.sparse.linalg.inv(a)
            a_times, _ = get_sparse_product(a)
            a_inv_times, _ = get_sparse_product(a_inv)
            self.set_preconditioner_functions(a_times, a_inv_times)

        else:
            def a_times(vec):
                return a @ vec

            if a_inv is None:
                # On one hand, this is a numerically instable way to solve a
                # linear system.  On the other hand, the inverse is
                # readily available from the eigenvalue decomposition
                # and the Cholesky factorization is not AFAIK.
                a_inv = np.linalg.inv(a)

            def a_inv_times(vec):
                return a_inv @ vec

            self.set_preconditioner_functions(a_times, a_inv_times)

    def set_preconditioner_functions(self, a_times, a_inv_times):
        """Set the preconditioner with a functions that perform matrix
        multiplication.
        """
        self._a_times = a_times
        self._a_inv_times = a_inv_times

    def check_preconditioner(self, vec, tol=1e-8):
        err = np.linalg.norm(vec - self._a_times(self._a_inv_times(vec)))
        if err > tol:
            raise ValueError(
                ('``a_times`` does not invert ``a_inv_times``.  ' +
                'Error {} > tol {}').format(err, tol))

    def set_preconditioner(self, preconditioner, preconditioner_inv=None):
        warnings.warn(
            '``set_preconditioner`` is deprecated.  Please use ' +
            '``set_preconditioner_matrix``',
            DeprecationWarning)
        self.set_preconditioner_matrix(preconditioner, preconditioner_inv)

    def set_preconditioner_with_hessian(self, x=None, hessian=None,
                                        ev_min=None, ev_max=None):
        """
        Set the preconditioner to the inverse square root of the Hessian of
        the original objective (or an approximation thereof).

        Parameters
        ---------------
        x: Numeric vector
            The point at which to evaluate the Hessian of ``original_fun``.
            If x is specified, the Hessian is evaluated with automatic
            differentiation.
            Specify either x or hessian but not both.
        hessian: Numeric matrix
            The hessian of ``original_fun`` or an approximation of it.
            Specify either x or hessian but not both.
        ev_min: float
            If not None, set eigenvaluse of ``hessian`` that are less than
            ``ev_min`` to ``ev_min`` before taking the square root.
        ev_maxs: float
            If not None, set eigenvaluse of ``hessian`` that are greater than
            ``ev_max`` to ``ev_max`` before taking the square root.

        Returns
        ------------
        Sets the precoditioner for the class and returns the Hessian with
        the eigenvalues thresholded by ``ev_min`` and ``ev_max``.
        """
        if x is not None and hessian is not None:
            raise ValueError('You must specify x or hessian but not both.')
        if x is None and hessian is None:
            raise ValueError('You must specify either x or hessian.')
        if hessian is None:
            # We now know x is not None.
            get_original_fun_hessian = autograd.hessian(self._original_fun)
            hessian = get_original_fun_hessian(x)

        if osp.sparse.issparse(hessian):
            raise NotImplementedError(
                '``set_preconditioner_with_hessian`` no longer works with ' +
                'sparse Hessians directly.  Instead, ' +
                'use ``sparse_preconditioners_lib.' +
                'get_preconditioner_functions_from_sparse_hessian``.')
        else:
            mult_hess_sqrt, mult_hess_inv_sqrt = \
                _get_sym_matrix_inv_sqrt_funcs(
                    hessian, ev_min=ev_min, ev_max=ev_max)
        self.set_preconditioner_functions(mult_hess_inv_sqrt, mult_hess_sqrt)

    def precondition(self, x):
        """
        Multiply by the inverse of the preconditioner to convert
        :math:`x` in the original domain to :math:`x_c` in the preconditioned
        domain.

        This function is provided for convenience, but it is more numerically
        stable to use np.linalg.solve(preconditioner, x).
        """
        return self._a_inv_times(x)

    def unprecondition(self, x_c):
        """
        Multiply by the preconditioner to convert
        :math:`x_c` in the preconditioned domain to :math:`x` in the
        original domain.
        """
        return self._a_times(x_c)

    def get_preconditioner(self, dim):
        """Return a matrix representation of the preconditioner.  This may
        be expensive and is intended for testing only.
        """
        return _get_matrix_from_operator(self._a_times, dim)

    def get_preconditioner_inv(self, dim):
        """Return a matrix representation of the preconditioner.  This may
        be expensive and is intended for testing only.
        """
        return _get_matrix_from_operator(self._a_inv_times, dim)

    def __call__(self, x_c):
        """
        Evaluate the preconditioned function at a point in the preconditioned
        domain.
        """
        return self._original_fun(self.unprecondition(x_c))



class OptimizationObjective():
    """
    Derivatives and logging for an optimization objective function.

    Attributes
    -------------
    optimization_log: Dictionary
        A record of the optimization progress as recorded by ``log_value``.

    Methods
    ---------------
    f:
        The objective function with logging.
    grad:
        The gradient of the objective function.
    hessian:
        The Hessian of the objective function.
    hessian_vector_product:
        The Hessian vector product of the objective function.
    set_print_every:
        Set how often to display optimization progress.
    set_log_every:
        Set how often to log optimization progress.
    reset_iteration_count:
        Reset the number of iterations for the purpose of printing and logging.
    reset_log:
        Clear the log.
    reset:
        Run ``reset_iteration_count`` and ``reset_log``.
    print_value:
        Display a function evaluation.
    log_value:
        Log a function evaluation.
    """
    def __init__(self, objective_fun, print_every=1, log_every=0):
        """
        Parameters
        -------------
        obj_fun: Callable function of one argumnet
            The function to be minimized.
        print_every: integer
            Print the optimization value every ``print_every`` iterations.
        log_every: integer
            Log the optimization value every ``log_every`` iterations.
        """

        self._objective_fun = objective_fun
        self.grad = autograd.grad(self._objective_fun)
        self.hessian = autograd.hessian(self._objective_fun)
        self.hessian_vector_product = \
            autograd.hessian_vector_product(self._objective_fun)

        self.set_print_every(print_every)
        self.set_log_every(log_every)

        self.reset()

    def set_print_every(self, n):
        """
        Parameters
        -------------
        n: integer
            Print the objective function value every ``n`` iterations.
            If 0, do not print any output.
        """
        self._print_every = n

    def set_log_every(self, n):
        """
        Parameters
        -------------
        n: integer
            Log the objective function value every ``n`` iterations.
            If 0, do not log.
        """
        self._log_every = n

    def reset(self):
        """
        Reset the itreation count and clear the log.
        """
        self.reset_iteration_count()
        self.reset_log()

    def reset_iteration_count(self):
        self._num_f_evals = 0

    def num_iterations(self):
        """
        Return the number of times the optimization function has been called,
        not counting any derivative evaluations.
        """
        return self._num_f_evals

    def print_value(self, num_f_evals, x, f_val):
        """
        Display the optimization progress.  To display a custom
        update, overload this function.

        Parameters
        -------------
        num_f_vals: Integer
            The total number of function evaluations.
        x:
            The current argument to the objective function.
        f_val:
            The value of the objective at ``x``.
        """
        print('Iter {}: f = {:0.8f}'.format(num_f_evals, f_val))

    def reset_log(self):
        self.optimization_log = []

    def log_value(self, num_f_evals, x, f_val):
        """
        Log the optimization progress.  To create a custom log,
        overload this function.  By default, the log is a list of tuples
        ``(iteration, x, f(x))``.

        Parameters
        -------------
        num_f_vals: Integer
            The total number of function evaluations.
        x:
            The current argument to the objective function.
        f_val:
            The value of the objective at ``x``.
        """
        self.optimization_log.append((num_f_evals, x, f_val))

    def f(self, x):
        f_val = self._objective_fun(x)
        if self._print_every > 0 and self._num_f_evals % self._print_every == 0:
            self.print_value(self._num_f_evals, x, f_val)
        if self._log_every > 0 and self._num_f_evals % self._log_every == 0:
            self.log_value(self._num_f_evals, x, f_val)
        self._num_f_evals += 1
        return f_val
