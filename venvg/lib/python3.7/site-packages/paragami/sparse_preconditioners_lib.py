# Only this library depends on scikit-sparse, and scikit-sparse depends on
# a C++ library.  So for easy of installation, this library is only imported
# if the user needs it, and scikit-sparse is inlucded in requirements-dev
# but not in requirements.

try:
    from sksparse.cholmod import cholesky
except ImportError:
    error = ('``sparse_preconditioners_lib`` requirs the ``scikit-sparse``' +
             'package.  ' +
             'For simplicity, this package is not a core requirement for ' +
             'paragami.  ' +
             'See https://github.com/scikit-sparse/scikit-sparse/ ' +
             'for installation instructions.')
    print(error)
    raise

import scipy as osp
import scipy.sparse
from .autograd_supplement_lib import \
    get_sparse_product, get_differentiable_solver

# For sparse preconditioners.
def _get_cholesky_sqrt_mat(mat_chol):
    """Extract the actual Cholesky square root from a decomposition provided
    by ``sksparse.cholmod.cholesky``.
    """
    return mat_chol.apply_Pt(mat_chol.L())


def _get_sparse_square_root_operators(mat_chol):
    """Get preconditioners from a sparse matrix.  The argument should be the
    output of ``sksparse.cholmod.cholesky``
    """
    mat_sqrt = _get_cholesky_sqrt_mat(mat_chol)

    # To avoid the sparse efficiency warning.
    mat_sqrt_t = osp.sparse.csc_matrix(mat_sqrt.T)

    _, mult_mat_sqrt_t_ad = get_sparse_product(mat_sqrt)
    _, solve_mat_sqrt_t_ad = \
        get_differentiable_solver(
            osp.sparse.linalg.factorized(mat_sqrt),
            osp.sparse.linalg.factorized(mat_sqrt_t))

    return solve_mat_sqrt_t_ad, mult_mat_sqrt_t_ad


def get_sym_matrix_inv_sqrt_funcs(hessian):
    """Get preconditioning functions from a sparse representation of a Hessian.

    Parameters
    ---------------
    hessian: Sparse matrix
        A sparse representation of the Hessian of the objective function
        which you want to precondition.

    Returns
    -----------
    mult_hess_sqrt mult_hess_inv_sqrt: Callable functions
        ``mult_hess_sqrt`` is linear function that multiplies a vector argument
        by a square root of the ``hessian`` argument, and ``mult_hess_inv_sqrt``
        is the inverse of ``mult_hess_sqrt``.  These functions
        can be passed to the ``set_preconditioner_functions`` method of
        an ``optimization_lib.PreconditionedFunction`` class to use a sparse
        preconditioner.
    """

    if not osp.sparse.issparse(hessian):
        raise ValueError('``hessian`` needs to be sparse.')

    # Use the Cholesky square root.
    hessian_chol = cholesky(hessian)
    mult_hess_inv_sqrt, mult_hess_sqrt = \
        _get_sparse_square_root_operators(hessian_chol)
    return mult_hess_sqrt, mult_hess_inv_sqrt
