# Define some forward-diff functions for np.linalg that are currently excluded
# from autogra.d
#
# Most of these are copied with minimal modification from
# https://github.com/HIPS/autograd/blob/65c21e2/autograd/numpy/linalg.py

import autograd
import autograd.numpy as np
import autograd.scipy as sp
import scipy as osp
from scipy import sparse
from autograd.core import primitive, defvjp, defjvp

from autograd.numpy.linalg import slogdet, solve, inv
from functools import partial



defjvp(sp.special.gammasgn, None)
defjvp(sp.special.polygamma,
    None, lambda g, ans, n, x: g * sp.special.polygamma(n + 1, x))
defjvp(sp.special.psi,       lambda g, ans, x: g * sp.special.polygamma(1, x))
defjvp(sp.special.digamma,   lambda g, ans, x: g * sp.special.polygamma(1, x))
defjvp(sp.special.gamma,     lambda g, ans, x: g * ans * sp.special.psi(x))
defjvp(sp.special.gammaln,   lambda g, ans, x: g * sp.special.psi(x))
defjvp(sp.special.rgamma,
    lambda g, ans, x: g * sp.special.psi(x) / -sp.special.gamma(x))
# defjvp(sp.special.multigammaln,
#        lambda g, ans, a, d:
#         g * np.sum(sp.special.digamma(np.expand_dims(a, -1) - np.arange(d)/2.), -1),
#        None)


# transpose by swapping last two dimensions
def T(x): return np.swapaxes(x, -1, -2)

def inv_jvp(g, ans, x):
    dot = np.dot if ans.ndim == 2 else partial(np.einsum, '...ij,...jk->...ik')
    return -dot(dot(ans, g), ans)

defjvp(inv, inv_jvp)

def jvp_solve(argnum, g, ans, a, b):
    def broadcast_matmul(a, b):
        return \
            np.matmul(a, b) if b.ndim == a.ndim \
            else np.matmul(a, b[..., None])[..., 0]
    if argnum == 0:
        return -broadcast_matmul(np.linalg.solve(a, g), ans)
    else:
        return np.linalg.solve(a, g)

defjvp(solve, partial(jvp_solve, 0), partial(jvp_solve, 1))


def slogdet_jvp(g, ans, x):
    # Due to https://github.com/HIPS/autograd/issues/115
    # and https://github.com/HIPS/autograd/blob/65c21e/tests/test_numpy.py#L302
    # it does not seem easy to take the trace of the last two dimensions of
    # a multi-dimensional array at this time.
    if len(x.shape) > 2:
        raise ValueError('JVP is only supported for 2d input.')
    return 0, np.trace(np.linalg.solve(x.T, g.T))

defjvp(slogdet, slogdet_jvp)


def get_sparse_product(z_mat):
    """
    Return an autograd-compatible function that calculates
    ``z_mat @ a`` and ``z_mat.T @ a`` when ``z_mat`` is a sparse matrix.

    Parameters
    ------------
    z_mat: A 2d matrix
        The matrix by which to multiply.  The matrix can be dense, but the only
        reason to use ``get_sparse_product`` is with a sparse matrix since
        dense matrix multiplication is supported natively by ``autograd``.

    Returns
    -----------
    z_mult:
        A function such that ``z_mult(b) = z_mat @ b``.
    zt_mult:
        A function such that ``zt_mult(b) = z_mat.T @ b``.
    Unlike standard sparse matrix multiplication, ``z_mult`` and ``zt_mult``
    can be used with ``autograd``.
    """

    if z_mat.ndim != 2:
        raise ValueError(
            'get_sparse_product can only be used with 2d arrays.')

    def check_b(b):
        b = np.atleast_1d(b)
        if (b.ndim > 2):
            raise ValueError('The argument must be at most two dimensional.')
        return b

    @primitive
    def z_mult(b):
        return z_mat @ check_b(b)

    @primitive
    def zt_mult(b):
        return z_mat.T @ check_b(b)

    def z_mult_jvp(g, ans, b):
        return z_mult(g) # z_mat @ g
    defjvp(z_mult, z_mult_jvp)

    def z_mult_vjp(ans, b):
        def vjp(g):
            return zt_mult(g) # z_mat.T @ g
        return vjp
    defvjp(z_mult, z_mult_vjp)

    def zt_mult_jvp(g, ans, b):
        return zt_mult(g) # z_mat.T @ g
    defjvp(zt_mult, zt_mult_jvp)

    def zt_mult_vjp(ans, b):
        def vjp(g):
            return z_mult(g) # (z_mat.T).T @ g
        return vjp
    defvjp(zt_mult, zt_mult_vjp)

    return z_mult, zt_mult


def get_differentiable_solver(z_solve, zt_solve):
    """
    Return an autograd-compatible function that calculates
    ``z_solve(b) = z^{-1} b`` where the solver may not be natively
    differentiable by autograd.

    Parameters
    ------------
    z_solve, zt_solve:
        Functions that take a vector input ``b`` and return ``z^{-1} b`` and
        ``(z^T)^{-1} b``, respectively.

    Returns
    -----------
    z_solve_ad, zt_solve_ad:
        Respective versions of ``z_solve`` and ``zt_solve`` that can be
        differentiated by autograd.

    This is particularly useful for differentiating the solutions of systems
    where the matrix ``z`` is sparse.
    """

    @primitive
    def z_solve_ad(b):
        return z_solve(b)

    @primitive
    def zt_solve_ad(b):
        return zt_solve(b)

    def get_grad_funs(z_solve_ad, zt_solve_ad):
        # Reverse mode
        def vjp_solve(ans, b):
            return lambda g: zt_solve_ad(g)

        # Forward mode
        def jvp_solve(g, ans, b):
            return z_solve_ad(g)

        return vjp_solve, jvp_solve

    vjp_solve, jvp_solve = get_grad_funs(z_solve_ad, zt_solve_ad)
    vjp_t_solve, jvp_t_solve = get_grad_funs(zt_solve_ad, z_solve_ad)

    defjvp(z_solve_ad, jvp_solve)
    defjvp(zt_solve_ad, jvp_t_solve)
    defvjp(z_solve_ad, vjp_solve)
    defvjp(zt_solve_ad, vjp_t_solve)

    return z_solve_ad, zt_solve_ad


@primitive
def grouped_sum(x, groups, num_groups=None):
    """Sum the array `x` by its first index according to indices in `groups`.

    Parameters
    ------------
    x: numpy.ndarray
        An array of dimension (N, D1, ..., DK)
    groups:
        A length-N vector of zero-indexed integers mapping the first index
        of x to groups.
    num_groups:
        Optional, the total number of groups.  If unspecified, one plus the
        largest element of `groups` is used.

    Returns
    -----------
    A (num_groups, D1, ..., DK) dimensional vector, where entry [g, ...]
    contains the sum of the entries `x[n, :]`` where `groups[n] == g`.
    """
    x = np.atleast_1d(x)
    groups = np.atleast_1d(groups).astype('int64')
    if (groups.ndim > 1):
        raise ValueError('groups must be a vector.')

    n_obs = len(groups)
    if x.shape[0] != n_obs:
        raise ValueError('The first dimension of x must match the length of groups')
    max_group = np.max(groups)
    if num_groups is None:
        num_groups = max_group + 1
    else:
        if max_group >= num_groups:
            raise ValueError(
                'The largest group is >= the number of groups.')

    result = np.zeros((num_groups, ) + x.shape[1:])
    for n in range(n_obs):
        if x.ndim > 1:
            result[groups[n], :] += x[n, :]
        else:
            result[groups[n]] += x[n]
    return result

def _ungroup(v, groups):
    if v.ndim > 1:
        return v[groups, :]
    else:
        return v[groups]

def grouped_sum_vjp(ans, x, groups, num_groups=None):
    def vjp(v):
        return _ungroup(v, groups)
    return vjp
defvjp(grouped_sum, grouped_sum_vjp)

def grouped_sum_jvp(v, ans, x, groups, num_groups=None):
    return grouped_sum(v, groups, num_groups=num_groups)
defjvp(grouped_sum, grouped_sum_jvp)



@primitive
def replace(x_sub, x, inds):
    """Differentiably replace elements of `x[inds]` with `x_sub` by making a copy.

    Parameters
    ------------
    x_sub: numpy.ndarray[D]
        A numeric array with replacement values.
    x: numpy.ndarray[N]
        A numeric array with values to be replaced.
    num_groups: numpy.ndarray[D]
        The indices to replace.  We will set x[inds] = x_sub.

    Returns
    -----------
    A new value of x, with the values in inds replaced with x_sub.
    """
    x_new = np.full(x.shape, float('nan'))
    x_new[:] = x
    x_new[inds] = x_sub
    return x_new

defvjp(replace,
       lambda ans, x_sub, x, inds: lambda g: g[inds],
       lambda ans, x_sub, x, inds: lambda g: replace(0, g, inds),
       None)

defjvp(replace,
       lambda g, ans, x_sub, x, inds: replace(g, np.zeros(len(x)), inds),
       lambda g, ans, x_sub, x, inds: replace(0, g, inds),
       None)
