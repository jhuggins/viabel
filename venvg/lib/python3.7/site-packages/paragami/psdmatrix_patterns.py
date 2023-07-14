from .base_patterns import Pattern
from .pattern_containers import register_pattern_json

import autograd.numpy as np

import math

from autograd.core import primitive, defvjp, defjvp


def _sym_index(k1, k2):
    """
    Get the index of an entry in a folded symmetric array.

    Parameters
    ------------
    k1, k2: int
        0-based indices into a symmetric matrix.

    Returns
    --------
    int
        Return the linear index of the (k1, k2) element of a symmetric
        matrix where the triangular part has been stacked into a vector.
    """
    def ld_ind(k1, k2):
        return int(k2 + k1 * (k1 + 1) / 2)

    if k2 <= k1:
        return ld_ind(k1, k2)
    else:
        return ld_ind(k2, k1)


def _vectorize_ld_matrix(mat):
    """
    Linearize the lower diagonal of a square matrix.

    Parameters:
    mat
        A square matrix.

    Returns:
    1-d vector
        The lower diagonal of `mat` stacked into a vector.

    Specifically, we map the matrix

    [ x11 x12 ... x1n ]
    [ x21 x22     x2n ]
    [...              ]
    [ xn1 ...     xnn ]

    to the vector

    [ x11, x21, x22, x31, ..., xnn ].

    The entries above the diagonal are ignored.
    """
    nrow, ncol = np.shape(mat)
    if nrow != ncol:
        raise ValueError('mat must be square')
    return mat[np.tril_indices(nrow)]


@primitive
def _unvectorize_ld_matrix(vec):
    """
    Invert the mapping of `_vectorize_ld_matrix`.

    Parameters
    -----------
    vec: A 1-d vector.

    Returns
    ----------
    A symmetric matrix.

    Specifically, we map a vector

    [ v1, v2, ..., vn ]

    to the symmetric matrix

    [ v1 ...          ]
    [ v2 v3 ...       ]
    [ v4 v5 v6 ...    ]
    [ ...             ]

    where the values above the diagonal are determined by symmetry.
    """
    mat_size = int(0.5 * (math.sqrt(1 + 8 * vec.size) - 1))
    if mat_size * (mat_size + 1) / 2 != vec.size:
        raise ValueError('Vector is an impossible size')
    mat = np.zeros((mat_size, mat_size))
    for k1 in range(mat_size):
        for k2 in range(k1 + 1):
            mat[k1, k2] = vec[_sym_index(k1, k2)]
    return mat

# Because we cannot use autograd with array assignment, define the
# vector jacobian product and jacobian vector products of
# _unvectorize_ld_matrix.


def _unvectorize_ld_matrix_vjp(g):
    assert g.shape[0] == g.shape[1]
    return _vectorize_ld_matrix(g)


defvjp(_unvectorize_ld_matrix,
       lambda ans, vec: lambda g: _unvectorize_ld_matrix_vjp(g))


def _unvectorize_ld_matrix_jvp(g):
    return _unvectorize_ld_matrix(g)


defjvp(_unvectorize_ld_matrix,
       lambda g, ans, x: _unvectorize_ld_matrix_jvp(g))


def _exp_matrix_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]
    # NB: make_diagonal() is only defined in the autograd version of numpy
    mat_exp_diag = np.make_diagonal(
        np.exp(np.diag(mat)), offset=0, axis1=-1, axis2=-2)
    mat_diag = np.make_diagonal(np.diag(mat), offset=0, axis1=-1, axis2=-2)
    return mat_exp_diag + mat - mat_diag


def _log_matrix_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]
    # NB: make_diagonal() is only defined in the autograd version of numpy
    mat_log_diag = np.make_diagonal(
        np.log(np.diag(mat)), offset=0, axis1=-1, axis2=-2)
    mat_diag = np.make_diagonal(np.diag(mat), offset=0, axis1=-1, axis2=-2)
    return mat_log_diag + mat - mat_diag


def _pack_posdef_matrix(mat, diag_lb=0.0):
    k = mat.shape[0]
    mat_lb = mat - np.make_diagonal(
        np.full(k, diag_lb), offset=0, axis1=-1, axis2=-2)
    return _vectorize_ld_matrix(
        _log_matrix_diagonal(np.linalg.cholesky(mat_lb)))


def _unpack_posdef_matrix(free_vec, diag_lb=0.0):
    mat_chol = _exp_matrix_diagonal(_unvectorize_ld_matrix(free_vec))
    mat = np.matmul(mat_chol, mat_chol.T)
    k = mat.shape[0]
    return mat + np.make_diagonal(
        np.full(k, diag_lb), offset=0, axis1=-1, axis2=-2)


# Convert a vector containing the lower diagonal portion of a symmetric
# matrix into the full symmetric matrix.
#
# This is not currently used but could be useful for a symmetric matrix type.
def _unvectorize_symmetric_matrix(vec_val):
    ld_mat = _unvectorize_ld_matrix(vec_val)
    mat_val = ld_mat + ld_mat.transpose()
    # We have double counted the diagonal.  For some reason the autograd
    # diagonal functions require axis1=-1 and axis2=-2
    mat_val = mat_val - \
        np.make_diagonal(np.diagonal(ld_mat, axis1=-1, axis2=-2),
                         axis1=-1, axis2=-2)
    return mat_val


class PSDSymmetricMatrixPattern(Pattern):
    """A pattern for a symmetric, positive-definite matrix parameter.

    Attributes
    -------------
    validate: Bool
        Whether or not the matrix is automatically checked for symmetry
        positive-definiteness, and the diagonal lower bound.
    """
    def __init__(self, size, diag_lb=0.0, default_validate=True,
                 free_default=None):
        """
        Parameters
        --------------
        size: `int`
            The length of one side of the square matrix.
        diag_lb: `float`
            A lower bound for the diagonal entries.  Must be >= 0.
        default_validate: `bool`, optional
            Whether or not to check for legal (i.e., symmetric
            positive-definite) folded values by default.
        free_default: `bool`, optional
            Default setting for free.
        """
        self.__size = int(size)
        self.__diag_lb = diag_lb
        self.default_validate = default_validate
        if diag_lb < 0:
            raise ValueError(
                'The diagonal lower bound diag_lb must be >-= 0.')

        super().__init__(self.__size ** 2, int(size * (size + 1) / 2),
                         free_default=free_default)

    def __str__(self):
        return 'PDMatrix {}x{} (diag_lb = {})'.format(
            self.__size, self.__size, self.__diag_lb)

    def as_dict(self):
        return {
            'pattern': self.json_typename(),
            'size': self.__size,
            'diag_lb': self.__diag_lb,
            'default_validate': self.default_validate}

    def size(self):
        """Returns the matrix size.
        """
        return self.__size

    def shape(self):
        """Returns the matrix shape, i.e., (size, size).
        """
        return (self.__size, self.__size)

    def diag_lb(self):
        """Returns the diagonal lower bound.
        """
        return self.__diag_lb

    def empty(self, valid):
        if valid:
            return np.eye(self.__size) * (self.__diag_lb + 1)
        else:
            return np.empty((self.__size, self.__size))

    def _validate_folded_shape(self, folded_val):
        expected_shape = (self.__size, self.__size)
        if folded_val.shape != (self.__size, self.__size):
            return \
                False, 'The matrix is not of shape {}'.format(expected_shape)
        else:
            return True, ''

    def validate_folded(self, folded_val, validate_value=None):
        """Check that the folded value is valid.

        If `validate_value = True`, checks that `folded_val` is a symmetric,
        matrix of the correct shape with diagonal entries
        greater than the specified lower bound.  Otherwise,
        only the shape is checked.

        .. note::
            This method does not currently check for positive-definiteness.

        Parameters
        -----------
        folded_val : Folded value
            A candidate value for a positive definite matrix.
        validate_value: `bool`, optional
            Whether to check the matrix for attributes other than shape.
            If `None`, the value of `self.default_validate` is used.

        Returns
        ----------
        is_valid : `bool`
            Whether ``folded_val`` is a valid positive semi-definite matrix.
        err_msg : `str`
            A message describing the reason the value is invalid or an empty
            string if the value is valid.
        """
        shape_ok, err_msg = self._validate_folded_shape(folded_val)
        if not shape_ok:
            raise ValueError(err_msg)

        if validate_value is None:
            validate_value = self.default_validate

        if validate_value:
            if np.any(np.diag(folded_val) < self.__diag_lb):
                error_string = \
                    'Diagonal is less than the lower bound {}.'.format(
                        self.__diag_lb)
                return False, error_string
            if not (folded_val.transpose() == folded_val).all():
                return False, 'Matrix is not symmetric.'

        return True, ''

    def flatten(self, folded_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        valid, msg = self.validate_folded(folded_val, validate_value)
        if not valid:
            raise ValueError(msg)
        if free:
            return _pack_posdef_matrix(folded_val, diag_lb=self.__diag_lb)
        else:
            return folded_val.flatten()

    def fold(self, flat_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        if flat_val.size != self.flat_length(free):
            raise ValueError(
                'Wrong length for PSDSymmetricMatrix flat value.')
        if free:
            return _unpack_posdef_matrix(flat_val, diag_lb=self.__diag_lb)
        else:
            folded_val = np.reshape(flat_val, (self.__size, self.__size))
            valid, msg = self.validate_folded(folded_val, validate_value)
            if not valid:
                raise ValueError(msg)
            return folded_val

    def flat_indices(self, folded_bool, free=None):
        # If no indices are specified, save time and return an empty array.
        if not np.any(folded_bool):
            return np.array([], dtype=int)

        free = self._free_with_default(free)
        shape_ok, err_msg = self._validate_folded_shape(folded_bool)
        if not shape_ok:
            raise ValueError(err_msg)
        if not free:
            folded_indices = self.fold(
                np.arange(self.flat_length(False), dtype=int),
                validate_value=False, free=False)
            return folded_indices[folded_bool]
        else:
            # This indicates that each folded value depends on each
            # free value.  I think this is not true, but getting the exact
            # pattern may be complicated and will
            # probably not make much of a difference in practice.
            if np.any(folded_bool):
                return np.arange(self.flat_length(True), dtype=int)
            else:
                return np.array([])



register_pattern_json(PSDSymmetricMatrixPattern)
