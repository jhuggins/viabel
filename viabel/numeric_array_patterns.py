from viabel.base_patterns import Pattern
from viabel.pattern_containers import register_pattern_json
import jax.numpy as np
import copy
import itertools
import json
import scipy as osp
from scipy import sparse


def _unconstrain_array(array, lb, ub):
    # Assume that the inputs obey the constraints, lb < ub and
    # lb <= array <= ub, which are checked in the pattern.
    if ub == float("inf"):
        if lb == -float("inf"):
            # For consistent behavior, never return a reference.
            return copy.copy(array)
        else:
            return np.log(array - lb)
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return -1 * np.log(ub - array)
        else:
            return np.log(array - lb) - np.log(ub - array)


def _unconstrain_array_jacobian(array, lb, ub):
    # The Jacobian of the unconstraining mapping in the same shape as
    # the original array.
    if ub == float("inf"):
        if lb == -float("inf"):
            return np.ones_like(array)
        else:
            return 1.0 / (array - lb)
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return 1.0 / (ub - array)
        else:
            return 1 / (array - lb) + 1 / (ub - array)


def _constrain_array(free_array, lb, ub):
    # Assume that lb < ub, which is checked in the pattern.
    if ub == float("inf"):
        if lb == -float("inf"):
            # For consistency, never return a reference.
            #return copy.deepcopy(free_array)
            #return free_array
            return copy.copy(free_array)
        else:
            return np.exp(free_array) + lb
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return ub - np.exp(-1 * free_array)
        else:
            exp_vec = np.exp(free_array)
            return (ub - lb) * exp_vec / (1 + exp_vec) + lb


def _constrain_array_jacobian(free_array, lb, ub):
    # The Jacobian of the constraining mapping in the same shape as the
    # original array.
    if ub == float("inf"):
        if lb == -float("inf"):
            return np.ones_like(free_array)
        else:
            return np.exp(free_array)
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return np.exp(-1 * free_array)
        else:
            # d/dx exp(x) / (1 + exp(x)) =
            #    exp(x) / (1 + exp(x)) - exp(x) ** 2 / (1 + exp(x)) ** 2
            exp_vec = np.exp(free_array)
            ratio = exp_vec / (1 + exp_vec)
            return (ub - lb) * ratio * (1 - ratio)


def _get_inbounds_value(lb, ub):
    assert lb < ub
    if lb > -float('inf') and ub < float('inf'):
        return 0.5 * (ub - lb) + lb
    else:
        if lb > -float('inf'):
            # The upper bound is infinite.
            return lb + 1.0
        elif ub < float('inf'):
            # The lower bound is infinite.
            return ub - 1.0
        else:
            # Both are infinite.
            return 0.0


def _constrain_array(free_array, lb, ub):
    # Assume that lb < ub, which is checked in the pattern.
    if ub == float("inf"):
        if lb == -float("inf"):
            # For consistency, never return a reference.
            #return copy.deepcopy(free_array)
            #return free_array
            return copy.copy(free_array)
        else:
            return np.exp(free_array) + lb
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return ub - np.exp(-1 * free_array)
        else:
            exp_vec = np.exp(free_array)
            return (ub - lb) * exp_vec / (1 + exp_vec) + lb


class NumericArrayPattern(Pattern):
    """
    A pattern for (optionally bounded) arrays of numbers.

    Attributes
    -------------
    default_validate: `bool`, optional
        Whether or not the array is checked by default to lie within the
        specified bounds.
    """
    def __init__(self, shape,
                 lb=-float("inf"), ub=float("inf"),
                 default_validate=True, free_default=None):
        """
        Parameters
        -------------
        shape: `tuple` of `int`
            The shape of the array.
        lb: `float`
            The (inclusive) lower bound for the entries of the array.
        ub: `float`
            The (inclusive) upper bound for the entries of the array.
        default_validate: `bool`, optional
            Whether or not the array is checked by default to lie within the
            specified bounds.
        free_default: `bool`, optional
            Whether the pattern is free by default.
        """
        self.default_validate = default_validate
        self._shape = shape
        self._lb = lb

        self._ub = ub
        assert lb >= -float('inf')
        assert ub <= float('inf')
        if lb >= ub:
            raise ValueError(
                'Upper bound ub must strictly exceed lower bound lb')

        free_flat_length = flat_length = int(np.prod(np.asarray(self._shape)))

        super().__init__(flat_length, free_flat_length,
                         free_default=free_default)

        # Cache arrays of indices for flat_indices
        # TODO: not sure this is a good idea or much of a speedup.
        self.__free_folded_indices = self.fold(
            np.arange(self.flat_length(free=True), dtype=int),
            validate_value=False, free=False)

        self.__nonfree_folded_indices = self.fold(
            np.arange(self.flat_length(free=False), dtype=int),
            validate_value=False, free=False)

    def __str__(self):
        return 'NumericArrayPattern {} (lb={}, ub={})'.format(
            self._shape, self._lb, self._ub)

    def as_dict(self):
        return {
            'pattern': self.json_typename(),
            'lb': self._lb,
            'ub': self._ub,
            'shape': self._shape,
            'default_validate': self.default_validate,
            'free_default': self.free_default }

    def empty(self, valid):
        if valid:
            return np.full(
                self._shape, _get_inbounds_value(self._lb, self._ub))
        else:
            return np.empty(self._shape)

    def _validate_folded_shape(self, folded_val):
        if folded_val.shape != tuple(self.shape()):
            err_msg = ('Wrong size for array.' +
                       ' Expected shape: ' + str(self.shape()) +
                       ' Got shape: ' + str(folded_val.shape))
            return False, err_msg
        else:
            return True, ''

    def validate_folded(self, folded_val, validate_value=None):
        folded_val = np.atleast_1d(folded_val)
        shape_ok, err_msg = self._validate_folded_shape(folded_val)
        if not shape_ok:
            return shape_ok, err_msg
        if validate_value is None:
            validate_value = self.default_validate
        if validate_value:
            if (np.array(folded_val < self._lb)).any():
                return False, 'Value beneath lower bound.'
            if (np.array(folded_val > self._ub)).any():
                return False, 'Value above upper bound.'
        return True, ''

    def fold(self, flat_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        flat_val = np.atleast_1d(flat_val)

        if flat_val.ndim != 1:
            raise ValueError('The argument to fold must be a 1d vector.')

        expected_length = self.flat_length(free=free)
        if flat_val.size != expected_length:
            error_string = \
                'Wrong size for array.  Expected {}, got {}'.format(
                    str(expected_length),
                    str(flat_val.size))

            raise ValueError(error_string)


        if free:
            constrained_array = \
                _constrain_array(flat_val, self._lb, self._ub)
            return constrained_array.reshape(self._shape)
        else:
            folded_val = flat_val.reshape(self._shape)
            valid, msg = self.validate_folded(folded_val, validate_value)
            if not valid:
                raise ValueError(msg)
            return folded_val

    def flatten(self, folded_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        folded_val = np.atleast_1d(folded_val)
        valid, msg = self.validate_folded(folded_val, validate_value)
        if not valid:
            raise ValueError(msg)
        if free:
            return \
                _unconstrain_array(folded_val, self._lb, self._ub).flatten()
        else:
            return folded_val.flatten()

    def shape(self):
        return self._shape

    def bounds(self):
        return self._lb, self._ub

    def flat_length(self, free=None):
        free = self._free_with_default(free)
        if free:
            return self._free_flat_length
        else:
            return self._flat_length

    def flat_indices(self, folded_bool, free=None):
        # If no indices are specified, save time and return an empty array.
        if not np.any(folded_bool):
            return np.array([], dtype=int)

        free = self._free_with_default(free)
        folded_bool = np.atleast_1d(folded_bool)
        shape_ok, err_msg = self._validate_folded_shape(folded_bool)
        if not shape_ok:
            raise ValueError(err_msg)
        if free:
            return self.__free_folded_indices[folded_bool]
        else:
            return self.__nonfree_folded_indices[folded_bool]

    def freeing_jacobian(self, folded_val, sparse=False):
        jac_array = \
            _unconstrain_array_jacobian(folded_val, self._lb, self._ub)
        jac_array = np.atleast_1d(jac_array).flatten()
        if sparse:
            return osp.sparse.diags(jac_array)
        else:
            return np.diag(jac_array)

    def unfreeing_jacobian(self, folded_val, sparse=False):
        jac_array = \
            _constrain_array_jacobian(
                _unconstrain_array(folded_val, self._lb, self._ub),
                self._lb, self._ub)
        jac_array = np.atleast_1d(jac_array).flatten()
        if sparse:
            return osp.sparse.diags(jac_array)
        else:
            return np.diag(jac_array)

    def log_abs_det_freeing_jacobian(self, folded_val):
        jac_array = \
            _unconstrain_array_jacobian(folded_val, self._lb, self._ub)
        return np.sum(np.log(np.abs(jac_array)))

    def log_abs_det_unfreeing_jacobian(self, folded_val):
        jac_array = \
            _constrain_array_jacobian(
                _unconstrain_array(folded_val, self._lb, self._ub),
                self._lb, self._ub)
        return np.sum(np.log(np.abs(jac_array)))

    def flat_names(self, free):
        # Free is ignored for numeric arrays.
        array_ranges = [range(0, t) for t in self._shape]
        flat_name_list = []
        for item in itertools.product(*array_ranges):
            flat_name_list.append('[' + ','.join([str(i) for i in item]) + ']')
        return flat_name_list


class NumericVectorPattern(NumericArrayPattern):
    """A pattern for a (optionally bounded) numeric vector.

    See Also
    ------------
    NumericArrayPattern
    """
    def __init__(self, length, lb=-float("inf"), ub=float("inf"),
                 default_validate=True, free_default=None):
        super().__init__(shape=(length, ), lb=lb, ub=ub,
                         default_validate=default_validate,
                         free_default=free_default)

    def length(self):
        return self._shape[0]

    def as_dict(self):
        return {
            'pattern': self.json_typename(),
            'length': self.length(),
            'lb': self._lb,
            'ub': self._ub,
            'default_validate': self.default_validate,
            'free_default': self.free_default }


class NumericScalarPattern(NumericArrayPattern):
    """A pattern for a (optionally bounded) numeric scalar.

    See Also
    ------------
    NumericArrayPattern
    """
    def __init__(self, lb=-float("inf"), ub=float("inf"),
                 default_validate=True, free_default=None):
        super().__init__(shape=(1, ), lb=lb, ub=ub,
                         default_validate=default_validate,
                         free_default=free_default)

    def as_dict(self):
        return {
            'pattern': self.json_typename(),
            'lb': self._lb,
            'ub': self._ub,
            'default_validate': self.default_validate,
            'free_default': self.free_default}


register_pattern_json(NumericVectorPattern)
register_pattern_json(NumericScalarPattern)
register_pattern_json(NumericArrayPattern)
