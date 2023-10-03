from collections import OrderedDict
import itertools
import numbers
import json
from scipy.sparse import block_diag
import numpy as npy
import jax
import jax.numpy as np
from viabel.base_patterns import Pattern

####################
# JSON helpers.

# A dictionary of registered types for loading to and from JSON.
# This allows PatternDict and PatternArray read JSON containing arbitrary
# pattern types without executing user code.
__json_patterns = dict()
def register_pattern_json(pattern, allow_overwrite=False):
    """
    Register a pattern for automatic conversion from JSON.

    Parameters
    ------------
    pattern: A Pattern class
        The pattern to register.
    allow_overwrite: Boolean
        If true, allow overwriting already-registered patterns.

    Examples
    -------------
    >>> class MyCustomPattern(paragami.Pattern):
    >>>    ... definitions ...
    >>>
    >>> paragami.register_pattern_json(paragmi.MyCustomPattern)
    >>>
    >>> my_pattern = MyCustomPattern(...)
    >>> my_pattern_json = my_pattern.to_json()
    >>>
    >>> # ``my_pattern_from_json`` should be identical to ``my_pattern``.
    >>> my_pattern_from_json = paragami.get_pattern_from_json(my_pattern_json)
    """
    pattern_name = pattern.json_typename()
    if (not allow_overwrite) and pattern_name in __json_patterns.keys():
        raise ValueError(
            'A pattern named {} is already registered for JSON.'.format(
                pattern_name))
    __json_patterns[pattern_name] = pattern


def get_pattern_from_json(pattern_json):
    """
    Return the appropriate pattern from ``pattern_json``.

    The pattern must have been registered using ``register_pattern_json``.

    Parameters
    --------------
    pattern_json: String
        A JSON string as created with a pattern's ``to_json`` method.

    Returns
    -----------
    The pattern instance encoded in the ``pattern_json`` string.
    """
    pattern_json_dict = json.loads(pattern_json)
    try:
        json_pattern_name = pattern_json_dict['pattern']
    except KeyError as orig_err_string:
        err_string = \
            'A pattern JSON string must have an entry called pattern ' + \
            'which is registered using ``register_pattern_json``.'
        raise KeyError(err_string)

    if not json_pattern_name in __json_patterns.keys():
        err_string = (
            'Before converting from JSON, the pattern {} must be ' +
            'registered with ``register_pattern_json``.'.format(
                json_pattern_name))
        raise KeyError(err_string)
    return __json_patterns[json_pattern_name].from_json(pattern_json)


def save_folded(file, folded_val, pattern, **argk):
    """
    Save a folded value to a file with its pattern.

    Flatten a folded value and save it with its pattern to a file using
    ``numpy.savez``.  Additional keyword arguments will also be saved to the
    file.

    Parameters
    ---------------
    file: String or file
        Follows the conventions of ``numpy.savez``.  Note that the ``npz``
        extension will be added if it is not present.
    folded_val:
        The folded value of a parameter.
    pattern:
        A ``paragami`` pattern for the folded value.
    """
    flat_val = pattern.flatten(folded_val, free=False)
    pattern_json = pattern.to_json()
    np.savez(file, flat_val=flat_val, pattern_json=pattern_json, **argk)


def load_folded(file):
    """
    Load a folded value and its pattern from a file together with any
    additional data.

    Note that ``pattern`` must be registered with ``register_pattern_json``
    to use ``load_folded``.

    Parameters
    ---------------
    file: String or file
        A file or filename of data saved with ``save_folded``.

    Returns
    -----------
    folded_val:
        The folded value of the saved parameter.
    pattern:
        The ``paragami`` pattern of the saved parameter.
    data:
        The data as returned from ``np.load``.  Additional saved values will
        exist as keys of ``data``.
    """
    data = np.load(file)
    pattern = get_pattern_from_json(str(data['pattern_json']))
    folded_val = pattern.fold(data['flat_val'], free=False)
    return folded_val, pattern, data


##########################
# Dictionary of patterns.

class PatternDict(Pattern):
    """
    A dictionary of patterns (which is itself a pattern).

    Methods
    ------------
    lock:
        Prevent additional patterns from being added or removed.

    Examples
    ------------
    .. code-block:: python

        import paragami

        # Add some patterns.
        dict_pattern = paragami.PatternDict()
        dict_pattern['vec'] = paragami.NumericArrayPattern(shape=(2, ))
        dict_pattern['mat'] = paragami.PSDSymmetricMatrixPattern(size=3)

        # Dictionaries can also contain dictionaries (but they have to
        # be populated /before/ being added to the parent).
        sub_dict_pattern = paragami.PatternDict()
        sub_dict_pattern['vec1'] = paragami.NumericArrayPattern(shape=(2, ))
        sub_dict_pattern['vec2'] = paragami.NumericArrayPattern(shape=(2, ))
        dict_pattern['sub_dict'] = sub_dict_pattern

        # We're done adding patterns, so lock the dictionary.
        dict_pattern.lock()

        # Get a random intial value for the whole dictionary.
        dict_val = dict_pattern.random()
        print(dict_val['mat']) # Prints a 3x3 positive definite numpy matrix.

        # Get a flattened value of the whole dictionary.
        dict_val_flat = dict_pattern.flatten(dict_val, free=True)

        # Get a new random folded value of the dictionary.
        new_dict_val_flat = np.random.random(len(dict_val_flat))
        new_dict_val = dict_pattern.fold(new_dict_val_flat, free=True)
    """
    def __init__(self, free_default=None):
        self.__pattern_dict = OrderedDict()

        # __lock determines whether new elements can be added.
        self.__lock = False
        super().__init__(0, 0, free_default=free_default)

    def lock(self):
        self.__lock = True

    def __str__(self):
        pattern_strings = [
            '\t[' + key + '] = ' + str(self.__pattern_dict[key])
            for key in self.__pattern_dict]
        return \
            'OrderedDict:\n' + \
            '\n'.join(pattern_strings)

    def __getitem__(self, key):
        return self.__pattern_dict[key]

    def as_dict(self):
        # json.loads returns a dictionary, not an OrderedDict, so
        # save the keys in the current order.
        contents = {}
        for pattern_name, pattern in self.__pattern_dict.items():
            contents[pattern_name] = pattern.to_json()
        keys = [ key for key in self.__pattern_dict.keys() ]
        return {
            'pattern': self.json_typename(),
            'keys': keys,
            'contents': contents}

    def _check_lock(self):
        if self.__lock:
            raise ValueError(
                'The dictionary is locked, and its values cannot be changed.')

    def __setitem__(self, pattern_name, pattern):
        self._check_lock()
        # if pattern_name in self.__pattern_dict.keys():
        #     self.__delitem__(pattern_name)

        self.__pattern_dict[pattern_name] = pattern

        # We cannot allow pattern dictionaries to change their size
        # once they've been included as members in another dictionary,
        # since we have no way of updating the parent dictionary's size.
        # To avoid unexpected errors, lock any dictionary that is set as
        # a member.
        if type(self.__pattern_dict[pattern_name]) is PatternDict:
            self.__pattern_dict[pattern_name].lock()

        self._free_flat_length = self._update_flat_length(free=True)
        self._flat_length = self._update_flat_length(free=False)

    def __delitem__(self, pattern_name):
        self._check_lock()

        pattern = self.__pattern_dict[pattern_name]
        self.__pattern_dict.pop(pattern_name)

        self._free_flat_length = self._update_flat_length(free=True)
        self._flat_length = self._update_flat_length(free=False)

    def keys(self):
        return self.__pattern_dict.keys()

    def empty(self, valid):
        empty_val = OrderedDict()
        for pattern_name, pattern in self.__pattern_dict.items():
            empty_val[pattern_name] = pattern.empty(valid)
        return empty_val

    def validate_folded(self, folded_val, validate_value=None):
        for pattern_name, pattern in self.__pattern_dict.items():
            if not pattern_name in folded_val:
                return \
                    False, \
                    '{} not in folded_val dictionary.'.format(pattern_name)
            valid, err_msg = pattern.validate_folded(
                folded_val[pattern_name], validate_value=validate_value)
            if not valid:
                err_msg = '{} is not valid.'.format(err_msg)
                return False, err_msg
        return True, ''

    def fold(self, flat_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        flat_val = np.asarray(flat_val)
        flat_val = flat_val.ravel()
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        flat_length = self.flat_length(free)
        if flat_val.size != flat_length:
            error_string = \
                ('Wrong size for pattern dictionary {}.\n' +
                 'Expected {}, got {}.').format(
                    str(self), str(flat_length), str(flat_val.size))
            raise ValueError(error_string)

        # TODO: add an option to do this -- and other operations -- in place.
        folded_val = OrderedDict()
        offset = 0
        for pattern_name, pattern in self.__pattern_dict.items():
            pattern_flat_length = pattern.flat_length(free)
            pattern_flat_val = flat_val[offset:(offset + pattern_flat_length)]
            offset += pattern_flat_length
            # Containers must not mix free and non-free values, so do not
            # use default values for free.
            folded_val[pattern_name] = \
                pattern.fold(pattern_flat_val,
                             free=free,
                             validate_value=validate_value)
        if not free:
            valid, msg = self.validate_folded(
                folded_val, validate_value=validate_value)
            if not valid:
                raise ValueError(msg)
        return folded_val

    def flatten(self, folded_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        valid, msg = self.validate_folded(
            folded_val, validate_value=validate_value)
        if not valid:
            raise ValueError(msg)

        # flat_length = self.flat_length(free)
        # offset = 0
        # flat_val = np.full(flat_length, float('nan'))
        flat_vals = []
        for pattern_name, pattern in self.__pattern_dict.items():
            pattern_flat_length = pattern.flat_length(free)
            # Containers must not mix free and non-free values, so do not
            # use default values for free.
            # flat_val[offset:(offset + pattern_flat_length)] = \
            flat_vals.append(
                pattern.flatten(
                    folded_val[pattern_name],
                    free=free,
                    validate_value=validate_value))
            #offset += pattern_flat_length
        return np.hstack(flat_vals)

    def _update_flat_length(self, free):
        # This is a little wasteful with the benefit of being less error-prone
        # than adding and subtracting lengths as keys are changed.
        return np.sum(np.array([pattern.flat_length(free) for pattern_name, pattern in
                       self.__pattern_dict.items()]))

    def unfreeing_jacobian(self, folded_val, sparse=False):
        jacobians = []
        for pattern_name, pattern in self.__pattern_dict.items():
            jac = pattern.unfreeing_jacobian(
                folded_val[pattern_name], sparse=False)
            jacobians.append(jac)

        sp_jac = block_diag(jacobians, format='coo')

        if sparse:
            return sp_jac
        else:
            return np.array(sp_jac.todense())

    def freeing_jacobian(self, folded_val, sparse=False):
        jacobians = []
        for pattern_name, pattern in self.__pattern_dict.items():
            jac = pattern.freeing_jacobian(
                folded_val[pattern_name])
            jacobians.append(jac)

        sp_jac = block_diag(jacobians, format='coo')
        if sparse:
            return sp_jac
        else:
            return np.array(sp_jac.todense())

    def log_abs_det_unfreeing_jacobian(self, folded_val):
        log_abs_det = 0.0
        for pattern_name, pattern in self.__pattern_dict.items():
            log_abs_det += pattern.log_abs_det_unfreeing_jacobian(
                folded_val[pattern_name])
        return log_abs_det

    def log_abs_det_freeing_jacobian(self, folded_val):
        log_abs_det = 0.0
        for pattern_name, pattern in self.__pattern_dict.items():
            log_abs_det += pattern.log_abs_det_freeing_jacobian(
                folded_val[pattern_name])
        return log_abs_det

    @classmethod
    def from_json(cls, json_string):
        json_dict = json.loads(json_string)
        if json_dict['pattern'] != cls.json_typename():
            error_string = \
                ('{}.from_json must be called on a json_string made ' +
                 'from a the same pattern type.  The json_string ' +
                 'pattern type was {}.').format(
                    cls.json_typename(), json_dict['pattern'])
            raise ValueError(error_string)
        pattern_dict = cls()
        for pattern_name in json_dict['keys']:
            pattern_dict[pattern_name] = get_pattern_from_json(
                json_dict['contents'][pattern_name])
        return pattern_dict

    def flat_indices(self, folded_bool, free=None):
        free = self._free_with_default(free)
        valid, msg = self.validate_folded(folded_bool, validate_value=False)
        if not valid:
            raise ValueError(msg)

        flat_length = self.flat_length(free)
        offset = 0
        indices = []
        for pattern_name, pattern in self.__pattern_dict.items():
            pattern_flat_length = pattern.flat_length(free)
            # Containers must not mix free and non-free values, so do not
            # use default values for free.
            pattern_indices = pattern.flat_indices(
                folded_bool[pattern_name], free=free)
            if len(pattern_indices) > 0:
                indices.append(pattern_indices + offset)
            offset += pattern_flat_length
        if len(indices) > 0:
            return np.hstack(indices)
        else:
            return np.array([], dtype=int)

    def flat_names(self, free, delim='_'):
        flat_names_list = []
        for pattern_name, pattern in self.__pattern_dict.items():
            pattern_flat_names = pattern.flat_names(free)
            # TODO: only append the delimiter for containers
            pattern_flat_names = \
                [ pattern_name + delim + t for t in pattern_flat_names]
            flat_names_list.append(pattern_flat_names)
        return np.hstack(flat_names_list)



##########################
# An array of a pattern.

class PatternArray(Pattern):
    """
    An array of a pattern (which is also itself a pattern).

    The first indices of the folded pattern are the array and the final
    indices are of the base pattern.  For example, if `shape=(3, 4)`
    and `base_pattern = PSDSymmetricMatrixPattern(size=5)`, then the folded
    value of the array will have shape `(3, 4, 5, 5)`, where the entry
    `folded_val[i, j, :, :]` is a 5x5 positive definite matrix.

    Currently this can only contain patterns whose folded values are
    numeric arrays (i.e., `NumericArrayPattern`, `SimplexArrayPattern`, and
    `PSDSymmetricMatrixPattern`).
    """
    def __init__(self, array_shape, base_pattern, free_default=None):
        """
        Parameters
        ------------
        array_shape: tuple of int
            The shape of the array (not including the base parameter)
        base_pattern:
            The base pattern.
        """
        # TODO: change the name shape -> array_shape
        # and have shape be the whole array, including the pattern.
        self.__array_shape = tuple(array_shape)
        self.__array_ranges = [range(0, t) for t in self.__array_shape]

        num_elements = np.prod(np.array(self.__array_shape))
        self.__base_pattern = base_pattern

        empty_pattern = self.__base_pattern.empty(valid=False)

        if type(empty_pattern) is npy.ndarray or np.ndarray:
            self.__folded_pattern_shape = empty_pattern.shape
        else:
            raise NotImplementedError(
                'PatternArray does not support patterns whose folded ' +
                'values are not numpy.ndarray types.')
        # Check whether the base_pattern takes values that are numpy arrays.
        # If they are, then the unfolded value will be a single numpy array
        # of shape __array_shape + base_pattern.empty().shape.

        self.__shape = tuple(self.__array_shape) + empty_pattern.shape

        super().__init__(
            num_elements * base_pattern.flat_length(free=False),
            num_elements * base_pattern.flat_length(free=True),
            free_default=free_default)

    def __str__(self):
        return('PatternArray {} of {}'.format(
            self.__array_shape, self.__base_pattern))

    def as_dict(self):
        return {
            'pattern': self.json_typename(),
            'shape': self.__shape,
            'array_shape': self.__array_shape,
            'base_pattern': self.__base_pattern.to_json() }

    def array_shape(self):
        """The shape of the array of parameters.

        This does not include the dimension of the folded parameters.
        """
        return self.__array_shape

    def shape(self):
        """The shape of a folded value.
        """
        return self.__shape

    def base_pattern(self):
        return self.__base_pattern

    def validate_folded(self, folded_val, validate_value=None):
        if folded_val.ndim != len(self.__shape):
            return \
                False, \
                'Wrong number of dimensions.  Expected {}, got {}.'.format(
                    folded_val.ndim, len(self.__shape))
        if folded_val.shape != self.__shape:
            return \
                False, \
                'Wrong shape.  Expected {}, got {}.'.format(
                    folded_val.shape, self.__shape)
        for item in itertools.product(*self.__array_ranges):
            valid, msg = self.__base_pattern.validate_folded(
                folded_val[item], validate_value=validate_value)
            if not valid:
                err_msg = 'Bad value in location {}: {}'.format(item, msg)
                return False, err_msg
        return True, ''

    def empty(self, valid):
        empty_pattern = self.__base_pattern.empty(valid=valid)
        repeated_array = np.asarray(
            [empty_pattern
             for item in itertools.product(*self.__array_ranges)])
        return np.reshape(repeated_array, self.__shape)

    def _stacked_obs_slice(self, item, flat_length):
        """
        Get the slice in a flat array corresponding to ``item``.

        Parameters
        -------------
        item: tuple
            A tuple of indices into the array of patterns (i.e.,
            into the shape ``__array_shape``).
        flat_length: integer
            The length of a single flat pattern.

        Returns
        ---------------
        A slice for the elements in a vector of length ``flat_length``
        corresponding to element item of the array, where ``item`` is a tuple
        indexing into the array of shape ``__array_shape``.
        """
        assert len(item) == len(self.__array_shape)
        linear_item = np.ravel_multi_index(item, self.__array_shape) * flat_length
        return slice(linear_item, linear_item + flat_length)

    def fold(self, flat_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        if isinstance(flat_val, npy.ndarray) or isinstance(flat_val, numbers.Number):
            flat_val = np.atleast_1d(flat_val)
        elif isinstance(flat_val, np.ndarray):
            flat_val = jax.device_get(flat_val)
        else:
            primal_flat_val = flat_val.primal
            flat_val = jax.device_get(primal_flat_val)
            flat_val = np.atleast_1d(flat_val)

        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        if flat_val.size != self.flat_length(free):
            error_string = \
                'Wrong size for parameter.  Expected {}, got {}'.format(
                    str(self.flat_length(free)), str(flat_val.size))
            raise ValueError(error_string)

        flat_length = self.__base_pattern.flat_length(free)
        folded_array = np.array([
            self.__base_pattern.fold(
                flat_val[self._stacked_obs_slice(item, flat_length)],
                free=free, validate_value=validate_value)
            for item in itertools.product(*self.__array_ranges)])

        folded_val = np.reshape(folded_array, self.__shape)

        if not free:
            valid, msg = self.validate_folded(
                folded_val, validate_value=validate_value)
            if not valid:
                raise ValueError(msg)
        return folded_val

    def flatten(self, folded_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        valid, msg = self.validate_folded(
            folded_val, validate_value=validate_value)
        if not valid:
            raise ValueError(msg)

        return np.hstack(np.array([
            self.__base_pattern.flatten(
                folded_val[item], free=free, validate_value=validate_value)
            for item in itertools.product(*self.__array_ranges)]))

    def flat_length(self, free=None):
        free = self._free_with_default(free)
        return self._free_flat_length if free else self._flat_length

    def unfreeing_jacobian(self, folded_val, sparse=False):
        base_flat_length = self.__base_pattern.flat_length(free=True)
        base_freeflat_length = self.__base_pattern.flat_length(free=True)

        jacobians = []
        for item in itertools.product(*self.__array_ranges):
            jac = self.__base_pattern.unfreeing_jacobian(
                folded_val[item], sparse=False)
            jacobians.append(jac)
        sp_jac = block_diag(jacobians, format='coo')

        if sparse:
            return sp_jac
        else:
            return np.array(sp_jac.todense())

    def freeing_jacobian(self, folded_val, sparse=False):
        base_flat_length = self.__base_pattern.flat_length(free=True)
        base_freeflat_length = self.__base_pattern.flat_length(free=True)

        jacobians = []
        for item in itertools.product(*self.__array_ranges):
            jac = self.__base_pattern.freeing_jacobian(
                folded_val[item])
            jacobians.append(jac)
        sp_jac = block_diag(jacobians, format='coo')

        if sparse:
            return sp_jac
        else:
            return np.array(sp_jac.todense())

    @classmethod
    def from_json(cls, json_string):
        json_dict = json.loads(json_string)
        if json_dict['pattern'] != cls.json_typename():
            error_string = \
                ('{}.from_json must be called on a json_string made ' +
                 'from a the same pattern type.  The json_string ' +
                 'pattern type was {}.').format(
                    cls.json_typename(), json_dict['pattern'])
            raise ValueError(error_string)
        base_pattern = get_pattern_from_json(json_dict['base_pattern'])
        return cls(
            array_shape=json_dict['array_shape'], base_pattern=base_pattern)

    def flat_indices(self, folded_bool, free=None):
        free = self._free_with_default(free)
        valid, msg = self.validate_folded(folded_bool, validate_value=False)
        if not valid:
            raise ValueError(msg)

        indices = []
        pattern_flat_length = self.__base_pattern.flat_length(free=free)
        offset = 0
        for item in itertools.product(*self.__array_ranges):
            if np.any(folded_bool[item]):
                pattern_indices = self.__base_pattern.flat_indices(
                    folded_bool[item], free=free)
                if len(pattern_indices) > 0:
                    indices.append(pattern_indices + offset)
            offset += pattern_flat_length
        if len(indices) > 0:
            return np.hstack(indices)
        else:
            return np.array([], dtype=int)


register_pattern_json(PatternDict)
register_pattern_json(PatternArray)
