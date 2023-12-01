# This file contains code that is derived from paragami (https://github.com/rgiordan/paragami).
# paragami is originally licensed under the Apache-2.0 license.


from abc import ABC, abstractmethod
import jax
import json
import numpy as np
from scipy.sparse import coo_matrix, block_diag
import jax.numpy as jnp
import copy
import itertools
import scipy as osp
from scipy import sparse
from collections import OrderedDict
import numbers
import math
from jax import custom_vjp, custom_jvp, device_get
from jax.scipy.special import logsumexp
import warnings


class Pattern(ABC):
    """A abstract class for a parameter pattern.

    See derived classes for examples.
    """
    def __init__(self, flat_length, free_flat_length, free_default=None):
        """
        Parameters
        -----------
        flat_length : `int`
            The length of a non-free flattened vector.
        free_flat_length : `int`
            The length of a free flattened vector.
        """
        self._flat_length = flat_length
        self._free_flat_length = free_flat_length

        # In practice you'll probably want to implement custom versions
        # of these Jacboians.
        self._freeing_jacobian = jax.jacrev(self._freeing_transform, allow_int = True)
        self._unfreeing_jacobian = jax.jacrev(self._unfreeing_transform, allow_int = True)

        self.free_default = free_default

    # Abstract methods that must be implemented by subclasses.

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def as_dict(self):
        """Return a dictionary of attributes describing the pattern.

        The dictionary should completely describe the pattern in the sense
        that if the contents of two patterns' dictionaries are identical
        the patterns should be considered identical.

        If the keys of the returned dictionary match the arguments to
        ``__init__``, then the default methods for ``to_json`` and
        ``from_json`` will work with no additional modification.
        """
        pass

    @abstractmethod
    def fold(self, flat_val, free=None, validate_value=None):
        """Fold a flat value into a parameter.

        Parameters
        -----------
        flat_val : `numpy.ndarray`, (N, )
            The flattened value.
        free : `bool`, optional.
            Whether or not the flattened value is a free parameterization.
            If not specified, the attribute ``free_default`` is used.
        validate_value : `bool`, optional.
            Whether to check that the folded value is valid.  If ``None``,
            the pattern will employ a default behavior.

        Returns
        ---------
        folded_val : Folded value
            The parameter value in its original folded shape.
        """
        pass

    @abstractmethod
    def flatten(self, folded_val, free=None, validate_value=None):
        """Flatten a folded value into a flat vector.

        Parameters
        -----------
        folded_val : Folded value
            The parameter in its original folded shape.
        free : `bool`, optional
            Whether or not the flattened value is to be in a free
            parameterization.  If not specified, the attribute
            ``free_default`` is used.
        validate_value : `bool`
            Whether to check that the folded value is valid.  If ``None``,
            the pattern will employ a default behavior.

        Returns
        ---------
        flat_val : ``numpy.ndarray``, (N, )
            The flattened value.
        """
        pass

    @abstractmethod
    def empty(self, valid):
        """Return an empty parameter in its folded shape.

        Parameters
        -------------
        valid : `bool`
            Whether or folded shape should be filled with valid values.

        Returns
        ---------
        folded_val : Folded value
            A parameter value in its original folded shape.
        """
        pass

    @abstractmethod
    def validate_folded(self, folded_val, validate_value=None):
        """Check whether a folded value is valid.

        Parameters
        ----------------
        folded_val : Folded value
            A parameter value in its original folded shape.
        validate_value : `bool`
            Whether to validate the value in addition to the shape.  The
            shape is always validated.

        Returns
        ------------
        is_valid : `bool`
            Whether ``folded_val`` is an allowable shape and value.
        err_msg : `str`
        """
        pass

    @abstractmethod
    def flat_indices(self, folded_bool, free=None):
        """Get which flattened indices correspond to which folded values.

        Parameters
        ------------
        folded_bool : Folded booleans
            A variable in the folded shape but containing booleans.  The
            elements that are ``True`` are the ones for which we will return
            the flat indices.
        free : `bool`
            Whether or not the flattened value is to be in a free
            parameterization.  If not specified, the attribute
            ``free_default`` is used.

        Returns
        --------
        indices : `numpy.ndarray` (N,)
            A list of indices into the flattened value corresponding to
            the ``True`` members of ``folded_bool``.
        """
        pass


    ##################################################
    # Methods that are standard for all patterns.

    def _free_with_default(self, free):
        """Check whether to use ``free_default`` and return the appropriate
        boolean.
        """
        if free is not None:
            return free
        else:
            if self.free_default is None:
                raise ValueError(
                    ('If ``free_default`` is ``None``, ``free`` ' +
                    'must be specified.'))
            else:
                return self.free_default

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.as_dict() == other.as_dict()

    @classmethod
    def json_typename(cls):
        return '.'.join([ cls.__module__, cls.__name__])

    def _freeing_transform(self, flat_val):
        """From the flat to the free flat value.
        """
        return self.flatten(self.fold(flat_val, free=False), free=True)

    def _unfreeing_transform(self, free_flat_val):
        """From the free flat to the flat value.
        """
        return self.flatten(self.fold(free_flat_val, free=True), free=False)

    def flat_length(self, free=None):
        """Return the length of the pattern's flattened value.

        Parameters
        -----------
        free : `bool`, optional
            Whether or not the flattened value is to be in a free
            parameterization.  If not specified, ``free_default`` is used.

        Returns
        ---------
        length : `int`
            The length of the pattern's flattened value.
        """
        free = self._free_with_default(free)
        if free:
            return self._free_flat_length
        else:
            return self._flat_length

    def random(self):
        """Return an random, valid parameter in its folded shape.

        .. note::
            There is no reason this provides a meaningful distribution over
            folded values.  This function is intended to be used as
            a convenience for testing.

        Returns
        ---------
        folded_val : Folded value
            A random parameter value in its original folded shape.
        """
        return self.fold(np.random.random(self._free_flat_length), free=True)

    def empty_bool(self, value):
        """Return folded shape containing booleans.

        Parameters
        -------------
        value : `bool`
            The value with which to fill the folded shape.

        Returns
        ---------
        folded_bool : Folded value
            A boolean value in its original folded shape.
        """
        flat_len = self.flat_length(free=False)
        bool_vec = np.full(flat_len, value, dtype='bool')
        return self.fold(bool_vec, free=False, validate_value=False)

    def freeing_jacobian(self, folded_val):
        """The Jacobian of the map from a flat free value to a flat value.

        If the folded value of the parameter is ``val``, ``val_flat =
        flatten(val, free=False)``, and ``val_freeflat = flatten(val,
        free=True)``, then this calculates the Jacobian matrix ``d val_free / d
        val_freeflat``.  For entries with no dependence between them, the
        Jacobian is taken to be zero.

        Parameters
        -------------
        folded_val : Folded value
            The folded value at which the Jacobian is to be evaluated.

        Returns
        -------------
        ``numpy.ndarray``, (N, M)
            The Jacobian matrix ``d val_free / d val_freeflat``. Consistent with
            standard Jacobian notation, the elements of ``val_free`` correspond
            to the rows of the Jacobian matrix and the elements of
            ``val_freeflat`` correspond to the columns.

        See also
        ------------
        Pattern.unfreeing_jacobian
        """
        flat_val = self.flatten(folded_val, free=False)
        jac = self._freeing_jacobian(flat_val)
        return jac

    def unfreeing_jacobian(self, folded_val, sparse=False):
        """The Jacobian of the map from a flat value to a flat free value.

        If the folded value of the parameter is ``val``, ``val_flat =
        flatten(val, free=False)``, and ``val_freeflat = flatten(val,
        free=True)``, then this calculates the Jacobian matrix ``d val_freeflat /
        d val_free``.  For entries with no dependence between them, the Jacobian
        is taken to be zero.

        Parameters
        -------------
        folded_val : Folded value
            The folded value at which the Jacobian is to be evaluated.
        sparse : `bool`, optional
            If ``True``, return a sparse matrix.  Otherwise, return a dense
            ``numpy`` 2d array.

        Returns
        -------------
        ``numpy.ndarray``, (N, N)
            The Jacobian matrix ``d val_freeflat / d val_free``. Consistent with
            standard Jacobian notation, the elements of ``val_freeflat``
            correspond to the rows of the Jacobian matrix and the elements of
            ``val_free`` correspond to the columns.

        See also
        ------------
        Pattern.freeing_jacobian
        """
        freeflat_val = self.flatten(folded_val, free=True)
        jac = self._unfreeing_jacobian(freeflat_val)
        if sparse:
            return coo_matrix(jac)
        else:
            return jac

    def log_abs_det_freeing_jacobian(self, folded_val):
        """Return the log absolute determinant of the freeing Jacobian.

        See ``freeing_jacobian`` for more details.  Because the output is not
        in the form of a matrix, this function should be both efficient and
        differentiable.  If the dimension of the free and unfree parameters
        are different, the extra dimensions are ignored.

        Parameters
        -------------
        folded_val : Folded value
            The folded value at which the Jacobian is to be evaluated.

        Returns
        -------------
        log_abs_det_jac : `float`
            The log absolute determinant of the freeing Jacobian.

        See also
        ------------
        Pattern.freeing_jacobian
        """
        raise NotImplementedError('Still thinking about the default.')

    def log_abs_det_unfreeing_jacobian(self, folded_val):
        """Return the log absolute determinant of the unfreeing Jacobian.

        See ``unfreeing_jacobian`` for more details.  Because the output is not
        in the form of a matrix, this function should be both efficient and
        differentiable.  If the dimension of the free and unfree parameters
        are different, the extra dimensions are ignored.

        Parameters
        -------------
        folded_val : Folded value
            The folded value at which the Jacobian is to be evaluated.

        Returns
        -------------
        log_abs_det_jac : `float`
            The log absolute determinant of the unfreeing Jacobian.

        See also
        ------------
        Pattern.unfreeing_jacobian
        """
        raise NotImplementedError('Still thinking about the default.')

    def to_json(self):
        """Return a JSON representation of the pattern.

        See also
        ------------
        Pattern.from_json
        """
        return json.dumps(self.as_dict())

    def flat_names(self, free):
        """Return a tidy named vector for the flat values.
        """
        return [ str(i) for i in range(self.flat_length(free)) ]

    @classmethod
    def _validate_json_dict_type(cls, json_dict):
        if json_dict['pattern'] != cls.json_typename():
            error_string = \
                ('{}.from_json must be called on a json_string made ' +
                 'from a the same pattern type.  The json_string ' +
                 'pattern type was {}.').format(
                    cls.json_typename(), json_dict['pattern'])
            raise ValueError(error_string)

    @classmethod
    def from_json(cls, json_string):
        """Return a pattern from ``json_string`` created by ``to_json``.

        See also
        ------------
        Pattern.to_json
        """
        json_dict = json.loads(json_string)
        cls._validate_json_dict_type(json_dict)
        del json_dict['pattern']
        return cls(**json_dict)

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
        flat_val = jnp.asarray(flat_val)
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

        if type(empty_pattern) is np.ndarray or jnp.ndarray:
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
        if isinstance(flat_val, np.ndarray) or isinstance(flat_val, numbers.Number):
            flat_val = np.atleast_1d(flat_val)
        elif isinstance(flat_val, jnp.ndarray):
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
        folded_array = jnp.array([
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

        return jnp.hstack(jnp.array([
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


def _unconstrain_array(array, lb, ub):
    # Assume that the inputs obey the constraints, lb < ub and
    # lb <= array <= ub, which are checked in the pattern.
    if ub == float("inf"):
        if lb == -float("inf"):
            # For consistent behavior, never return a reference.
            return copy.copy(array)
        else:
            return jnp.log(array - lb)
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return -1 * jnp.log(ub - array)
        else:
            return jnp.log(array - lb) - jnp.log(ub - array)


def _unconstrain_array_jacobian(array, lb, ub):
    # The Jacobian of the unconstraining mapping in the same shape as
    # the original array.
    if ub == float("inf"):
        if lb == -float("inf"):
            return jnp.ones_like(array)
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
            return jnp.exp(free_array) + lb
    else:  # the upper bound is finite
        if lb == -float("inf"):
            return ub - jnp.exp(-1 * free_array)
        else:
            exp_vec = jnp.exp(free_array)
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
        folded_val = jnp.atleast_1d(folded_val)
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
        flat_val = jnp.atleast_1d(flat_val)

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
        folded_val = jnp.atleast_1d(folded_val)
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
    nrow, ncol = jnp.shape(mat)
    if nrow != ncol:
        raise ValueError('mat must be square')
    return mat[jnp.tril_indices(nrow)]


@custom_vjp
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
    mat = jnp.zeros((mat_size, mat_size))
    for k1 in range(mat_size):
        for k2 in range(k1 + 1):
            idx = _sym_index(k1, k2)
            mat = mat.at[k1, k2].set(vec[idx])
    return mat

def _unvectorize_ld_matrix_fwd(vec):
    return  _unvectorize_ld_matrix(vec), vec

def _unvectorize_ld_matrix_bwd(vec, g):
    return (_vectorize_ld_matrix(g),)



_unvectorize_ld_matrix.defvjp(_unvectorize_ld_matrix_fwd, _unvectorize_ld_matrix_bwd)


def make_diagonal(mat):
    diag_elements = jnp.diag(mat)
    diagonal_matrix = jnp.diag(diag_elements)
    return diagonal_matrix

def _exp_matrix_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]
    mat_exp_diag = make_diagonal(
        jnp.exp(mat))
    mat_diag = make_diagonal(mat)
    return mat_exp_diag + mat - mat_diag
A = jnp.array([[4, 2], [2, 3]])


def _log_matrix_diagonal(mat):
    assert mat.shape[0] == mat.shape[1]
    mat_log_diag = make_diagonal(
        jnp.log(mat))
    mat_diag = make_diagonal(mat)
    return mat_log_diag + mat - mat_diag

def _pack_posdef_matrix(mat, diag_lb=0.0):
    k = mat.shape[0]
    mat_lb = mat - jnp.diag(jnp.full(k, diag_lb))
    return _vectorize_ld_matrix(
        _log_matrix_diagonal(jnp.linalg.cholesky(mat_lb)))

def _unpack_posdef_matrix(free_vec, diag_lb=0.0):
    lower_triangular = _unvectorize_ld_matrix(free_vec)
    exp_diag = jnp.exp(jnp.diag(lower_triangular))
    lower_triangular = lower_triangular - make_diagonal(lower_triangular) + make_diagonal(exp_diag)
    mat = jnp.matmul(lower_triangular, lower_triangular.T)
    k = mat.shape[0]
    return mat + make_diagonal(jnp.full(k, diag_lb))

def _unpack_posdef_matrix(free_vec, diag_lb=0.0):
    mat_chol = _exp_matrix_diagonal(_unvectorize_ld_matrix(free_vec))
    mat = jnp.matmul(mat_chol, mat_chol.T)
    k = mat.shape[0]
    return mat + jnp.diag(jnp.full(k, diag_lb))



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
            return jnp.eye(self.__size) * (self.__diag_lb + 1)
        else:
            return jnp.empty((self.__size, self.__size))

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
            if jnp.any(jnp.diag(folded_val) < self.__diag_lb):
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
        if isinstance(flat_val, np.ndarray) or isinstance(flat_val, numbers.Number):
            flat_val = np.atleast_1d(flat_val)
        elif isinstance(flat_val, jnp.ndarray):
            flat_val = device_get(flat_val)
        else:
            primal_flat_val = flat_val.primal
            flat_val = device_get(primal_flat_val)
            flat_val = np.atleast_1d(flat_val)
        if len(flat_val.shape) != 1:
            raise ValueError('The argument to fold must be a 1d vector.')
        if flat_val.size != self.flat_length(free):

            raise ValueError(
                'Wrong length for PSDSymmetricMatrix flat value.')
        if free:
            return _unpack_posdef_matrix(flat_val, diag_lb=self.__diag_lb)
        else:
            folded_val = jnp.reshape(flat_val, (self.__size, self.__size))
            valid, msg = self.validate_folded(folded_val, validate_value)
            if not valid:
                raise ValueError(msg)
            return folded_val

    def flat_indices(self, folded_bool, free=None):
        # If no indices are specified, save time and return an empty array.
        if not jnp.any(folded_bool):
            return jnp.array([], dtype=int)

        free = self._free_with_default(free)
        shape_ok, err_msg = self._validate_folded_shape(folded_bool)
        if not shape_ok:
            raise ValueError(err_msg)
        if not free:
            folded_indices = self.fold(
                jnp.arange(self.flat_length(False), dtype=int),
                validate_value=False, free=False)
            return folded_indices[folded_bool]
        else:
            # This indicates that each folded value depends on each
            # free value.  I think this is not true, but getting the exact
            # pattern may be complicated and will
            # probably not make much of a difference in practice.
            if jnp.any(folded_bool):
                return jnp.arange(self.flat_length(True), dtype=int)
            else:
                return jnp.array([])



def _constrain_simplex_matrix(free_mat):
    # The first column is the reference value.  Append a column of zeros
    # to each simplex representing this reference value.
    reference_col = jnp.expand_dims(np.full(free_mat.shape[0:-1], 0), axis=-1)
    free_mat_aug = jnp.concatenate([reference_col, free_mat], axis=-1)

    log_norm = logsumexp(free_mat_aug, axis=-1, keepdims=True)
    return jnp.exp(free_mat_aug - log_norm)


def _constrain_simplex_jacobian(simplex_vec):
    jac = \
        -1 * jnp.outer(simplex_vec, simplex_vec) + \
        jnp.diag(simplex_vec)
    return jac[:, 1:]


def _unconstrain_simplex_matrix(simplex_array):
    return jnp.log(simplex_array[..., 1:]) - \
           jnp.expand_dims(jnp.log(simplex_array[..., 0]), axis=-1)


def _unconstrain_simplex_jacobian(simplex_vec):
    """Get the unconstraining Jacobian for a single simplex vector.
    """
    return np.hstack(
        [ jnp.full(len(simplex_vec) - 1, -1 / simplex_vec[0])[:, None],
          jnp.diag(1 / simplex_vec[1: ]) ])


class SimplexArrayPattern(Pattern):
    """
    A pattern for an array of simplex parameters.

    The last index represents entries of the simplex.  For example,
    if `array_shape=(2, 3)` and `simplex_size=4`, then the pattern is
    for a 2x3 array of 4d simplexes.  If such value of the simplex
    array is given by `val`, then `val.shape = (2, 3, 4)` and
    `val[i, j, :]` is the `i,j`th of the six simplicial vectors, e.g,
    `np.sum(val[i, j, :])` equals 1 for each `i` and `j`.

    Attributes
    -------------
    default_validate: Bool
        Whether or not the simplex is checked by default to be
        non-negative and to sum to one.

    Methods
    ---------
    array_shape: tuple of ints
        The shape of the array of simplexes, not including the simplex
        dimension.

    simplex_size: int
        The length of each simplex.

    shape: tuple of ints
        The shape of the entire array including the simplex dimension.
    """
    def __init__(self, simplex_size, array_shape, default_validate=True,
                 free_default=None):
        """
        Parameters
        ------------
        simplex_size: `int`
            The length of the simplexes.
        array_shape: `tuple` of `int`
            The size of the array of simplexes (not including the simplexes
            themselves).
        default_validate: `bool`, optional
            Whether or not to check for legal (i.e., positive and normalized)
            folded values by default.
        free_default: `bool`, optional
            The default value for free.
        """
        self.__simplex_size = int(simplex_size)
        if self.__simplex_size <= 1:
            raise ValueError('simplex_size must be >= 2.')
        self.__array_shape = array_shape
        self.__shape = self.__array_shape + (self.__simplex_size, )
        self.__free_shape = self.__array_shape + (self.__simplex_size - 1, )
        self.default_validate = default_validate
        super().__init__(np.prod(np.array(self.__shape)),
                         np.prod(np.array(self.__free_shape)),
                         free_default=free_default)

    def __str__(self):
        return 'SimplexArrayPattern {} of {}-d simplices'.format(
            self.__array_shape, self.__simplex_size)

    def array_shape(self):
        return self.__array_shape

    def simplex_size(self):
        return self.__simplex_size

    def shape(self):
        return self.__shape

    def as_dict(self):
        return {
            'pattern': self.json_typename(),
            'simplex_size': self.__simplex_size,
            'array_shape': self.__array_shape,
            'default_validate': self.default_validate}

    def empty(self, valid):
        if valid:
            return np.full(self.__shape, 1.0 / self.__simplex_size)
        else:
            return np.empty(self.__shape)

    def _validate_folded_shape(self, folded_val):
        if folded_val.shape != self.__shape:
            return False, 'The folded value has the wrong shape.'
        else:
            return True, ''

    def validate_folded(self, folded_val, validate_value=None):
        shape_ok, err_msg = self._validate_folded_shape(folded_val)
        if not shape_ok:
            raise ValueError(err_msg)
        if validate_value is None:
            validate_value = self.default_validate
        if validate_value:
            if jnp.any(folded_val < 0):
                return False, 'Some values are negative.'
            simplex_sums = jnp.sum(folded_val, axis=-1)
            if jnp.any(jnp.abs(simplex_sums - 1) > 1e-6):
                return False, 'The simplexes do not sum to one.'
        return True, ''

    def fold(self, flat_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        flat_size = self.flat_length(free)
        if len(flat_val) != flat_size:
            raise ValueError('flat_val is the wrong length.')
        if free:
            free_mat = np.reshape(flat_val, self.__free_shape)
            return _constrain_simplex_matrix(free_mat)
        else:
            folded_val = np.reshape(flat_val, self.__shape)
            valid, msg = self.validate_folded(folded_val, validate_value)
            if not valid:
                raise ValueError(msg)
            return folded_val

    def flatten(self, folded_val, free=None, validate_value=None):
        free = self._free_with_default(free)
        valid, msg = self.validate_folded(folded_val, validate_value)
        if not valid:
            raise ValueError(msg)
        if free:
            return _unconstrain_simplex_matrix(folded_val).flatten()
        else:
            return folded_val.flatten()

    def freeing_jacobian(self, folded_val, sparse=False):
        array_ranges = [ range(i) for i in self.__array_shape ]
        jacobians = []
        for item in itertools.product(*array_ranges):
            jac = _unconstrain_simplex_jacobian(folded_val[item][:])
            jacobians.append(jac)
        sp_jac = block_diag(jacobians, format='coo')

        if sparse:
            return sp_jac
        else:
            return sp_jac.todense()

    def unfreeing_jacobian(self, folded_val, sparse=False):
        array_ranges = [ range(i) for i in self.__array_shape ]
        jacobians = []
        for item in itertools.product(*array_ranges):
            jac = _constrain_simplex_jacobian(folded_val[item][:])
            jacobians.append(jac)
        sp_jac = block_diag(jacobians, format='coo')

        if sparse:
            return sp_jac
        else:
            return sp_jac.todense()

    @classmethod
    def from_json(cls, json_string):
        """
        Return a pattern instance from ``json_string`` created by ``to_json``.
        """
        json_dict = json.loads(json_string)
        cls._validate_json_dict_type(json_dict)
        return cls(
            simplex_size=json_dict['simplex_size'],
            array_shape=tuple(json_dict['array_shape']),
            default_validate=json_dict['default_validate'])

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
            # Every element of a particular simplex depends on all
            # the free values for that simplex.

            # The simplex is the last index, which moves the fastest.
            indices = []
            offset = 0
            free_simplex_length = self.__simplex_size - 1
            array_ranges = (range(n) for n in self.__array_shape)
            for ind in itertools.product(*array_ranges):
                if np.any(folded_bool[ind]):
                    free_inds = np.arange(
                        offset * free_simplex_length,
                        (offset + 1) * free_simplex_length,
                        dtype=int)
                    indices.append(free_inds)
                offset += 1
            if len(indices) > 0:
                return np.hstack(indices)
            else:
                return np.array([])


register_pattern_json(SimplexArrayPattern)
register_pattern_json(PSDSymmetricMatrixPattern)
register_pattern_json(NumericVectorPattern)
register_pattern_json(NumericScalarPattern)
register_pattern_json(NumericArrayPattern)
register_pattern_json(PatternDict)
register_pattern_json(PatternArray)



