from abc import ABC, abstractmethod
import autograd
import json
import numpy as np
from scipy.sparse import coo_matrix

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
        self._freeing_jacobian = autograd.jacobian(self._freeing_transform)
        self._unfreeing_jacobian = autograd.jacobian(self._unfreeing_transform)

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

    def freeing_jacobian(self, folded_val, sparse=True):
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
        sparse : `bool`, optional
            Whether to return a sparse or a dense matrix.

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
        if sparse:
            return coo_matrix(jac)
        else:
            return jac

    def unfreeing_jacobian(self, folded_val, sparse=True):
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
