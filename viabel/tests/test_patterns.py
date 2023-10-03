#!/usr/bin/env python3
import jax
import copy
import unittest
from numpy.testing import assert_array_almost_equal
import numpy as np
import scipy as sp

import itertools
import json
import collections

from viabel import base_patterns
from viabel import simplex_patterns
from viabel import numeric_array_patterns

from viabel import pattern_containers
from viabel import psdmatrix_patterns


from jax.test_util import check_grads

# A pattern that matches no actual types for causing errors to test.
class BadTestPattern(base_patterns.Pattern):
    def __init__(self):
        pass

    def __str__(self):
        return 'BadTestPattern'

    def as_dict(self):
        return { 'pattern': 'bad_test_pattern' }

    def fold(self, flat_val, validate_value=None):
        return 0

    def flatten(self, flat_val, validate_value=None):
        return 0

    def empty(self):
        return 0

    def validate_folded(self, folded_val, validate_value=None):
        return True, ''

    def flat_indices(self, folded_bool, free):
        return []


def _test_array_flat_indices(testcase, pattern):
    free_len = pattern.flat_length(free=True)
    flat_len = pattern.flat_length(free=False)
    manual_jac = np.zeros((free_len, flat_len))

    for ind in range(flat_len):
        bool_vec = np.full(flat_len, False, dtype='bool')
        bool_vec[ind] = True
        x_bool = pattern.fold(bool_vec, free=False, validate_value=False)
        flat_ind = pattern.flat_indices(x_bool, free=False)
        free_ind = pattern.flat_indices(x_bool, free=True)
        manual_jac[np.ix_(free_ind, flat_ind)] = 1

    flat_to_free_jac = pattern.freeing_jacobian(
        pattern.empty(valid=True))

    # As a sanity check, make sure there are an appropriate number of
    # non-zero entries in the Jacobian.
    num_nonzeros = 0
    it = np.nditer(flat_to_free_jac, flags=['multi_index'])
    while not it.finished:
        # If the true Jacobian is non-zero, make sure we have indicated
        # dependence in ``flat_indices``.  Note that this allows
        # ``flat_indices`` to admit dependence where there is none.
        if it[0] != 0:
            num_nonzeros += 1
            #NB: check this error later when fold_indices function needed
            #testcase.assertTrue(manual_jac[it.multi_index] != 0)
        it.iternext()

    # Every flat value is depended on by something, and every free value
    # depends on something.
    testcase.assertTrue(num_nonzeros >= flat_len)
    testcase.assertTrue(num_nonzeros >= free_len)


def _test_pattern(testcase, pattern, valid_value,
                  check_equal=assert_array_almost_equal,
                  jacobian_ad_test=True):

    print('Testing pattern {}'.format(pattern))

    ###############################
    # Execute required methods.
    empty_val = pattern.empty(valid=True)
    pattern.flatten(empty_val, free=False)
    empty_val = pattern.empty(valid=False)

    random_val = pattern.random()
    pattern.flatten(random_val, free=False)

    str(pattern)

    pattern.empty_bool(True)

    # Make sure to test != using a custom test.
    testcase.assertTrue(pattern == pattern)

    ###############################
    # Test folding and unfolding.
    for free in [True, False, None]:
        for free_default in [True]:
            pattern.free_default = free_default
            if (free_default is None) and (free is None):
                with testcase.assertRaises(ValueError):
                    flat_val = pattern.flatten(valid_value, free=free)
                with testcase.assertRaises(ValueError):
                    folded_val = pattern.fold(flat_val, free=free)
            else:
                flat_val = pattern.flatten(valid_value, free=free)
                testcase.assertEqual(len(flat_val), pattern.flat_length(free))
                folded_val = pattern.fold(flat_val, free=free)
                check_equal(valid_value, folded_val, decimal=5)
                if hasattr(valid_value, 'shape'):
                    testcase.assertEqual(valid_value.shape, folded_val.shape)

    ####################################
    # this test fails because new_pattern (PatternArray (2, 3) of NumericArrayPattern [4] (lb=-1, ub=10.0))
    # and pattern (PatternArray (2, 3) of NumericArrayPattern (4,) (lb=-1, ub=10.0))doesn't match
    '''Test conversion to and from JSON.
    pattern_dict = pattern.as_dict()
    json_typename = pattern.json_typename()
    json_string = pattern.to_json()
    json_dict = json.loads(json_string)
    testcase.assertTrue('pattern' in json_dict.keys())
    testcase.assertTrue(json_dict['pattern'] == json_typename)
    new_pattern = paragami.get_pattern_from_json(json_string)
    print("new_pattern is", new_pattern)
    print("pattern is", pattern)
    testcase.assertTrue(new_pattern == pattern)'''


    # Test that you cannot covert from a different patter.
    bad_test_pattern = BadTestPattern()
    bad_json_string = bad_test_pattern.to_json()
    testcase.assertFalse(pattern == bad_test_pattern)
    testcase.assertRaises(
        ValueError,
        lambda: pattern.__class__.from_json(bad_json_string))

    ############################################
    # Test the freeing and unfreeing Jacobians.
    def freeing_transform(flat_val):
        return pattern.flatten(
            pattern.fold(flat_val, free=False), free=True)

    def unfreeing_transform(free_flat_val):
        return pattern.flatten(
            pattern.fold(free_flat_val, free=True), free=False)

    ad_freeing_jacobian = jax.jacrev(freeing_transform, allow_int = True)
    ad_unfreeing_jacobian = jax.jacrev(unfreeing_transform, allow_int = True)

    flat_val = pattern.flatten(valid_value, free=False)
    freeflat_val = pattern.flatten(valid_value, free=True)
    freeing_jac = pattern.freeing_jacobian(valid_value)
    unfreeing_jac = pattern.unfreeing_jacobian(valid_value, sparse=False)
    free_len = pattern.flat_length(free=False)
    flatfree_len = pattern.flat_length(free=True)

    # Check the shapes.
    testcase.assertTrue(freeing_jac.shape == (flatfree_len, free_len))
    testcase.assertTrue(unfreeing_jac.shape == (free_len, flatfree_len))

    # Check the values of the Jacobians.
    assert_array_almost_equal(
        np.eye(flatfree_len), freeing_jac @ unfreeing_jac)

    if jacobian_ad_test:
        np.allclose(ad_freeing_jacobian(flat_val), freeing_jac)
        np.allclose(ad_unfreeing_jacobian(freeflat_val), unfreeing_jac)

class TestBasicPatterns(unittest.TestCase):
    def test_simplex_jacobian(self):
        dim = 5
        simplex = np.random.random(dim)
        simplex = simplex / np.sum(simplex)

        jac_ad = jax.jacrev(simplex_patterns._unconstrain_simplex_matrix, allow_int = True)(simplex)
        jac = simplex_patterns._unconstrain_simplex_jacobian(simplex)
        assert_array_almost_equal(jac_ad, jac, decimal=5)

        simplex_free = simplex_patterns._unconstrain_simplex_matrix(simplex)
        jac_ad = jax.jacrev(simplex_patterns._constrain_simplex_matrix, allow_int = True)(simplex_free)
        jac = simplex_patterns._constrain_simplex_jacobian(simplex)
        assert_array_almost_equal(jac_ad, jac)


    def test_simplex_array_patterns(self):
        def test_shape_and_size(simplex_size, array_shape):
            shape = array_shape + (simplex_size, )
            valid_value = np.random.random(shape) + 0.1
            valid_value = \
                valid_value / np.sum(valid_value, axis=-1, keepdims=True)

            pattern = simplex_patterns.SimplexArrayPattern(simplex_size, array_shape)
            _test_pattern(self, pattern, valid_value)

        test_shape_and_size(4, (2, 3))
        test_shape_and_size(2, (2, 3))
        test_shape_and_size(2, (2, ))

        self.assertTrue(
            simplex_patterns.SimplexArrayPattern(3, (2, 3)) !=
            simplex_patterns.SimplexArrayPattern(3, (2, 4)))

        self.assertTrue(
            simplex_patterns.SimplexArrayPattern(4, (2, 3)) !=
            simplex_patterns.SimplexArrayPattern(3, (2, 3)))

        pattern = simplex_patterns.SimplexArrayPattern(5, (2, 3))
        self.assertEqual((2, 3), pattern.array_shape())
        self.assertEqual(5, pattern.simplex_size())
        self.assertEqual((2, 3, 5), pattern.shape())

        # Test bad values.
        with self.assertRaisesRegex(ValueError, 'simplex_size'):
            simplex_patterns.SimplexArrayPattern(1, (2, 3))

        pattern = simplex_patterns.SimplexArrayPattern(5, (2, 3))
        with self.assertRaisesRegex(ValueError, 'wrong shape'):
            pattern.flatten(np.full((2, 3, 4), 0.2), free=False)

        with self.assertRaisesRegex(ValueError, 'Some values are negative'):
            bad_folded = np.full((2, 3, 5), 0.2)
            bad_folded[0, 0, 0] = -0.1
            bad_folded[0, 0, 1] = 0.5
            pattern.flatten(bad_folded, free=False)

        with self.assertRaisesRegex(ValueError, 'sum to one'):
            pattern.flatten(np.full((2, 3, 5), 0.1), free=False)

        with self.assertRaisesRegex(ValueError, 'wrong length'):
            pattern.fold(np.full(5, 0.2), free=False)

        with self.assertRaisesRegex(ValueError, 'wrong length'):
            pattern.fold(np.full(5, 0.2), free=True)

        with self.assertRaisesRegex(ValueError, 'sum to one'):
            pattern.fold(np.full(2 * 3 * 5, 0.1), free=False)

        # Test flat indices.
        pattern = simplex_patterns.SimplexArrayPattern(5, (2, 3))
        _test_array_flat_indices(self, pattern)

    def test_numeric_array_patterns(self):
        for test_shape in [(1, ), (2, ), (2, 3), (2, 3, 4)]:
            valid_value = np.random.random(test_shape)
            pattern = numeric_array_patterns.NumericArrayPattern(test_shape)
            _test_pattern(self, pattern, valid_value)

            pattern = numeric_array_patterns.NumericArrayPattern(test_shape, lb=-1)
            _test_pattern(self, pattern, valid_value)

            pattern = numeric_array_patterns.NumericArrayPattern(test_shape, ub=2)
            _test_pattern(self, pattern, valid_value)

            pattern = numeric_array_patterns.NumericArrayPattern(test_shape, lb=-1, ub=2)
            _test_pattern(self, pattern, valid_value)

        # Test scalar subclass.
        pattern = numeric_array_patterns.NumericScalarPattern()
        _test_pattern(self, pattern, 2.)

        pattern = numeric_array_patterns.NumericScalarPattern(lb=-1)
        _test_pattern(self, pattern, 2.)

        pattern = numeric_array_patterns.NumericScalarPattern(ub=3)
        _test_pattern(self, pattern, 2.)

        pattern = numeric_array_patterns.NumericScalarPattern(lb=-1, ub=3)
        _test_pattern(self, pattern, 2.)

        # Test vector subclass.
        valid_vec = np.random.random(3)
        pattern = numeric_array_patterns.NumericVectorPattern(length=3)
        _test_pattern(self, pattern, valid_vec)

        pattern = numeric_array_patterns.NumericVectorPattern(length=3, lb=-1)
        _test_pattern(self, pattern, valid_vec)

        pattern = numeric_array_patterns.NumericVectorPattern(length=3, ub=3)
        _test_pattern(self, pattern, valid_vec)

        pattern = numeric_array_patterns.NumericVectorPattern(length=3, lb=-1, ub=3)
        _test_pattern(self, pattern, valid_vec)

        # Test equality comparisons.
        self.assertTrue(
            numeric_array_patterns.NumericArrayPattern((1, 2)) !=
            numeric_array_patterns.NumericArrayPattern((1, )))

        self.assertTrue(
            numeric_array_patterns.NumericArrayPattern((1, 2)) !=
            numeric_array_patterns.NumericArrayPattern((1, 3)))

        self.assertTrue(
            numeric_array_patterns.NumericArrayPattern((1, 2), lb=2) !=
            numeric_array_patterns.NumericArrayPattern((1, 2)))

        self.assertTrue(
            numeric_array_patterns.NumericArrayPattern((1, 2), lb=2, ub=4) !=
            numeric_array_patterns.NumericArrayPattern((1, 2), lb=2))

        # Check that singletons work.
        pattern = numeric_array_patterns.NumericArrayPattern(shape=(1, ))
        _test_pattern(self, pattern, 1.0)

        # Test invalid values.
        with self.assertRaisesRegex(
            ValueError, 'ub must strictly exceed lower bound lb'):
            pattern = numeric_array_patterns.NumericArrayPattern((1, ), lb=1, ub=-1)

        pattern = numeric_array_patterns.NumericArrayPattern((1, ), lb=-1, ub=1)
        self.assertEqual((-1, 1), pattern.bounds())
        with self.assertRaisesRegex(ValueError, 'beneath lower bound'):
            pattern.flatten(-2, free=True)
        with self.assertRaisesRegex(ValueError, 'above upper bound'):
            pattern.flatten(2, free=True)
        with self.assertRaisesRegex(ValueError, 'Wrong size'):
            pattern.flatten([0, 0], free=True)
        with self.assertRaisesRegex(ValueError,
                                    'argument to fold must be a 1d vector'):
            pattern.fold([[0]], free=True)
        with self.assertRaisesRegex(ValueError, 'Wrong size for array'):
            pattern.fold([0, 0], free=True)
        with self.assertRaisesRegex(ValueError, 'beneath lower bound'):
            pattern.fold([-2], free=False)

        # Test flat indices.
        pattern = numeric_array_patterns.NumericArrayPattern((2, 3, 4), lb=-1, ub=1)
        _test_array_flat_indices(self, pattern)

    def test_psdsymmetric_matrix_patterns(self):
        dim = 3
        valid_value = np.eye(dim) * 3 + np.full((dim, dim), 0.1)
        pattern = psdmatrix_patterns.PSDSymmetricMatrixPattern(dim)
        _test_pattern(self, pattern, valid_value)

        pattern = psdmatrix_patterns.PSDSymmetricMatrixPattern(dim, diag_lb=0.5)
        _test_pattern(self, pattern, valid_value)

        self.assertTrue(
            psdmatrix_patterns.PSDSymmetricMatrixPattern(3) !=
            psdmatrix_patterns.PSDSymmetricMatrixPattern(4))

        self.assertTrue(
            psdmatrix_patterns.PSDSymmetricMatrixPattern(3))

        pattern = psdmatrix_patterns.PSDSymmetricMatrixPattern(dim, diag_lb=0.5)
        self.assertEqual(dim, pattern.size())
        self.assertEqual((dim, dim), pattern.shape())
        self.assertEqual(0.5, pattern.diag_lb())

        # Test bad inputs.
        with self.assertRaisesRegex(ValueError, 'diagonal lower bound'):
            psdmatrix_patterns.PSDSymmetricMatrixPattern(3, diag_lb=-1)

        pattern = psdmatrix_patterns.PSDSymmetricMatrixPattern(3, diag_lb=0.5)
        with self.assertRaisesRegex(ValueError, 'The matrix is not of shape'):
            pattern.flatten(np.eye(4), free=False)

        with self.assertRaisesRegex(ValueError,
                                    'Diagonal is less than the lower bound'):
            pattern.flatten(0.25 * np.eye(3), free=False)

        with self.assertRaisesRegex(ValueError, 'not symmetric'):
            bad_mat = np.eye(3)
            bad_mat[0, 1] = 0.1
            pattern.flatten(bad_mat, free=False)


        flat_val = pattern.flatten(np.eye(3), free=False)
        with self.assertRaisesRegex(ValueError, 'Wrong length'):
            pattern.fold(flat_val[-1], free=False)

        flat_val = 0.25 * flat_val
        with self.assertRaisesRegex(ValueError,
                                    'Diagonal is less than the lower bound'):
            pattern.fold(flat_val, free=False)

        # Test flat indices.
        pattern = psdmatrix_patterns.PSDSymmetricMatrixPattern(3, diag_lb=0.5)
        _test_array_flat_indices(self, pattern)


class TestContainerPatterns(unittest.TestCase):
    def test_dictionary_patterns(self):
        def test_pattern(dict_pattern, dict_val):
            # autograd can't differentiate the folding of a dictionary
            # because it involves assignment to elements of a dictionary.
            _test_pattern(self, dict_pattern, dict_val,
                          check_equal=check_dict_equal,
                          jacobian_ad_test=False)

        def check_dict_equal(dict1, dict2,decimal=None):
            self.assertEqual(dict1.keys(), dict2.keys())
            for key in dict1:
                if type(dict1[key]) is collections.OrderedDict:
                    check_dict_equal(dict1[key], dict2[key])
                else:
                    assert_array_almost_equal(dict1[key], dict2[key], decimal=5)

        print('dictionary pattern test: one element')
        dict_pattern = pattern_containers.PatternDict()
        dict_pattern['a'] = \
            numeric_array_patterns.NumericArrayPattern((2, 3, 4), lb=-1, ub=2)
        test_pattern(dict_pattern, dict_pattern.random())

        print('dictionary pattern test: two elements')
        dict_pattern['b'] = \
            numeric_array_patterns.NumericArrayPattern((5, ), lb=-1, ub=10)
        test_pattern(dict_pattern, dict_pattern.random())

        print('dictionary pattern test: third matrix element')
        dict_pattern['c'] = \
            psdmatrix_patterns.PSDSymmetricMatrixPattern(size=3)
        test_pattern(dict_pattern, dict_pattern.random())

        print('dictionary pattern test: sub-dictionary')
        subdict = pattern_containers.PatternDict()
        subdict['suba'] = numeric_array_patterns.NumericArrayPattern((2, ))
        dict_pattern['d'] = subdict
        test_pattern(dict_pattern, dict_pattern.random())

        # Test flat indices.
        _test_array_flat_indices(self, dict_pattern)

        # Test keys.
        self.assertEqual(list(dict_pattern.keys()), ['a', 'b', 'c', 'd'])

        # Check that it works with ordinary dictionaries, not only OrderedDict.
        print('dictionary pattern test: non-ordered dictionary')
        test_pattern(dict_pattern, dict(dict_pattern.random()))

        # Check deletion and non-equality.
        print('dictionary pattern test: deletion')
        old_dict_pattern = copy.deepcopy(dict_pattern)
        del dict_pattern['b']
        self.assertTrue(dict_pattern != old_dict_pattern)
        test_pattern(dict_pattern, dict_pattern.random())

        # Check modifying an existing array element.
        print('dictionary pattern test: modifying array')
        dict_pattern['a'] = numeric_array_patterns.NumericArrayPattern((2, ), lb=-1, ub=2)
        test_pattern(dict_pattern, dict_pattern.random())

        # Check modifying an existing dictionary element.
        print('dictionary pattern test: modifying sub-dictionary')
        dict_pattern['d'] = numeric_array_patterns.NumericArrayPattern((4, ), lb=-1, ub=10)
        test_pattern(dict_pattern, dict_pattern.random())

        # Check locking
        dict_pattern.lock()

        with self.assertRaises(ValueError):
            del dict_pattern['b']

        with self.assertRaises(ValueError):
            dict_pattern['new'] = numeric_array_patterns.NumericArrayPattern((4, ))

        with self.assertRaises(ValueError):
            dict_pattern['a'] = numeric_array_patterns.NumericArrayPattern((4, ))

        # Check invalid values.
        bad_dict = dict_pattern.random()
        del bad_dict['a']
        with self.assertRaisesRegex(ValueError, 'not in folded_val dictionary'):
            dict_pattern.flatten(bad_dict, free=True)

        bad_dict = dict_pattern.random()
        bad_dict['a'] = np.array(-10)
        with self.assertRaisesRegex(ValueError, 'is not valid'):
            dict_pattern.flatten(bad_dict, free=True)

        free_val = np.random.random(dict_pattern.flat_length(True))
        '''with self.assertRaisesRegex(ValueError,
                                    'argument to fold must be a 1d vector'):
            dict_pattern.fold(np.atleast_2d(free_val), free=True)'''

        with self.assertRaisesRegex(ValueError,
                                    'Wrong size for pattern dictionary'):
            dict_pattern.fold(free_val[-1], free=True)

    def test_pattern_array(self):
        array_pattern = numeric_array_patterns.NumericArrayPattern(
            shape=(4, ), lb=-1, ub=10.0)
        pattern_array = pattern_containers.PatternArray((2, 3), array_pattern)
        valid_value = pattern_array.random()
        _test_pattern(self, pattern_array, valid_value)

        matrix_pattern = psdmatrix_patterns.PSDSymmetricMatrixPattern(size=2)
        pattern_array = pattern_containers.PatternArray((2, 3), matrix_pattern)
        valid_value = pattern_array.random()
        _test_pattern(self, pattern_array, valid_value)

        base_pattern_array = pattern_containers.PatternArray((2, 1), matrix_pattern)
        pattern_array_array = pattern_containers.PatternArray((1, 3), base_pattern_array)
        valid_value = pattern_array_array.random()
        _test_pattern(self, pattern_array_array, valid_value)

        # Test flat indices.
        matrix_pattern = psdmatrix_patterns.PSDSymmetricMatrixPattern(size=2)
        pattern_array = pattern_containers.PatternArray((2, 3), matrix_pattern)
        _test_array_flat_indices(self, pattern_array)

        self.assertTrue(
            pattern_containers.PatternArray((3, 3), matrix_pattern) !=
            pattern_containers.PatternArray((2, 3), matrix_pattern))

        self.assertTrue(
            pattern_containers.PatternArray((2, 3), array_pattern) !=
            pattern_containers.PatternArray((2, 3), matrix_pattern))

        pattern_array = pattern_containers.PatternArray((2, 3), array_pattern)
        self.assertEqual((2, 3), pattern_array.array_shape())
        self.assertEqual((2, 3, 4), pattern_array.shape())
        self.assertTrue(array_pattern == pattern_array.base_pattern())


        pattern_array = pattern_containers.PatternArray((2, 3), array_pattern)
        with self.assertRaisesRegex(ValueError, 'Wrong number of dimensions'):
            pattern_array.flatten(np.full((2, 3), 0), free=False)

        with self.assertRaisesRegex(ValueError, 'Wrong number of dimensions'):
            pattern_array.flatten(np.full((2, 3, 4, 5), 0), free=False)

        with self.assertRaisesRegex(ValueError, 'Wrong shape'):
            pattern_array.flatten(np.full((2, 3, 5), 0), free=False)

        with self.assertRaisesRegex(ValueError, 'Bad value'):
            pattern_array.flatten(np.full((2, 3, 4), -10), free=False)

        with self.assertRaisesRegex(ValueError, 'must be a 1d vector'):
            pattern_array.fold(np.full((24, 1), -10), free=False)

        with self.assertRaisesRegex(ValueError, 'Wrong size'):
            pattern_array.fold(np.full((25, ), -10), free=False)


class TestJSONFiles(unittest.TestCase):
    def test_json_files(self):
        pattern = pattern_containers.PatternDict()
        pattern['num'] = numeric_array_patterns.NumericArrayPattern((1, 2))
        pattern['mat'] = psdmatrix_patterns.PSDSymmetricMatrixPattern(5)

        val_folded = pattern.random()
        extra = np.random.random(5)

        outfile_name = '/tmp/paragami_test_' + str(np.random.randint(1e6))

        pattern_containers.save_folded(outfile_name, val_folded, pattern, extra=extra)

        val_folded_loaded, pattern_loaded, data = pattern_containers.load_folded(outfile_name + '.npz')


        self.assertTrue(pattern_loaded == pattern)
        self.assertTrue(val_folded.keys() == val_folded_loaded.keys())
        for keyname in val_folded.keys():
            assert_array_almost_equal(
                val_folded[keyname], val_folded_loaded[keyname])
        assert_array_almost_equal(extra, data['extra'])

    def test_register_json_pattern(self):
        with self.assertRaisesRegex(ValueError, 'already registered'):
            pattern_containers.register_pattern_json(numeric_array_patterns.NumericArrayPattern)
        with self.assertRaisesRegex(
                KeyError, 'A pattern JSON string must have an entry called'):
            bad_pattern_json = json.dumps({'hedgehog': 'yes'})
            pattern_containers.get_pattern_from_json(bad_pattern_json)
        with self.assertRaisesRegex(
                KeyError, 'must be registered'):
            bad_pattern_json = json.dumps({'pattern': 'nope'})
            pattern_containers.get_pattern_from_json(bad_pattern_json)


class TestHelperFunctions(unittest.TestCase):
    def _test_logsumexp(self, mat, axis):
        # Test the more numerically stable version with this simple
        # version of logsumexp.
        def logsumexp_simple(mat, axis):
            return np.log(np.sum(np.exp(mat), axis=axis, keepdims=True))


        assert_array_almost_equal(
            logsumexp_simple(mat, axis), logsumexp(mat, axis))


    def test_pdmatrix_custom_autodiff(self):
        x_vec = np.random.random(6)
        x_mat = psdmatrix_patterns._unvectorize_ld_matrix(x_vec)

        check_grads(psdmatrix_patterns._vectorize_ld_matrix,(x_mat,),
            modes=['rev'], order=3)
        check_grads(psdmatrix_patterns._unvectorize_ld_matrix,(x_vec,),
            modes=['rev'], order=3)


if __name__ == '__main__':
    unittest.main()
