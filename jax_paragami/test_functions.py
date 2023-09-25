#!/usr/bin/env python3

import unittest
from numpy.testing import assert_array_almost_equal

import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads
from copy import deepcopy
import itertools
import jax_paragami as paragami


def get_test_pattern():
    # autograd will pass invalid values, so turn off value checking.
    pattern = paragami.PatternDict()
    pattern['array'] = paragami.NumericArrayPattern(
        (2, 3, 4), lb=-1, ub=20, default_validate=False)
    pattern['mat'] = paragami.PSDSymmetricMatrixPattern(
        3, default_validate=False)
    pattern['simplex'] = paragami.SimplexArrayPattern(
        2, (3, ), default_validate=False)
    subdict = paragami.PatternDict()
    subdict['array2'] = paragami.NumericArrayPattern(
        (2, ), lb=-3, ub=10, default_validate=False)
    pattern['dict'] = subdict

    return pattern

def get_small_test_pattern():
    # autograd will pass invalid values, so turn off value checking.
    pattern = paragami.PatternDict()
    pattern['array'] = paragami.NumericArrayPattern(
        (2, 3, 4), lb=-1, ub=10, default_validate=False)
    pattern['mat'] = paragami.PSDSymmetricMatrixPattern(
        3, default_validate=False)

    return pattern


def assert_test_dict_equal(d1, d2):
    """Assert that dictionaries corresponding to test pattern are equal.
    """
    for k in ['array', 'mat', 'simplex']:
        assert_array_almost_equal(d1[k], d2[k])
    assert_array_almost_equal(d1['dict']['array2'], d2['dict']['array2'])


# Test functions that work with get_test_pattern() or
# get_small_test_pattern().
def fold_to_num(param_folded):
    return \
        np.mean(param_folded['array'] ** 2) + \
        np.mean(param_folded['mat'] ** 2)

def flat_to_num(param_flat, pattern, free):
    param_folded = pattern.fold(param_flat, free=free)
    return fold_to_num(param_folded)

def num_to_fold(x, pattern):
    new_param = pattern.empty(valid=True)
    new_param['array'] = new_param['array'] + x
    new_param['mat'] = x * new_param['mat']
    return new_param

def num_to_flat(x, pattern, free):
    new_param = num_to_fold(x, pattern)
    return pattern.flatten(new_param, free=free)


class TestFlatteningAndFolding(unittest.TestCase):
    def _test_transform_input(
        self, original_fun, patterns, free, argnums, original_is_flat,
        folded_args, flat_args, kwargs):

        orig_args = flat_args if original_is_flat else folded_args
        trans_args = folded_args if original_is_flat else flat_args
        fun_trans = paragami.TransformFunctionInput(
            original_fun, patterns, free,
            original_is_flat, argnums)

        # Check that the flattened and original function are the same.
        jnp.allclose(
            original_fun(*orig_args, **kwargs),
            fun_trans(*trans_args, **kwargs))

        # Check that the string method works.
        str(fun_trans)

    def _test_transform_output(
        self, original_fun, patterns, free, retnums, original_is_flat):
        # original_fun must take no arguments.

        fun_trans = paragami.TransformFunctionOutput(
            original_fun, patterns, free, original_is_flat, retnums)

        # Check that the flattened and original function are the same.
        def check_equal(orig_val, trans_val, pattern, free):
            # Use the flat representation to check that parameters are equal.
            if original_is_flat:
                jnp.allclose(
                    orig_val, pattern.flatten(trans_val, free=free))
            else:
                jnp.allclose(
                    pattern.flatten(orig_val, free=free), trans_val)

        patterns_array = np.atleast_1d(patterns)
        free_array = np.atleast_1d(free)
        retnums_array = np.atleast_1d(retnums)

        orig_rets = original_fun()
        trans_rets = fun_trans()
        if isinstance(orig_rets, tuple):
            self.assertTrue(len(orig_rets) == len(trans_rets))

            # Check that the non-transformed return values are the same.
            for ind in range(len(orig_rets)):
                if not np.isin(ind, retnums):
                    assert_array_almost_equal(
                        orig_rets[ind], trans_rets[ind])

            # Check that the transformed return values are the same.
            for ret_ind in range(len(retnums_array)):
                ind = retnums_array[ret_ind]
                check_equal(
                    orig_rets[ind], trans_rets[ind],
                    patterns_array[ret_ind], free_array[ret_ind])
        else:
            check_equal(
                orig_rets, trans_rets, patterns_array[0], free_array[0])

        # Check that the string method works.
        str(fun_trans)


    def test_transform_input(self):
        pattern = get_test_pattern()
        param_val = pattern.random()
        x = 3
        y = 4
        z = 5

        def scalarfun(x, y, z):
            return x**2 + 2 * y**2 + 3 * z**2

        ft = [False, True]
        for free, origflat in itertools.product(ft, ft):
            def this_flat_to_num(x):
                return flat_to_num(x, pattern, free)

            param_flat = pattern.flatten(param_val, free=free)
            tf1 = this_flat_to_num if origflat else fold_to_num

            def tf2(x, val, y=5):
                return tf1(val) + scalarfun(x, y, 0)

            def tf3(val, x, y=5):
                return tf1(val) + scalarfun(x, y, 0)

            self._test_transform_input(
                original_fun=tf1, patterns=pattern, free=free,
                argnums=0,
                original_is_flat=origflat,
                folded_args=(param_val, ),
                flat_args=(param_flat, ),
                kwargs={})

            # Just call the wrappers -- assume that their functionality
            # is tested with TransformFunctionInput.
            if origflat:
                fold_tf1 = paragami.FoldFunctionInput(
                    tf1, pattern, free, 0)
                assert_array_almost_equal(
                    fold_tf1(param_val), tf1(param_flat))
            else:
                flat_tf1 = paragami.FlattenFunctionInput(
                    tf1, pattern, free, 0)
                jnp.allclose(
                    flat_tf1(param_flat), tf1(param_val))

            self._test_transform_input(
                original_fun=tf2, patterns=pattern, free=free,
                argnums=1,
                original_is_flat=origflat,
                folded_args=(x, param_val, ),
                flat_args=(x, param_flat, ),
                kwargs={'y': 5})

            self._test_transform_input(
                original_fun=tf3, patterns=pattern, free=free,
                argnums=0,
                original_is_flat=origflat,
                folded_args=(param_val, x, ),
                flat_args=(param_flat, x, ),
                kwargs={'y': 5})

            # Test once with arrays.
            self._test_transform_input(
                original_fun=tf3, patterns=[pattern], free=[free],
                argnums=[0],
                original_is_flat=origflat,
                folded_args=(param_val, x, ),
                flat_args=(param_flat, x, ),
                kwargs={'y': 5})

            # Test bad inits
            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionInput(
                    tf1, [[ pattern ]], free, origflat, 0)

            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionInput(
                    tf1, pattern, free, origflat, [[0]])

            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionInput(
                    tf1, pattern, free, origflat, [0, 0])

            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionInput(
                    tf1, pattern, free, origflat, [0, 1])

        # Test two-parameter flattening.
        pattern0 = get_test_pattern()
        pattern1 = get_small_test_pattern()
        param0_val = pattern0.random()
        param1_val = pattern1.random()
        for (free0, free1, origflat) in itertools.product(ft, ft, ft):

            if origflat:
                def tf1(p0, p1):
                    return flat_to_num(p0, pattern0, free0) + \
                           flat_to_num(p1, pattern1, free1)
            else:
                def tf1(p0, p1):
                    return fold_to_num(p0) + fold_to_num(p1)

            def tf2(x, p0, z, p1, y=5):
                return tf1(p0, p1) + scalarfun(x, y, z)

            def tf3(p0, z, p1, x, y=5):
                return tf1(p0, p1) + scalarfun(x, y, z)

            param0_flat = pattern0.flatten(param0_val, free=free0)
            param1_flat = pattern1.flatten(param1_val, free=free1)

            self._test_transform_input(
                original_fun=tf1,
                patterns=[pattern0, pattern1],
                free=[free0, free1],
                argnums=[0, 1],
                original_is_flat=origflat,
                folded_args=(param0_val, param1_val),
                flat_args=(param0_flat, param1_flat),
                kwargs={})

            # Test switching the order of the patterns.
            self._test_transform_input(
                original_fun=tf1,
                patterns=[pattern1, pattern0],
                free=[free1, free0],
                argnums=[1, 0],
                original_is_flat=origflat,
                folded_args=(param0_val, param1_val),
                flat_args=(param0_flat, param1_flat),
                kwargs={})

            self._test_transform_input(
                original_fun=tf2,
                patterns=[pattern1, pattern0],
                free=[free1, free0],
                argnums=[3, 1],
                original_is_flat=origflat,
                folded_args=(x, param0_val, z, param1_val, ),
                flat_args=(x, param0_flat, z, param1_flat),
                kwargs={'y': 5})

            self._test_transform_input(
                original_fun=tf3,
                patterns=[pattern1, pattern0],
                free=[free1, free0],
                argnums=[2, 0],
                original_is_flat=origflat,
                folded_args=(param0_val, z, param1_val, x, ),
                flat_args=(param0_flat, z, param1_flat, x),
                kwargs={'y': 5})


    def test_transform_output(self):
        pattern = get_test_pattern()
        param_val = pattern.random()
        x = 3.
        y = 4.
        z = 5.

        ft = [False, True]
        def this_num_to_fold():
            return num_to_fold(x, pattern)

        for free, origflat in itertools.product(ft, ft):
            def this_num_to_flat():
                return num_to_flat(x, pattern, free)

            #param_flat = pattern.flatten(param_val, free=free)
            tf1 = this_num_to_flat if origflat else this_num_to_fold

            def tf2():
                return tf1(), y

            def tf3():
                return y, tf1(), z

            self._test_transform_output(
                original_fun=tf1, original_is_flat=origflat,
                free=free, patterns=pattern, retnums=0)

            # Just call the wrappers -- assume that their functionality
            # is tested with TransformFunctionOutput.
            if origflat:
                fold_tf1 = paragami.FoldFunctionOutput(
                    tf1, pattern, free, 0)
                jnp.allclose(
                    pattern.flatten(fold_tf1(), free=free), tf1())
            else:
                flat_tf1 = paragami.FlattenFunctionOutput(
                    tf1, pattern, free, 0)
                jnp.allclose(
                    pattern.flatten(tf1(), free=free), flat_tf1())

            self._test_transform_output(
                original_fun=tf2, original_is_flat=origflat,
                free=free, patterns=pattern, retnums=0)

            self._test_transform_output(
                original_fun=tf3, original_is_flat=origflat,
                free=free, patterns=pattern, retnums=1)

            # Test bad inits
            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionOutput(
                    tf1, [[ pattern ]], free, origflat, 0)

            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionOutput(
                    tf1, pattern, free, origflat, [[0]])

            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionOutput(
                    tf1, pattern, free, origflat, [0, 0])

            with self.assertRaises(ValueError):
                fun_flat = paragami.TransformFunctionOutput(
                    tf1, pattern, free, origflat, [0, 1])

        # Test two-parameter transforms.
        pattern0 = get_test_pattern()
        pattern1 = get_small_test_pattern()
        for (free0, free1, origflat) in itertools.product(ft, ft, ft):

            if origflat:
                def basef0():
                    return num_to_flat(x, pattern0, free0)
                def basef1():
                    return num_to_flat(y, pattern1, free1)
            else:
                def basef0():
                    return num_to_fold(x, pattern0)
                def basef1():
                    return num_to_fold(y, pattern1)

            def tf1():
                return basef0(), basef1()

            def tf2():
                return basef0(), z, basef1(), x

            def tf3():
                return x, basef0(), z, basef1()

            self._test_transform_output(
                original_fun=tf1, original_is_flat=origflat,
                free=[free0, free1],
                patterns=[pattern0, pattern1],
                retnums=[0, 1])

            # Test switching the order of the patterns.
            self._test_transform_output(
                original_fun=tf1, original_is_flat=origflat,
                free=[free1, free0],
                patterns=[pattern1, pattern0],
                retnums=[1, 0])

            self._test_transform_output(
                original_fun=tf2, original_is_flat=origflat,
                free=[free1, free0],
                patterns=[pattern1, pattern0],
                retnums=[2, 0])

            self._test_transform_output(
                original_fun=tf3, original_is_flat=origflat,
                free=[free1, free0],
                patterns=[pattern1, pattern0],
                retnums=[3, 1])


    def test_flatten_and_fold(self):
        pattern = get_test_pattern()
        pattern_val = pattern.random()
        free_val = pattern.flatten(pattern_val, free=True)

        def flat_to_flat(par_flat):
            return par_flat + 1.0

        folded_fun = paragami.FoldFunctionInputAndOutput(
            original_fun=flat_to_flat,
            input_patterns=pattern,
            input_free=True,
            input_argnums=0,
            output_patterns=pattern,
            output_free=True)

        folded_out = folded_fun(pattern_val)
        folded_out_test = pattern.fold(
            flat_to_flat(free_val), free=True)
        assert_test_dict_equal(folded_out_test, folded_out)


        def fold_to_fold(par_fold):
            num = fold_to_num(par_fold)
            out_par = deepcopy(par_fold)
            out_par['mat'] *= num
            return out_par

        flat_fun = paragami.FlattenFunctionInputAndOutput(
            original_fun=fold_to_fold,
            input_patterns=pattern,
            input_free=True,
            input_argnums=0,
            output_patterns=pattern,
            output_free=True)

        flat_out = flat_fun(free_val)
        flat_out_test = pattern.flatten(
            fold_to_fold(pattern_val), free=True)
        jnp.allclose(flat_out, flat_out_test)


    '''def test_jax(self):
        pattern = get_test_pattern()

        # The autodiff tests produces non-symmetric matrices.
        pattern['mat'].default_validate = False
        param_val = pattern.random()

        def tf1(param_val):
            return \
                np.mean(param_val['array'] ** 2) + \
                np.mean(param_val['mat'] ** 2)

        for free in [True, False]:
            tf1_flat = paragami.FlattenFunctionInput(tf1, pattern, free)
            param_val_flat = pattern.flatten(param_val, free=free)
            check_grads(tf1_flat, (param_val_flat,), modes=['rev'], order=1)'''


if __name__ == '__main__':
    unittest.main()
