from viabel import approximations

import numpy as np
from scipy import stats

import pytest


MC_SAMPLES = 1000000
test_size = 0.0001


def _test_entropy(approx, var_param, entropy_offset):
    entropy = approx.entropy(var_param) + entropy_offset
    log_probs = approx.log_density(var_param, approx.sample(var_param, MC_SAMPLES))

    p_value = stats.ttest_1samp(log_probs, -entropy)[1]
    assert p_value > test_size, "expected: {}, estimated: {}".format(entropy, -np.mean(log_probs))


def _test_kl(approx, var_param0, var_param1):
    kl = approx.kl(var_param0, var_param1)
    samples = approx.sample(var_param0, MC_SAMPLES)
    log_prob_diffs = approx.log_density(var_param0, samples) - approx.log_density(var_param1, samples)

    p_value = stats.ttest_1samp(log_prob_diffs, kl)[1]
    assert p_value > test_size


def _test_mean_and_cov(approx, var_param):
    mean, cov = approx.mean_and_cov(var_param)
    second_moments = np.outer(mean, mean) + cov

    samples = approx.sample(var_param, MC_SAMPLES)
    samples_outer = np.einsum('ij,ik->ijk', samples, samples)

    mean_p_values = stats.ttest_1samp(samples, mean, axis=0)[1]
    np.testing.assert_array_less(test_size, mean_p_values)

    second_moments_p_values = stats.ttest_1samp(samples_outer, second_moments, axis=0)[1]
    np.testing.assert_array_less(test_size, second_moments_p_values)


def _test_pth_moment(approx, var_param, p):
    pth_moment = approx.pth_moment(var_param, p)

    samples = approx.sample(var_param, MC_SAMPLES)
    sample_mean = np.mean(samples, axis=0)
    sample_norms = np.linalg.norm(samples - sample_mean, axis=1, ord=2)

    p_value = stats.ttest_1samp(sample_norms**p, pth_moment)[1]
    assert p_value > test_size, "expected: {}, estimated: {}".format(pth_moment, np.mean(sample_norms**p))


def _test_family(approx, var_param0, var_param1, should_support=[], entropy_offset=0):
    # These tests check that the variational family is defined self-consistently
    if approx.supports_entropy:
        _test_entropy(approx, var_param0, entropy_offset)
    else: # pragma: no cover
        with pytest.raises(NotImplementedError):
            approx.entropy(var_param0)
    if approx.supports_kl:
        _test_kl(approx, var_param0, var_param1)
    else: # pragma: no cover
        with pytest.raises(NotImplementedError):
            approx.kl(var_param0, var_param1)
    _test_mean_and_cov(approx, var_param0)
    for p in set([1, 2,4]) | set(should_support):
        if p in should_support:
            assert approx.supports_pth_moment(p)
        if approx.supports_pth_moment(p):
            _test_pth_moment(approx, var_param0, p)
        else:
            with pytest.raises(ValueError):
                approx.pth_moment(var_param0, p)


def test_MFGaussian():
    np.random.seed(341)
    for dim in [1, 3]:
        approx = approximations.MFGaussian(dim)
        for i in range(3):
            var_param0 = np.random.randn(approx.var_param_dim)
            var_param1 = np.random.randn(approx.var_param_dim)
            _test_family(approx, var_param0, var_param1, [2,4])
    # TODO: check behavior in corner cases


def test_MFStudentT():
    np.random.seed(226)
    df = 20
    entropy_offset_1d = stats.t.entropy(df)
    for dim in [1, 3]:
        entropy_offset = dim * entropy_offset_1d
        approx = approximations.MFStudentT(dim, df)
        for i in range(3):
            var_param0 = np.random.randn(approx.var_param_dim)
            var_param1 = np.random.randn(approx.var_param_dim)
            _test_family(approx, var_param0, var_param1, [2,4], entropy_offset)
    # TODO: check behavior in corner cases


def test_MultivariateT():
    np.random.seed(56)
    df = 100
    entropy_offset_1d = stats.t.entropy(df)
    for dim in [1, 3]:
        entropy_offset = dim * entropy_offset_1d
        approx = approximations.MultivariateT(dim, df)
        for i in range(3):
            var_param0 = np.random.randn(approx.var_param_dim)
            var_param1 = np.random.randn(approx.var_param_dim)
            _test_family(approx, var_param0, var_param1, [2,4], entropy_offset)
    # TODO: check behavior in corner cases


def test_NeuralNet():
    np.random.seed(56)
    df = 100
    for dim in [1, 3]:
        layers_shapes = [[dim, 10], [10, dim]]
        approx = approximations.NeuralNet(layers_shapes)
        for i in range(3):
            var_param0 = np.random.randn(approx.var_param_dim)
            var_param1 = np.random.randn(approx.var_param_dim)
            _test_family(approx, var_param0, var_param1, [0])
    # TODO: check behavior in corner cases


def test_NVP():
    np.random.seed(56)
    df = 100
    for dim in [1, 3]:
        layers_shapes = [[dim, 10], [10, dim]]
        prior = approximations.MFGaussian(dim)
        prior_param = np.concatenate([[0] * dim, [1] * dim]])
        approx = approximations.NVPFlow(layers_shapes, layers_shapes, prior,
                                        prior_param, dim)
        for i in range(3):
            var_param0 = np.random.randn(approx.var_param_dim)
            var_param1 = np.random.randn(approx.var_param_dim)
            _test_family(approx, var_param0, var_param1, [0])
    # TODO: check behavior in corner cases
