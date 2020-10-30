from viabel import family

import numpy as np
from scipy.stats import ttest_1samp, t


MC_SAMPLES = 1000000
test_size = 0.0001


def _test_entropy(vf, var_param, entropy_offset):
    entropy = vf.entropy(var_param) + entropy_offset
    log_probs = vf.logdensity(vf.sample(var_param, MC_SAMPLES), var_param)

    p_value = ttest_1samp(log_probs, -entropy)[1]
    assert p_value > test_size, "expected: {}, estimated: {}".format(entropy, -np.mean(log_probs))


def _test_kl(vf, var_param0, var_param1):
    kl = vf.kl(var_param0, var_param1)
    samples = vf.sample(var_param0, MC_SAMPLES)
    log_prob_diffs = vf.logdensity(samples, var_param0) - vf.logdensity(samples, var_param1)

    p_value = ttest_1samp(log_prob_diffs, kl)[1]
    assert p_value > test_size


def _test_mean_and_cov(vf, var_param):
    mean, cov = vf.mean_and_cov(var_param)
    second_moments = np.outer(mean, mean) + cov

    samples = vf.sample(var_param, MC_SAMPLES)
    samples_outer = np.einsum('ij,ik->ijk', samples, samples)

    mean_p_values = ttest_1samp(samples, mean, axis=0)[1]
    np.testing.assert_array_less(test_size, mean_p_values)

    second_moments_p_values = ttest_1samp(samples_outer, second_moments, axis=0)[1]
    np.testing.assert_array_less(test_size, second_moments_p_values)


def _test_pth_moment(vf, var_param, p):
    pth_moment = vf.pth_moment(p, var_param)

    samples = vf.sample(var_param, MC_SAMPLES)
    sample_mean = np.mean(samples, axis=0)
    sample_norms = np.linalg.norm(samples - sample_mean, axis=1, ord=2)

    p_value = ttest_1samp(sample_norms**p, pth_moment)[1]
    assert p_value > test_size, "expected: {}, estimated: {}".format(pth_moment, np.mean(sample_norms**p))


def _test_family(vf, var_param0, var_param1, entropy_offset=0,
                 exclude_kl=False):
    # These tests check that the variational family is defined self-consistently
    _test_entropy(vf, var_param0, entropy_offset)
    if not exclude_kl:
        _test_kl(vf, var_param0, var_param1)
    _test_mean_and_cov(vf, var_param0)
    _test_pth_moment(vf, var_param0, 2)
    _test_pth_moment(vf, var_param0, 4)
    # TODO: check behavior for invalid choice of p


def test_mf_gaussian_vf():
    np.random.seed(341)
    for dim in [1, 3]:
        vf = family.mean_field_gaussian_variational_family(dim)
        for i in range(3):
            var_param0 = np.random.randn(vf.var_param_dim)
            var_param1 = np.random.randn(vf.var_param_dim)
            _test_family(vf, var_param0, var_param1)
    # TODO: check behavior in corner cases


def test_mf_t_vf():
    np.random.seed(226)
    df = 20
    entropy_offset_1d = t.entropy(df)
    for dim in [1, 3]:
        entropy_offset = dim * entropy_offset_1d
        vf = family.mean_field_t_variational_family(dim, df)
        for i in range(3):
            var_param0 = np.random.randn(vf.var_param_dim)
            var_param1 = np.random.randn(vf.var_param_dim)
            _test_family(vf, var_param0, var_param1, entropy_offset, True)
    # TODO: check behavior in corner cases


def test_t_vf():
    np.random.seed(56)
    df = 100
    entropy_offset_1d = t.entropy(df)
    for dim in [1, 2]:
        entropy_offset = dim * entropy_offset_1d
        vf = family.t_variational_family(dim, df)
        for i in range(3):
            var_param0 = np.random.randn(vf.var_param_dim)
            var_param1 = np.random.randn(vf.var_param_dim)
            _test_family(vf, var_param0, var_param1, entropy_offset, True)
    # TODO: check behavior in corner cases
