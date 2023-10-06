import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
import pytest
from viabel.approximations import MFGaussian, MFStudentT
from viabel.objectives import AlphaDivergence, DISInclusiveKL, ExclusiveKL
from viabel.optimization import RMSProp


def _test_objective(objective_cls, num_mc_samples, **kwargs):
    np.random.seed(851)
    # mean = np.random.randn(1,dimension)
    # stdev = np.exp(np.random.randn(1,dimension))
    mean = np.array([1., -1.])[np.newaxis, :]
    stdev = np.array([2., 5.])[np.newaxis, :]

    def log_p(x):
        return jnp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)

    approx = MFStudentT(2, 100)
    objective = objective_cls(approx, log_p, num_mc_samples, **kwargs)
    # large number of MC samples and smaller epsilon and learning rate to ensure accuracy
    init_param = np.array([0, 0, 1, 1], dtype=np.float32)
    opt = RMSProp(0.1)
    opt_results = opt.optimize(400, objective, init_param)
    # iterate averaging introduces some bias, so use last iterate
    est_mean, est_cov = approx.mean_and_cov(opt_results['opt_param'])
    est_stdev = np.sqrt(np.diag(est_cov))
    print(est_stdev, stdev)
    np.testing.assert_almost_equal(mean.squeeze(), est_mean, decimal=1)
    np.testing.assert_almost_equal(stdev.squeeze(), est_stdev, decimal=1)


def test_ExclusiveKL():
    _test_objective(ExclusiveKL, 100)


def test_ExclusiveKL_path_deriv():
    _test_objective(ExclusiveKL, 100, use_path_deriv=True)


def test_ExclusiveKL_full_hessian():
    _test_objective(ExclusiveKL, 100, hessian_approx_method='full')


def test_ExclusiveKL_mean_cv():
    _test_objective(ExclusiveKL, 100, hessian_approx_method='mean_only')


def test_ExclusiveKL_loo_diag():
    _test_objective(ExclusiveKL, 100, hessian_approx_method='loo_diag_approx')


def test_ExclusiveKL_loo_direct():
    _test_objective(ExclusiveKL, 100, hessian_approx_method='loo_direct_approx')


def test_ExclusiveKL_full_hessian_path_deriv():
    _test_objective(ExclusiveKL, 100, use_path_deriv=True, hessian_approx_method='full')


def test_ExclusiveKL_mean_cv_path_deriv():
    _test_objective(ExclusiveKL, 100, use_path_deriv=True, hessian_approx_method='mean_only')


def test_ExclusiveKL_loo_diag_path_deriv():
    _test_objective(ExclusiveKL, 100, use_path_deriv=True, hessian_approx_method='loo_diag_approx')


def test_ExclusiveKL_loo_direct_path_deriv():
    _test_objective(ExclusiveKL, 100, use_path_deriv=True, hessian_approx_method='loo_direct_approx')


def test_invalid_hessian_approx_method():
    with pytest.raises(ValueError) as exception_info:
        _test_objective(ExclusiveKL, 100, hessian_approx_method='invalid method')
    assert str(
        exception_info.value) == "Name of approximation must be one of 'full', 'mean_only', 'loo_diag_approx', 'loo_direct_approx' or None object."


def test_DISInclusiveKL():
    dim = 2
    _test_objective(DISInclusiveKL, 100,
                    temper_prior=MFGaussian(dim),
                    temper_prior_params=np.concatenate([[0] * dim, [1] * dim]),
                    ess_target=50)


def test_AlphaDivergence():
    _test_objective(AlphaDivergence, 100, alpha=2)
