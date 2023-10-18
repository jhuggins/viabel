import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import pytest
from viabel.approximations import MFGaussian, MFStudentT
from viabel.objectives import AlphaDivergence, DISInclusiveKL, ExclusiveKL
from viabel.optimization import RMSProp

def _test_objective(objective_cls, num_mc_samples, **kwargs):
    rng_key = random.PRNGKey(851)

    mean = jnp.array([1., -1.])[jnp.newaxis, :]
    stdev = jnp.array([2., 5.])[jnp.newaxis, :]

    def log_p(x):
        return jnp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)

    approx = MFStudentT(2, 100)
    objective = objective_cls(approx, log_p, num_mc_samples, **kwargs)
    init_param = jnp.array([0, 0, 1, 1], dtype=jnp.float32)
    opt = RMSProp(0.1)
    opt_results = opt.optimize(400, objective, init_param)
    est_mean, est_cov = approx.mean_and_cov(opt_results['opt_param'])
    est_stdev = jnp.sqrt(jnp.diag(est_cov))
    print(est_stdev, stdev)
    
    assert jnp.allclose(mean.squeeze(), est_mean, atol=1e-1)
    assert jnp.allclose(stdev.squeeze(), est_stdev, atol=1e-1)



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
                    temper_prior_params=jnp.concatenate([jnp.array([0] * dim), jnp.array([1] * dim)], axis=0),
                    ess_target=50)


def test_AlphaDivergence():
    _test_objective(AlphaDivergence, 100, alpha=2)
