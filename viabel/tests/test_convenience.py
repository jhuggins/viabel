import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

from viabel import convenience
from viabel.models import Model
import nest_asyncio
nest_asyncio.apply()

def test_bbvi():
    np.random.seed(851)
    mean = np.array([3., -4.])[np.newaxis, :]
    stdev = np.array([2., 5.])[np.newaxis, :]

    def log_p(x):
        return jnp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)
    # use large number of MC samples to ensure accuracy
    for adaptive in [True, False]:
        if adaptive:
            for fixed_lr in [True, False]:
                results = convenience.bbvi(2, log_density=log_p, num_mc_samples=1000,
                                           RAABBVI_kwargs=dict(mcse_threshold=.005,accuracy_threshold=.005),
                                           FASO_kwargs=dict(mcse_threshold=.005),
                                           adaptive=adaptive, fixed_lr=fixed_lr, n_iters=30000)
                est_mean, est_cov = results['objective'].approx.mean_and_cov(results['opt_param'])
                est_stdev = np.sqrt(np.diag(est_cov))
                jnp.allclose(mean.squeeze(), est_mean)
                jnp.allclose(stdev.squeeze(), est_stdev)
        else:
            results = convenience.bbvi(2, log_density=log_p, num_mc_samples=50,
                                           RAABBVI_kwargs=dict(mcse_threshold=.005,accuracy_threshold=.005),
                                           FASO_kwargs=dict(mcse_threshold=.005),
                                           adaptive=adaptive, fixed_lr=True, n_iters=30000)
            est_mean, est_cov = results['objective'].approx.mean_and_cov(results['opt_param'])
            est_stdev = np.sqrt(np.diag(est_cov))
            jnp.allclose(mean.squeeze(), est_mean)
            jnp.allclose(stdev.squeeze(), est_stdev)
                                           
    with pytest.raises(ValueError):
        convenience.bbvi(2)
    with pytest.raises(ValueError):
        convenience.bbvi(2, objective=True, fit=True)
    with pytest.raises(ValueError):
        convenience.bbvi(2, log_density=True, fit=True)
    with pytest.raises(ValueError):
        convenience.bbvi(2, objective=True, log_density=True)


def test_vi_diagnostics():
    np.random.seed(153)

    def log_p(x):
        return jnp.sum(norm.logpdf(x), axis=1)
    results = convenience.bbvi(2, log_density=log_p, num_mc_samples=100)
    diagnostics = convenience.vi_diagnostics(results['opt_param'],
                                             objective=results['objective'])
    assert diagnostics['khat'] < .1
    assert diagnostics['d2'] < 0.1

    def log_p2(x):
        return jnp.sum(norm.logpdf(x, scale=3), axis=1)
    model2 = Model(log_p2)
    diagnostics2 = convenience.vi_diagnostics(results['opt_param'],
                                              approx=results['objective'].approx,
                                              model=model2)
    assert diagnostics2['khat'] > 0.7
    assert 'd2' not in diagnostics2

    def log_p3(x):
        return jnp.sum(norm.logpdf(x, scale=.5), axis=1)
    model3 = Model(log_p3)
    diagnostics3 = convenience.vi_diagnostics(results['opt_param'],
                                              approx=results['objective'].approx,
                                              model=model3)
    print(diagnostics3)
    assert diagnostics3['khat'] < 0  # weights are bounded
    assert diagnostics3['d2'] > 2
