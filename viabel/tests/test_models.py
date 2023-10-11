import pickle

import jax.numpy as jnp
import numpy as np
import bridgestan as bs
import pytest
from jax.scipy.stats import norm
from jax.test_util import check_vjp
from jax import vjp

from viabel import models

def _test_model(m, x, supports_tempering, supports_constrain):
    #check_vjp(m, (x,), modes=['rev'], order=2)
    #check_vjp(m, x[0])
    assert supports_tempering == m.supports_tempering
    if supports_tempering:  # pragma: no cover
        m.set_inverse_temperature(.5)
    else:
        with pytest.raises(NotImplementedError):
            m.set_inverse_temperature(.5)
    if supports_constrain:
        m.constrain(x[0]) == supports_constrain
    else:
        with pytest.raises(NotImplementedError):
            m.constrain(x[0])




def test_Model():
    mean = np.array([1., -1.])[np.newaxis, :]
    stdev = np.array([2., 5.])[np.newaxis, :]

    def log_p(x):
        return jnp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)
    model = models.Model(log_p)
    x = 4 * np.random.randn(10, 2)
    _test_model(model, x, False, False)


def test_StanModel():
    
    regression_model = bs.StanModel.from_stan_file(stan_file='../viabel/data/test_model.stan', model_data='../viabel/data/test_model.data.json')


    fit = regression_model
    model = models.StanModel(fit)

    x = np.random.random(fit.param_unc_num())
    
    _,grad_expected = fit.log_density_gradient(x)
    _, vjpfun = vjp(model, x)
    grad = vjpfun(1.0)
    grad_actual = np.asarray(grad[0],dtype = np.float32)

    return np.testing.assert_allclose(grad_actual, grad_expected)
    #_test_model(model, x, False, dict(beta=x[0]))
