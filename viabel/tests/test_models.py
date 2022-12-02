import pickle

import autograd.numpy as anp
import numpy as np
import pytest
import stan
from autograd.scipy.stats import norm
from autograd.test_util import check_vjp

from viabel import models


def _test_model(m, x, supports_tempering, supports_constrain):
    check_vjp(m, x)
    check_vjp(m, x[0])
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


test_model = """data {
  int<lower=0> N;   // number of observations
  matrix[N, 2] x;   // predictor matrix
  vector[N] y;      // outcome vector
  real<lower=1> df; // degrees of freedom
}

parameters {
  vector[2] beta;       // coefficients for predictors
}

model {
  beta ~ normal(0, 10);
  y ~ student_t(df, x * beta, 1);  // likelihood
}"""


def test_Model():
    mean = np.array([1., -1.])[np.newaxis, :]
    stdev = np.array([2., 5.])[np.newaxis, :]

    def log_p(x):
        return anp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)
    model = models.Model(log_p)
    x = 4 * np.random.randn(10, 2)
    _test_model(model, x, False, False)


def test_StanModel():
    np.random.seed(5039)
    beta_gen = np.array([-2, 1])
    N = 25
    x = np.random.randn(N, 2).dot(np.array([[1, .75], [.75, 1]]))
    y_raw = x.dot(beta_gen) + np.random.standard_t(40, N)
    y = y_raw - np.mean(y_raw)
    data = dict(N=N, x=x, y=y, df=40)

    compiled_model_file = 'robust_reg_model.pkl'
    try:
        with open(compiled_model_file, 'rb') as f:
            regression_model = pickle.load(f)
    except BaseException:  # pragma: no cover
        regression_model = stan.build(program_code=test_model, data=data)
        with open('robust_reg_model.pkl', 'wb') as f:
            pickle.dump(regression_model, f)

    model = models.StanModel(regression_model)

    x = 4 * np.random.randn(10, 2)
    _test_model(model, x, False, dict(beta=x[0]))
