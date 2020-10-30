from viabel import models

import pickle

import numpy as np
import pystan

from autograd.test_util import check_grads


def _test_model(m, x):
    check_grads(m, modes=['rev'], order=1)(x)


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
  y ~ student_t(df, x * beta, 1);  // likelihood"""


def test_stan_model():
    compiled_model_file = 'robust_reg_model.pkl'
    try:
        with open(compiled_model_file, 'rb') as f:
            regression_model = pickle.load(f)
    except:
        regression_model = pystan.StanModel(model_code=test_model,
                                            model_name='regression_model')
        with open('robust_reg_model.pkl', 'wb') as f:
            pickle.dump(sm, f)
    np.random.seed(5039)
    beta_gen = np.array([-2, 1])
    N = 25
    x = np.random.randn(N, 2).dot(np.array([[1,.75],[.75, 1]]))
    y_raw = x.dot(beta_gen) + np.random.standard_t(40, N)
    y = y_raw - np.mean(y_raw)

    data = dict(N=N, x=x, y=y, df=40)
    fit = regression_model.sampling(data=data, iter=10, thin=1, chains=1)
    stan_log_density = models.make_stan_log_density(fit)

    x = 4*np.random.randn(10,2)
    _test_model(stan_log_density, x)
