
import autograd.numpy as np

def compute_posterior_moments(prior_mean, prior_covariance, noise_variance, x, y):
    prior_L = np.linalg.cholesky(prior_covariance)
    inv_L = np.linalg.inv(prior_L)
    prior_precision = inv_L.T@inv_L
    S_precision = prior_precision + x.T @ x *(1. / noise_variance)
    a = np.linalg.cholesky(S_precision)
    tmp1 = np.linalg.inv(a)
    S = tmp1.T @ tmp1
    post_S=S
    post_mu = prior_precision@prior_mean + (1./noise_variance)* x.T@ y
    post_mu = post_S@ post_mu
    return post_mu, post_S

def get_samples_and_log_weights(logdensity, var_family, var_param, n_samples):
    samples = var_family.sample(var_param, n_samples)
    log_weights = logdensity(samples) - var_family.logdensity(samples, var_param)
    return samples, log_weights