import autograd.numpy as np
from autograd.scipy import stats
from autograd.scipy import special
from autograd.numpy import linalg


# See: https://github.com/scipy/scipy/blob/master/scipy/stats/_multivariate.py
def multivariate_t_logpdf(x, m, S, df=np.inf):
    """calculate log pdf for each value

    Parameters
    ----------
    x : array_like, shape=(n_samples, n_features)

    m : array_like, shape=(n_features,)

    S : array_like, shape=(n_features, n_features)
        covariance  matrix
    df : int or float
        degrees of freedom
    """
    #m = np.asarray(m)
    d = m.shape[-1]
    if df == np.inf:
        return stats.multivariate_normal.logpdf(x, m, S)
    #psd = _PSD(S)
    s, u = linalg.eigh(S)
    eps = 1e-10
    s_pinv = np.array([0 if abs(x) <= eps else 1/x for x in s], dtype=float)
    U = np.multiply(u, np.sqrt(s_pinv))
    log_pdet = np.sum(np.log(s))

    log_pdf = special.gammaln(.5*(df + d)) - special.gammaln(.5*df) - .5*d * np.log(np.pi * df)
    log_pdf += -.5*log_pdet
    dev = x - m
    maha = np.sum(np.square(np.dot(dev, U)), axis=-1)
    log_pdf += -.5*(df + d) * np.log(1 + maha / df)
    return log_pdf
