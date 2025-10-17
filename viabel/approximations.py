from abc import ABC, abstractmethod
import scipy

import jax.numpy as np
import numpy.random as npr
import jax.scipy.stats.norm as norm
import jax.scipy.stats.t as t_dist
from jax import jvp
from viabel.function_patterns import FlattenFunctionInput
from viabel.patterns import NumericArrayPattern, NumericVectorPattern, PatternDict, PSDSymmetricMatrixPattern

from viabel._distributions import multivariate_t_logpdf

__all__ = [
    'ApproximationFamily',
    'MFGaussian',
    'MFStudentT',
    'MultivariateT',
    'NeuralNet',
    'NVPFlow',
    'LRGaussian'
]


class ApproximationFamily(ABC):
    """An abstract class for an variational approximation family.

    See derived classes for examples.
    """

    def __init__(self, dim, var_param_dim, supports_entropy, supports_kl):
        """
        Parameters
        ----------
        dim : `int`
            The dimension of the space the distributions in the approximation family are
            defined on.
        var_param_dim : `int`
            The dimension of the variational parameter.
        supports_entropy : `bool`
            Whether the approximation family supports closed-form entropy computation.
        supports_kl : `bool`
            Whether the approximation family supports closed-form KL divergence
            computation.
        """
        self._dim = dim
        self._var_param_dim = var_param_dim
        self._supports_entropy = supports_entropy
        self._supports_kl = supports_kl

    def init_param(self):
        """A variational parameter to use for initialization.

        Returns
        -------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
        """
        return np.zeros(self.var_param_dim)

    @abstractmethod
    def sample(self, var_param, n_samples, seed=None):
        """Generate samples from the variational distribution

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        n_samples : `int`
            The number of samples to generate.

        Returns
        -------
        samples : `numpy.ndarray`, shape (n_samples, var_param_dim)
        """

    def entropy(self, var_param):
        """Compute entropy of variational distribution.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.

        Raises
        ------
        NotImplementedError
            If entropy computation is not supported."""
        if self.supports_entropy:
            return self._entropy(var_param)
        raise NotImplementedError()

    def _entropy(self, var_param):
        raise NotImplementedError()

    @property
    def supports_entropy(self):
        """Whether the approximation family supports closed-form entropy computation."""
        return self._supports_entropy

    def kl(self, var_param0, var_param1):
        """Compute the Kullback-Leibler (KL) divergence.

        Parameters
        ----------
        var_param0, var_param1 : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameters.

        Raises
        ------
        NotImplementedError
            If KL divergence computation is not supported.
        """
        if self.supports_kl:
            return self._kl(var_param0, var_param1)
        raise NotImplementedError()

    def _kl(self, var_param):
        raise NotImplementedError()

    @property
    def supports_kl(self):
        """Whether the approximation family supports closed-form KL divergence computation."""
        return self._supports_kl

    @abstractmethod
    def log_density(self, var_param, x):
        """The log density of the variational distribution.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        x : `numpy.ndarray`, shape (dim,)
            Value at which to evaluate the density."""

    @abstractmethod
    def mean_and_cov(self, var_param):
        """The mean and covariance of the variational distribution.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        """

    def pth_moment(self, var_param, p):
        """The absolute pth moment of the variational distribution.

        The absolute pth moment is given by :math:`\\mathbb{E}[|X|^p]`.

        Parameters
        ----------
        var_param : `numpy.ndarray`, shape (var_param_dim,)
            The variational parameter.
        p : `int`

        Raises
        ------
        ValueError
            If `p` value not supported"""
        if self.supports_pth_moment(p):
            return self._pth_moment(var_param, p)
        raise ValueError('p = {} is not a supported moment'.format(p))

    @abstractmethod
    def _pth_moment(self, var_param, p):
        """Get pth moment of the approximating distribution"""

    @abstractmethod
    def supports_pth_moment(self, p):
        """Whether analytically computing the pth moment is supported"""

    @property
    def dim(self):
        """Dimension of the space the distribution is defined on"""
        return self._dim

    @property
    def var_param_dim(self):
        """Dimension of the variational parameter"""
        return self._var_param_dim


def _get_mu_log_sigma_pattern(dim):
    ms_pattern = PatternDict(free_default=True)
    ms_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_pattern['log_sigma'] = NumericVectorPattern(length=dim)
    return ms_pattern


class MFGaussian(ApproximationFamily):
    """A mean-field Gaussian approximation family."""

    def __init__(self, dim, seed=1):
        """Create mean field Gaussian approximation family.

        Parameters
        ----------
        dim : `int`
            dimension of the underlying parameter space
        """
        self._rs = npr.RandomState(seed)
        self._pattern = _get_mu_log_sigma_pattern(dim)
        super().__init__(dim, self._pattern.flat_length(True), True, True)

    def init_param(self):
        init_param_dict = dict(mu=np.zeros(self.dim),
                               log_sigma=2 * np.ones(self.dim))
        return self._pattern.flatten(init_param_dict)

    def sample(self, var_param, n_samples, seed=None):
        my_rs = self._rs if seed is None else npr.RandomState(seed)
        param_dict = self._pattern.fold(var_param)
        return param_dict['mu'] + np.exp(param_dict['log_sigma']) * \
            my_rs.randn(n_samples, self.dim)

    def _entropy(self, var_param):
        param_dict = self._pattern.fold(var_param)
        return 0.5 * self.dim * (1.0 + np.log(2 * np.pi)) + np.sum(param_dict['log_sigma'])

    def _kl(self, var_param0, var_param1):
        param_dict0 = self._pattern.fold(var_param0)
        param_dict1 = self._pattern.fold(var_param1)
        mean_diff = param_dict0['mu'] - param_dict1['mu']
        log_stdev_diff = param_dict0['log_sigma'] - param_dict1['log_sigma']
        return .5 * np.sum(np.exp(2 * log_stdev_diff)
                           + mean_diff ** 2 / np.exp(2 * param_dict1['log_sigma'])
                           - 2 * log_stdev_diff - 1)

    def log_density(self, var_param, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        param_dict = self._pattern.fold(var_param)
        return np.sum(norm.logpdf(x, param_dict['mu'], 
                                  np.exp(param_dict['log_sigma'])), axis=-1)

    def mean_and_cov(self, var_param):
        param_dict = self._pattern.fold(var_param)
        return param_dict['mu'], np.diag(np.exp(2 * param_dict['log_sigma']))

    def _pth_moment(self, var_param, p):
        param_dict = self._pattern.fold(var_param)
        vars = np.exp(2 * param_dict['log_sigma'])
        if p == 2:
            return np.sum(vars)
        else:  # p == 4
            return 2 * np.sum(vars**2) + np.sum(vars)**2

    def supports_pth_moment(self, p):
        return p in [2, 4]


class MFStudentT(ApproximationFamily):
    """A mean-field Student's t approximation family."""

    def __init__(self, dim, df, seed=1):
        if df <= 2:
            raise ValueError('df must be greater than 2')
        self._df = df
        self._rs = npr.RandomState(seed)
        self._pattern = _get_mu_log_sigma_pattern(dim)
        super().__init__(dim, self._pattern.flat_length(True), True, False)

    def init_param(self):
        init_param_dict = dict(mu=np.zeros(self.dim),
                               log_sigma=2 * np.ones(self.dim))
        return self._pattern.flatten(init_param_dict)

    def sample(self, var_param, n_samples, seed=None):
        my_rs = self._rs if seed is None else npr.RandomState(seed)
        param_dict = self._pattern.fold(var_param)
        return param_dict['mu'] + np.exp(param_dict['log_sigma']) * \
            my_rs.standard_t(self.df, size=(n_samples, self.dim))

    def entropy(self, var_param):
        # ignore terms that depend only on df
        param_dict = self._pattern.fold(var_param)
        return np.sum(param_dict['log_sigma'])

    def log_density(self, var_param, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        param_dict = self._pattern.fold(var_param)
        return np.sum(t_dist.logpdf(x, self.df, param_dict['mu'], np.exp(
            param_dict['log_sigma'])), axis=-1)

    def mean_and_cov(self, var_param):
        param_dict = self._pattern.fold(var_param)
        df = self.df
        cov = df / (df - 2) * np.diag(np.exp(2 * param_dict['log_sigma']))
        return param_dict['mu'], cov

    def _pth_moment(self, var_param, p):
        df = self.df
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = self._pattern.fold(var_param)
        scales = np.exp(param_dict['log_sigma'])
        c = df / (df - 2)
        if p == 2:
            return c * np.sum(scales**2)
        else:  # p == 4
            return c**2 * (2 * (df - 1) / (df - 4) * np.sum(scales**4) + np.sum(scales**2)**2)

    def supports_pth_moment(self, p):
        return p in [2, 4] and p < self.df

    @property
    def df(self):
        """Degrees of freedom."""
        return self._df


def _get_mu_sigma_pattern(dim):
    ms_pattern = PatternDict(free_default=True)
    ms_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_pattern['Sigma'] = PSDSymmetricMatrixPattern(size=dim)
    return ms_pattern

def sqrtm(matrix):
    L = scipy.linalg.cholesky(matrix)
    return L
    
class MultivariateT(ApproximationFamily):
    """A full-rank multivariate t approximation family."""

    def __init__(self, dim, df, seed=1):
        if df <= 2:
            raise ValueError('df must be greater than 2')
        self._df = df
        self._rs = npr.RandomState(seed)
        self._pattern = _get_mu_sigma_pattern(dim)
        self._log_density = FlattenFunctionInput(
            lambda param_dict, x: multivariate_t_logpdf(
                x, param_dict['mu'], param_dict['Sigma'], df),
            patterns=self._pattern, free=True, argnums=0)
        super().__init__(dim, self._pattern.flat_length(True), True, False)

    def init_param(self):
        init_param_dict = dict(mu=np.zeros(self.dim),
                               Sigma=10 * np.eye(self.dim))
        return self._pattern.flatten(init_param_dict)

    def sample(self, var_param, n_samples, seed=None):
        my_rs = self._rs if seed is None else npr.RandomState(seed)
        df = self.df
        s = np.sqrt(my_rs.chisquare(df, n_samples) / df)
        param_dict = self._pattern.fold(var_param)
        z = my_rs.randn(n_samples, self.dim)
        sqrtSigma = sqrtm(param_dict['Sigma'])
        return param_dict['mu'] + np.dot(z, sqrtSigma) / s[:, np.newaxis]

    def entropy(self, var_param):
        # ignore terms that depend only on df
        param_dict = self._pattern.fold(var_param)
        return .5 * np.log(np.linalg.det(param_dict['Sigma']))

    def log_density(self, var_param, x):
        return self._log_density(var_param, x)

    def mean_and_cov(self, var_param):
        param_dict = self._pattern.fold(var_param)
        df = self.df
        return param_dict['mu'], df / (df - 2.) * param_dict['Sigma']

    def _pth_moment(self, var_param, p):
        df = self.df
        if df <= p:
            raise ValueError('df must be greater than p')
        param_dict = self._pattern.fold(var_param)
        sq_scales = np.linalg.eigvalsh(param_dict['Sigma'])
        c = df / (df - 2)
        if p == 2:
            return c * np.sum(sq_scales)
        else:  # p == 4
            return c**2 * (2 * (df - 1) / (df - 4) * np.sum(sq_scales**2) + np.sum(sq_scales)**2)

    def supports_pth_moment(self, p):
        return p in [2, 4] and p < self.df

    @property
    def df(self):
        """Degrees of freedom."""
        return self._df


class NeuralNet(ApproximationFamily):
    def __init__(self, layers_shapes, nonlinearity=np.tanh, last=np.tanh,
                 mc_samples=10000, seed=1):
        """
        Parameters
        ----------
        layers_shapes : `list of int`
            The hidden layers dimensions.
        nonlinearity : `function`
            Non linear function to apply after each layer except the last layer.
        last : `function`
            Non linear function to apply after the last layer.
        mc_samples : `int`
            Number of samples to draw internally for computing mean and cov.
        seed : `int`
            Internal seed representation.
        """
        self._pattern = PatternDict(free_default=True)
        self.mc_samples = mc_samples
        self._layers = len(layers_shapes)
        self._nonlinearity = nonlinearity
        self._last = last
        self._rs = npr.RandomState(seed)
        self.input_dim = layers_shapes[0][0]
        for layer_id in range(len(layers_shapes)):
            self._pattern[str(layer_id)] = NumericArrayPattern(shape=layers_shapes[layer_id])
            self._pattern[str(layer_id) + "_b"] = NumericArrayPattern(
                shape=[layers_shapes[layer_id][1]])

        super().__init__(layers_shapes[-1][-1], self._pattern.flat_length(True), False, False)

    def forward(self, var_param, x):
        log_det_J = np.zeros(x.shape[0])
        for layer_id in range(self._layers):
            W = var_param[str(layer_id)]
            b = var_param[str(layer_id) + "_b"]
            if layer_id + 1 == self._layers:
                x = self._last(np.dot(x, W) + b)
                _, elementwise_gradient_last = jvp(self._last, (x,), (np.ones_like(x),))

                log_det_J += np.log(np.abs(np.dot(elementwise_gradient_last, W.T).sum(axis=1)))
            else:
                x = self._nonlinearity(np.dot(x, W) + b)
                _, elementwise_gradient = jvp(self._nonlinearity, (x,), (np.ones_like(x),))

                log_det_J += np.log(np.abs(np.dot(elementwise_gradient, W.T).sum(axis=1)))
        return x, log_det_J

    def sample(self, var_param, n_samples):
        z_0 = npr.multivariate_normal(mean=[0] * self.input_dim,
                                      cov=np.identity(self.input_dim),
                                      size=n_samples)
        z_k, _ = self.forward(var_param, z_0)
        return z_k

    def log_density(self, var_param, x):
        raise NotImplementedError

    def mean_and_cov(self, var_param):
        samples = self.sample(var_param, self.mc_samples)
        return np.mean(samples, axis=0), np.cov(samples.T)

    def _pth_moment(self, var_param, p):
        raise NotImplementedError

    def supports_pth_moment(self, p):
        return False


class NVPFlow(ApproximationFamily):
    def __init__(self, layers_t, layers_s, mask, prior, prior_param, dim, activation=np.tanh,
                 seed=1, mc_samples=10000):
        """
        Parameters
        ----------
        layers_t : `list of int`
            The hidden layers dimensions for the translation operator.
        layers_s : `list of int`
            The hidden layers dimensions for the scaling operator.
        mask : `mask int`
            Mask to apply to the entry of each operator.
        prior : `ApproximationFamily`
            Prior for the latent space Z.
        prior_param : `numpy array`
            Parameter vector for the prior, must follow the same format as any
            variational family.
        dim : `int`
            Input dimension.
        seed : `int`
            Random seed for reproducibility.
        mc_samples : `int`
            Number of samples to draw internally for computing mean and cov.
        """
        assert len(layers_t) == len(layers_s)
        self.prior = prior
        self.prior_param = prior_param
        self.mc_samples = mc_samples
        self._dim = dim
        self._rs = npr.RandomState(seed)
        self.mask = mask
        self._pattern = PatternDict(free_default=True)
        self.t = [NeuralNet(layers_t, nonlinearity=activation, last=lambda x: x)
                  for _ in range(len(mask))]
        self.s = [NeuralNet(layers_s, nonlinearity=activation, last=np.tanh)
                  for _ in range(len(mask))]
        for layer_id in range(len(mask)):
            self._pattern[str(layer_id) + "t"] = self.t[layer_id]._pattern
            self._pattern[str(layer_id) + "s"] = self.s[layer_id]._pattern

        super().__init__(dim, self._pattern.flat_length(True), False, False)

    def g(self, var_param, z):
        """Inverse NVP flow.

        Parameters
        ----------
        var_param : `numpy array`
            Flat array of variational parameters.
        z : `numpy array`
            Latent space sample.
        """
        x = z
        param_dict = self._pattern.fold(var_param)
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i].forward(param_dict[str(i) + "s"], x_)[0] * (1 - self.mask[i])
            t = self.t[i].forward(param_dict[str(i) + "t"], x_)[0] * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * np.exp(s) + t)
        return x

    def f(self, var_param, x):
        """Forward NVP flow.

        Parameters
        ----------
        var_param : `numpy array`
            Flat array of variational parameters.
        x : `numpy array`
            Original space data.
        """
        param_dict = self._pattern.fold(var_param)
        log_det_J, z = np.zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i].forward(param_dict[str(i) + "s"], z_)[0] * (1 - self.mask[i])
            t = self.t[i].forward(param_dict[str(i) + "t"], z_)[0] * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * np.exp(-s) + z_
            log_det_J -= s.sum(axis=1)
        return z, log_det_J

    def log_density(self, var_param, x):
        z, logp = self.f(var_param, x)
        return self.prior.log_density(self.prior_param, z) + logp

    def sample(self, var_param, n_samples, seed=None):
        z_0 = self.prior.sample(self.prior_param, int(n_samples), seed=seed)
        z_k = self.g(var_param, z_0)
        return z_k

    def mean_and_cov(self, var_param):
        samples = self.sample(var_param, self.mc_samples)
        return np.mean(samples, axis=0), np.cov(samples.T)

    def _pth_moment(self, var_param, p):
        raise NotImplementedError

    def supports_pth_moment(self, p):
        return False
    
def _get_low_rank_mu_sigma_pattern(dim, k):
    ms_pattern = PatternDict(free_default=True)
    ms_pattern['mu'] = NumericVectorPattern(length=dim)
    ms_pattern['log_sigma'] = NumericVectorPattern(length=dim)
    ms_pattern['low_rank'] = NumericArrayPattern(shape=(dim, k))
    return ms_pattern

def _get_log_determinant(D, B):
    """Compute the determinant of the matrix B @ B.T + np.diag(D) using the matrix determinant lemma.

        Parameters
        ----------
        D : `numpy vector`
        diagnal component of covariance matrix.
        B : `numpy array`
        low rank component of covariance matrix.
    """
    log_det_D = 2*np.sum(D)
    _,log_det_IpDBBT = np.linalg.slogdet(np.eye(len(D)) + B @ B.T/np.exp(2*D[:,np.newaxis]))
    log_det_M = log_det_D + log_det_IpDBBT
    return log_det_M

def _get_trace(D0, B0, D1, B1):
    """Compute the trace of the product of the inverse of B1 @ B1.T + np.diag(D1) and B0 @ B0.T + np.diag(D0).

    Parameters
    ----------
    D0 : `numpy vector`
        Diagonal elements of sigma0.
    B0 : `numpy vector`
        Low-rank component of sigma0.
    D1 : `numpy array`
        Diagonal elements of sigma1.
    B1 : `numpy array`
        Low-rank component of sigma1.
    """
    
    I_B1D1B1 = np.eye(B1.shape[1]) + B1.T / D1 @ B1
    invD1_B1 = B1 / D1[:, np.newaxis]
    invD1_B1_I_B1D1B1_inv = np.linalg.solve(I_B1D1B1.T, invD1_B1.T).T
    product = invD1_B1_I_B1D1B1_inv @ (B1.T / D1)

    # Compute Tr(D0 * B1 * (I + B1^T * D1^-1 * B1)^-1 * B1^T * D1^-1)
    trace_product = np.trace(product * D0)

    # Compute Tr(D0 * D1^-1)
    trace_D0_invD1 = np.sum(D0 / D1)

    # Compute Tr(np.diag(D1)^(-1) * B0 @ B0.T)
    trace_invD1_B0B0T = np.trace(B0 @ B0.T / D1)

    # Compute Tr(np.diag(D1)^(-1) * B1 * (I + B1.T * D1^-1 * B1)^-1 * B1^T * D1^-1 * B0 @ B0.T)
    trace_extra_term = np.trace(product @ B0 @ B0.T)

    # Return Tr(sigma0 * sigma1^-1) = Tr(D0 * D1^-1) + Tr(np.diag(D1)^(-1) * B0 @ B0.T) - Tr(D0 * np.diag(D1)^(-1) * B1 * (I + B1^T * D1^-1 * B1)^-1 * B1^T * D1^-1) - Tr(np.diag(D1)^(-1) * B1 * (I + B1.T * D1^-1 * B1)^-1 * B1^T * D1^-1 * B0 @ B0.T)
    return trace_D0_invD1 + trace_invD1_B0B0T - trace_product - trace_extra_term


class LRGaussian(ApproximationFamily):
    """A low rank Gaussian approximation family."""

    def __init__(self, dim, seed=1, k=0):
        """Create multivariate Gaussian approximation family.

        Parameters
        ----------
        dim : `int`
            dimension of the underlying parameter space
        k : 'int'
            number of low rank
        """
        self._rs = npr.RandomState(seed)
        self._pattern = _get_low_rank_mu_sigma_pattern(dim, k)
        self._k = k
        super().__init__(dim, self._pattern.flat_length(True), True, True)

    def init_param(self):
        init_param_dict = dict(mu=np.zeros(self.dim),
                               log_sigma=np.ones(self.dim),
                               low_rank=self._rs.randn(self.dim,self._k))
        return self._pattern.flatten(init_param_dict)



    def sample(self, var_param, n_samples, seed=None):
        my_rs = self._rs if seed is None else npr.RandomState(seed)
        param_dict = self._pattern.fold(var_param)
        z = my_rs.randn(n_samples,self._k)
        epsilon = my_rs.randn(n_samples, self.dim)
        D_exp = np.exp(param_dict['log_sigma'])
        B = param_dict['low_rank']

        return param_dict['mu'] + np.dot(z,B.T) + D_exp * epsilon

    def _entropy(self, var_param):
        param_dict = self._pattern.fold(var_param)
        B = param_dict['low_rank']
        D = param_dict['log_sigma']
        sigma_log_det = _get_log_determinant(D, B)

        return 0.5 * self.dim * (np.log(2 * np.pi ) + 1) + 0.5 * sigma_log_det

    def _kl(self, var_param0, var_param1):
        param_dict0 = self._pattern.fold(var_param0)
        param_dict1 = self._pattern.fold(var_param1)
        mean_diff = param_dict0['mu'] - param_dict1['mu']

        B0 = param_dict0['low_rank']
        D0 = param_dict0['log_sigma']
        D0_exp = np.exp(2 * param_dict0['log_sigma'])
        sigma0_log_det = _get_log_determinant(D0, B0)

        B1 = param_dict1['low_rank']
        D1 = param_dict1['log_sigma']
        D1_exp = np.exp(2 * param_dict1['log_sigma'])
        D1_inv = np.diag(1/D1_exp)

        sigma1_log_det = _get_log_determinant(D1, B1)

        #By the Woodbury formula
        D1_invB = np.dot(D1_inv, B1)
        I_BDB = np.eye(self._k) + np.dot(B1.T, D1_invB)
        I_BDB_inv = np.linalg.solve(I_BDB, np.identity(I_BDB.shape[0]))
        Sigma_inv = D1_inv - D1_invB @ I_BDB_inv @ D1_invB.T


        #KL divergence
        sigma_log_diff = sigma1_log_det - sigma0_log_det
        mean_sigma = mean_diff.T @ Sigma_inv @ mean_diff
        sigma_trace = _get_trace(D0_exp, B0, D1_exp, B1)
        return .5 * (sigma_log_diff - self.dim + mean_sigma + sigma_trace)


    def log_density(self,var_param, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        param_dict = self._pattern.fold(var_param)
        mean = param_dict['mu']
        B = param_dict['low_rank']
        D = param_dict['log_sigma']
        D_exp = np.exp(2*D)
        D_inv = np.diag(1/D_exp)

        sigma_log_det = _get_log_determinant(D, B)

        # By the Woodbury formula
        D_invB = D_inv @ B
        I_BDB = np.eye(self._k) + np.dot(B.T, D_invB)
        I_BDB_inv = np.linalg.solve(I_BDB, np.identity(I_BDB.shape[0]))
        sigma_inv = D_inv - D_invB @ I_BDB_inv @ D_invB.T

        # Compute the log density of the multivariate Gaussian distribution for each row of X
        diff = x - mean
        log_p = -0.5 * (self.dim * np.log(2 * np.pi) + sigma_log_det + np.sum(diff @ sigma_inv * diff, axis=1))

        return log_p

    def mean_and_cov(self, var_param):
        param_dict = self._pattern.fold(var_param)
        B = param_dict['low_rank']
        D_exp = np.exp(2 * param_dict['log_sigma'])
        return param_dict['mu'], np.dot(B, B.T) + np.diag(D_exp)

    def _pth_moment(self, var_param, p):
        param_dict = self._pattern.fold(var_param)
        D_exp = np.exp(2*param_dict['log_sigma'])
        B = param_dict['low_rank']

        covariance = B @ B.T + np.diag(D_exp) #check later

        eigvals = np.linalg.eigvalsh(covariance)

        if p == 2:
            return np.sum(eigvals)
        else:  # p == 4
            return 2 * np.sum(eigvals ** 2) + np.sum(eigvals) ** 2


    def supports_pth_moment(self, p):
        return p in [2, 4]
