from abc import ABC, abstractmethod
import tqdm
import autograd.numpy as np
from viabel.approximations import MFGaussian
from viabel._mc_diagnostics import MCSE, R_hat_convergence_check
from viabel._utils import StanModel_cache


__all__ = [
    'Optimizer',
    'StochasticGradientOptimizer',
    'RMSProp',
    'AdaGrad',
    'RAABBVI'
]


class Optimizer(ABC):
    """An abstract class for optimization
    """

    @abstractmethod
    def optimize(self, n_iters, objective, init_param, **kwargs):
        """
        Parameters
        ----------
        n_iters : `int`
            Number of iterations of the optimization
        objective : `function`
            Function for constructing the objective and gradient function
        init_param : `numpy.ndarray`, shape(var_param_dim,)
            Initial values of the variational parameters
        **kwargs
            Keyword arguments to pass (example: smoothed_prop)

        Returns
        ----------
        Dictionary
        smoothed_opt_param : `numpy.ndarray`, shape(var_param_dim,)
            Iterate averaged estimated variational parameters
        variational_param_history : `numpy.ndarray`, shape(n_iters, var_param_dim)\
            Estimated variational parameters over all iterations
        value_history : `numpy.ndarray`, shape(n_iters,)
            Estimated loss (ELBO) over all iterations
        """
        pass


class StochasticGradientOptimizer(Optimizer):
    """An abstract class of descent direction and a subclass of Optimizer
    """
    def __init__(self, learning_rate):
        """
        Parameters
        -----------
        learning_rate : `float`
            Tuning parameter that determines the step size
        """
        self._learning_rate = learning_rate

    def optimize(self, n_iters, objective, init_param, smoothed_prop=0.2):
        variational_param = init_param.copy()
        smoothing_window = int(n_iters*smoothed_prop)
        history = None
        value_history = []
        variational_param_history = []
        descent_dir_history = []
        with tqdm.trange(n_iters) as progress:
            try:
                for t in progress:
                    object_val, object_grad = objective(variational_param)
                    value_history.append(object_val)
                    descent_dir, history = self.descent_direction(object_grad, history)
                    variational_param -= self._learning_rate * descent_dir
                    variational_param_history.append(variational_param.copy())
                    descent_dir_history.append(descent_dir)
                    if t % 10 == 0:
                        avg_loss = np.mean(value_history[max(0, t - 1000):t + 1])
                        progress.set_description(
                            'average loss = {:,.5g}'.format(avg_loss))
            except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                progress.close()
        variational_param_history = np.array(variational_param_history)
        variational_param_latter = variational_param_history[-smoothing_window:,:]
        smoothed_opt_param = np.mean(variational_param_latter, axis = 0)
        return dict(smoothed_opt_param = smoothed_opt_param,
                    variational_param_history = variational_param_history,
                    value_history = np.array(value_history))

    @abstractmethod
    def descent_direction(self, grad, history):
        """
        Parameters
        -----------
        learning_rate : `float`
            Tuning parameter that determines the step size
        beta : `float`, optional
            Discounting factor for the history. The default value is 0.9
        jitter : `float`, optional
            Smoothing term that avoids division by zero

        Returns
        ----------
        descent_dir : `numpy.ndarray`, shape(var_param_dim,)
            Descent direction of the optimization algorithm
        history
            Additional information needed for computing future descent directions.
        """
        pass


class RMSProp(StochasticGradientOptimizer):
    """RMSprop optimization method
    """
    def __init__(self, learning_rate, beta=0.9, jitter=1e-8):
        self._beta = beta
        self._jitter = jitter
        super().__init__(learning_rate)

    def descent_direction(self, grad, history):
        if history is None:
            history  = grad**2
        history = history*self._beta + (1.-self._beta)*grad**2
        descent_dir = grad / np.sqrt(self._jitter+history)
        return (descent_dir, history)

class RMSProp_modified(StochasticGradientOptimizer):
    """RMSprop optimization method
    """
    def __init__(self, learning_rate, beta=0.9, jitter=1e-8):
        self._beta = beta
        self._jitter = jitter
        super().__init__(learning_rate)

    def descent_direction(self, grad, history):
        if history is None:
            history  = grad**2
        descent_dir = grad / np.sqrt(self._jitter+history)
        history = history*self._beta + (1.-self._beta)*grad**2
        return (descent_dir, history)


class AdaGrad(StochasticGradientOptimizer):
    """Adagrad optimization method
    """
    def __init__(self, learning_rate, jitter=1e-8):
        self._jitter = jitter
        super().__init__(learning_rate)

    def descent_direction(self, grad, history):
        if history is None:
            history = 0
        history = history + grad**2
        descent_dir = grad / np.sqrt(self._jitter+history)
        return (descent_dir, history)


class RAABBVI(Optimizer):
    """A robust, automated, and accurate BBVI optimizer

    Parameters
    ----------
    sgo : `class`
        A subclass of StochasticGradientOptimizer
    dim : `int`
        dimension of the underlying parameter space
    rho : `float`, optional
        Learning rate reducing factor. The default is 0.5
    eps : `float`, optional
        Threshold to determine the stopping iterations. The default is 0.01.
    tol :`float` optional
        Tolerance level to determine MCSE of variational estimates. The default
        is 0.1.
    W_min : `int`, optional
        Minimum window size for checking convergence. The default is 200.
    k_check : `int`, optional
        Frequency with which to check convergence. The default is `W_min`.
    """
    def __init__(self, sgo, dim, rho=0.5, eps=1e-3, tol=0.1, W_min=200, k_check=200):
        if not isinstance(sgo, StochasticGradientOptimizer):
            raise ValueError('sgo must be a subclass of StochasticGradientOptimizer')
        self._sgo = sgo
        self._dim = dim
        self._rho = rho
        self._eps = eps
        self._tol = tol
        self._W_min = W_min
        self._k_check = W_min if k_check is None else k_check

    def weighted_linear_regression(self, model, y, x, s=4, a=0.25):
        """
        Parameters
        ----------
        model : `pystan model`
            Pystan model to conduct the sampling
        y : `numpy_ndarray`
            Response variable: log(SKL)
        x : `numpy_ndarray`
            Predictor variable: log(\gamma) + log((1-\rho)/\rho)

        Returns
        -------
        kappa : `float`
            Power
        c : `float`
            Constant 
        """
        def initfun(kappa, log_c, sigma, chain_id=1):
            return dict(kappa=kappa, log_c=log_c, sigma=sigma)
        N = len(y)
        w = 1/(1 + np.arange(N)[::-1]**2/s**2)**a
        n_chains = 4
        data = dict(N=N, y=y, x=x, rho=self._rho, w=np.array(w))
        init = [initfun(0.5, 100, 5, chain_id=i) for i in range(n_chains) ]
        fit = model.sampling(data=data, init=init, iter=1000, chains=n_chains,
                             control=dict(adapt_delta=0.95))
        kappa = np.mean(fit['kappa'])
        c = np.exp(np.mean(fit['log_c']))
        return kappa, c


    def optimize(self, n_iters, objective, init_param):
        """
        Parameters
        ----------
        n_iters : `int`
            Number of iterations of the optimization
        objective: `function`
            Function for constructing the objective and gradient function
        init_param : `numpy.ndarray`, shape(var_param_dim,)
            Initial values of the variational parameters
        int_learning_rate: `float`
            Initial learning rate of optimization (step size to reach the (local) minimum)

        Returns
        ----------
        Dictionary
            smoothed_opt_param : `numpy.ndarray`, shape(var_param_dim,)
                 Iterate averaged estimated variational parameters
            variational_param_history : `numpy.ndarray`, shape(n_iters, var_param_dim)
                Estimated variational parameters over all iterations
            value_history : `numpy.ndarray`, shape(n_iters,)
                 Estimated loss (ELBO) over all iterations
        """
        if not objective.approx.supports_kl:
            print('Approximation does not support KL. Using base stochastic'
                  ' optimization algorithm instead.', flush=True)
            return self._sgo.optimize(n_iters, objective, init_param)

        k0 = 0
        k_conv = None # Iteration number when reached convergence
        history = None # Information needed to compute descent direction in optimization algorithm
        success = False
        learning_rate = self._sgo._learning_rate
        model = StanModel_cache(model_name='weighted_lin_regression')
        variational_param = init_param.copy()
        variational_param_mean = init_param.copy()
        variational_param_history = []
        value_history = []
        SKL_history = []
        learn_rate_hist = []
        stopped = False
        with tqdm.trange(n_iters) as progress:
            try:
                for k in progress:
                    object_val, object_grad = objective(variational_param)
                    value_history.append(object_val)
                    descent_dir, history = self._sgo.descent_direction(object_grad, history)
                    variational_param -= learning_rate * descent_dir
                    variational_param_history.append(variational_param.copy())

                    # when convergence has not been reached check if the
                    # recheck for convergence using R_hat approach
                    if k_conv is None and k % self._k_check == 0:
                        W_upper = int(0.95*k)
                        if W_upper > self._W_min:
                            windows = np.linspace(self._W_min, W_upper, num=5, dtype=int)
                            W = R_hat_convergence_check(variational_param_history, windows)
                            if W is not None:
                                k_conv = k-W-k0
                                W_check = W  # immediately check MCSE

                    # Once convergence has been reached compute the MCS
                    if k_conv is not None and k - k_conv - k0 == W_check:
                        W = W_check
                        if isinstance(objective.approx, MFGaussian):
                            # For MF Gaussian, use MCSE(mu/sigma,log_sigma)
                            var_param_log_sd = np.array(variational_param_history)[-W:,-self._dim:]
                            var_param_mu = np.array(variational_param_history)[-W:,self._dim:]
                            var_param_mu_stand = var_param_mu/np.exp(var_param_log_sd)
                            var_param_new = np.concatenate((var_param_mu_stand,var_param_log_sd),axis=1)
                            mcse = MCSE(var_param_new)
                        else:
                            var_param = np.array(variational_param_history)[-W:,:]
                            mcse = MCSE(var_param)
                        if np.mean(mcse) < self._tol:
                            success = True
                            learning_rate = self._rho * learning_rate
                            learn_rate_hist.append(learning_rate)
                            variational_param_mean_prev = variational_param_mean
                            variational_param_mean = np.mean(np.array(variational_param_history[-W:]), axis=0)
                            variational_param = variational_param_mean
                            if len(learn_rate_hist) > 1:
                                SKL = (objective.approx.kl(variational_param_mean_prev, variational_param_mean) +
                                       objective.approx.kl(variational_param_mean, variational_param_mean_prev))
                                SKL_history.append(SKL)
                            k_conv = None
                            k0 = k
                            # Conduct weighted linear regression to estimate parameters
                            # for the stopping rule
                            if len(SKL_history) > 0:
                                y = np.log(SKL_history)
                                x = np.log(learn_rate_hist[:-1])
                                kappa, c = self.weighted_linear_regression(model, y, x)
                                if (c * learn_rate_hist[-2]**(2*kappa) < self._eps):
                                    stopped = True
                                    break
                        else:
                            W_check = int(1.25*W_check)


                    if k % 10 == 0:
                        avg_loss = np.mean(value_history[max(0, k - 1000):k + 1])
                        progress.set_description(
                            'average loss = {:,.5g} | learning rate = {:,.5g} |'.format(avg_loss, learning_rate))
            except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                progress.close()
        if not success:
            UserWarning('Failed to converge')
        if stopped:
            print('Stopping rule reached at iteration', k)
        if k - k0 > self._W_min:
            variational_param_mean = np.mean(np.array(variational_param_history[-self._W_min:]), axis = 0)
        return dict(smoothed_opt_param = variational_param_mean,
                    variational_param_history = variational_param_history,
                    value_history = np.array(value_history))
