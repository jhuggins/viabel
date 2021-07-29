from abc import ABC, abstractmethod
import tqdm
import autograd.numpy as np
from viabel.approximations import MFGaussian
from viabel._mc_diagnostics import MCSE, R_hat_convergence_check
from viabel._utils import Timer

__all__ = [
    'Optimizer',
    'StochasticGradientOptimizer',
    'RMSProp',
    'AdaGrad',
    'WindowedAdaGrad',
    'FASO'
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
        results : `dict`
            Must contain at least `opt_param`, the estimate for the optimal
            variational parameter.
        """
        pass


class StochasticGradientOptimizer(Optimizer):
    """Stochastic gradient descent.
    """
    def __init__(self, learning_rate, *, weight_decay=0, iterate_avg_prop=0.2, diagnostics=False):
        """
        Parameters
        -----------
        learning_rate : `float`
            Tuning parameter that determines the step size
        weight_decay: `float`
            L2 regularization weight
        iterate_avg_prop : `float`
            Proportion of iterates to use for computing iterate average. `None`
            means no iterate averaging. The default is 0.2.
        diagnostics : `bool`, optional
            Record diagnostic information if `True`. The default is `False`.
        """
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        if iterate_avg_prop is not None and (iterate_avg_prop > 1.0 or
                                             iterate_avg_prop <= 0.0):
            raise ValueError('"iterate_avg_prop" must be None or between 0 and 1')
        self._iterate_avg_prop = iterate_avg_prop
        self._diagnostics = diagnostics
        self.reset_state()

    def reset_state(self):
        """Reset internal state of the optimizer"""
        pass

    def optimize(self, n_iters, objective, init_param):
        variational_param = init_param.copy()
        iap = self._iterate_avg_prop
        value_history = []
        variational_param_history = []
        descent_dir_history = []
        with tqdm.trange(n_iters) as progress:
            try:
                for k in progress:
                    # take step in descent direction
                    object_val, object_grad = objective(variational_param)
                    descent_dir = self.descent_direction(object_grad)
                    variational_param -= self._learning_rate * descent_dir
                    if variational_param.ndim == 2:
                        variational_param *= (1 - self._weight_decay)
                    # record state information
                    value_history.append(object_val)
                    if self._diagnostics or iap is not None:
                        variational_param_history.append(variational_param.copy())
                        if (iap is not None and
                            len(variational_param_history) > iap * k):
                            variational_param_history.pop(0)
                    if self._diagnostics:
                        descent_dir_history.append(descent_dir)
                    if k % 10 == 0:
                        avg_loss = np.mean(value_history[max(0, k - 1000):k+1])
                        progress.set_description(
                            'average loss = {:,.5g}'.format(avg_loss))
            except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                progress.close()
        results = dict(value_history=value_history)
        variational_param_history = np.array(variational_param_history)
        if iap is not None:
            window = max(1, int(k*iap))
            opt_param = np.mean(variational_param_history[-window:], axis=0)
        else:
            results['opt_param'] = variational_param.copy()
        if descent_dir_history is not None:
            results['descent_dir_history']  = descent_dir_history
        return dict(opt_param = opt_param,
                    variational_param_history = variational_param_history,
                    value_history = np.array(value_history),
                    descent_dir_history = np.array(descent_dir_history),
                    )

    def descent_direction(self, grad):
        """Compute descent direction for optimization.

        Default implementation returns ``grad``.

        Parameters
        -----------
        grad : `numpy.ndarray`, shape(var_param_dim,)
            (stochastic) gradient of the objective function

        Returns
        ----------
        descent_dir : `numpy.ndarray`, shape(var_param_dim,)
            Descent direction of the optimization algorithm
        """
        return grad


class RMSProp(StochasticGradientOptimizer):
    """RMSProp optimization method
    """
    def __init__(self, learning_rate, *, weight_decay=0, beta=0.9, jitter=1e-8,
                 diagnostics=False):
        self._beta = beta
        self._jitter = jitter
        super().__init__(learning_rate, weight_decay=weight_decay, diagnostics=diagnostics)

    def reset_state(self):
        self._avg_grad_sq = None

    def descent_direction(self, grad):
        if self._avg_grad_sq is None:
            avg_grad_sq = grad**2
        else:
            avg_grad_sq = self._avg_grad_sq
        avg_grad_sq *= self._beta
        avg_grad_sq += (1.-self._beta)*grad**2
        descent_dir = grad / np.sqrt(self._jitter+avg_grad_sq)
        self._avg_grad_sq = avg_grad_sq
        return descent_dir


class WindowedAdaGrad(StochasticGradientOptimizer):
    """Adam optimization method
    """
    def __init__(self, learning_rate, *, weight_decay=0, window_size=10, jitter=1e-8,
                 diagnostics=False):
        self._window_size = window_size
        self._jitter = jitter
        super().__init__(learning_rate, weight_decay=weight_decay, diagnostics=diagnostics)

    def reset_state(self):
        self._history = []

    def descent_direction(self, grad):
        self._history.append(grad**2)
        if len(self._history) > self._window_size:
            self._history.pop(0)
        mean_grad_squared = np.mean(self._history, axis=0)
        descent_dir = grad / np.sqrt(self._jitter+mean_grad_squared)
        return descent_dir


class AdaGrad(StochasticGradientOptimizer):
    """Adagrad optimization method
    """
    def __init__(self, learning_rate, *, weight_decay=0, jitter=1e-8, diagnostics=False):
        self._jitter = jitter
        super().__init__(learning_rate, weight_decay=weight_decay, diagnostics=diagnostics)

    def reset_state(self):
        self._sum_grad_sq = 0

    def descent_direction(self, grad):
        self._sum_grad_sq += grad**2
        descent_dir = grad / np.sqrt(self._jitter + self._sum_grad_sq)
        return descent_dir


class FASO(Optimizer):
    """Fixed-learning rate stochastic optimization meta-algorithm

    Parameters
    ----------
    sgo : `StochasticGradientOptimizer` object
        optimization method to use
    mcse_threshold : `float` optional
        Monte Carlo standard error threshold for convergence. The default is 0.1.
    W_min : `int`, optional
        Minimum window size for checking convergence. The default is 200.
    ESS_min : `int`, optional
        Minimum ESS for computing iterate average. Default is `W_min / 8`.
    k_check : `int`, optional
        Frequency with which to check convergence. The default is `W_min`.
    """
    def __init__(self, sgo, *, mcse_threshold=0.1, W_min=200, ESS_min=None,
                 k_check=None):
        if not isinstance(sgo, StochasticGradientOptimizer):
            raise ValueError('sgo must be a subclass of StochasticGradientOptimizer')
        self._sgo = sgo
        self._mcse_threshold = mcse_threshold
        self._W_min = W_min
        self._ESS_min = W_min // 8 if ESS_min is None else ESS_min
        self._k_check = W_min if k_check is None else k_check
        if mcse_threshold <= 0:
            raise ValueError('"mcse_threshold" must be greater than zero')
        if W_min <= 0:
            raise ValueError('"W_min" must be greater than zero')
        if self._k_check <= 0:
            raise ValueError('"k_check" must be greater than zero')
        if self._ESS_min <= 0:
            raise ValueError('"ESS_min" must be greater than zero')


    def optimize(self, n_iters, objective, init_param):
        dim = init_param.size
        diagnostics = self._sgo._diagnostics
        k_conv = None # Iteration number when reached convergence
        k_stopped = None # Iteration number when MCSE/ESS conditions met
        k_Rhat = None # Iteration number when R hat convergence criterion met
        learning_rate = self._sgo._learning_rate
        variational_param = init_param.copy()
        variational_param_history = []
        value_history = []
        descent_dir_history = []
        ess_and_mcse_k_history = []
        ess_history = []
        mcse_history = []
        iterate_average_k_history = []
        iterate_average_history = []
        iterate_average = variational_param.copy()
        if diagnostics:
            iterate_average_k_history.append(0)
            iterate_average_history.append(iterate_average)
        total_opt_time = 0  # total time spent on optimization
        stopped = False
        with tqdm.trange(n_iters) as progress:
            try:
                for k in progress:
                    # take step in descent direction
                    with Timer() as opt_timer:
                        object_val, object_grad = objective(variational_param)
                        value_history.append(object_val)
                        descent_dir = self._sgo.descent_direction(object_grad)
                        variational_param -= learning_rate * descent_dir
                        variational_param_history.append(variational_param.copy())
                        if diagnostics:
                            descent_dir_history.append(descent_dir)
                    total_opt_time += opt_timer.interval
                    # If convergence has not been reached then check for
                    # convergence using R hat
                    if k_conv is None and k % self._k_check == 0:
                        W_upper = int(0.95*k)
                        if W_upper > self._W_min:
                            windows = np.linspace(self._W_min, W_upper, num=5, dtype=int)
                            R_hat_success, best_W = R_hat_convergence_check(
                                variational_param_history, windows)
                            iterate_average = np.mean(variational_param_history[-best_W:], axis=0)
                            if diagnostics:
                                iterate_average_k_history.append(k)
                                iterate_average_history.append(iterate_average)
                            if R_hat_success:
                                k_Rhat = k
                                k_conv = k - best_W
                                W_check = best_W  # immediately check MCSE

                    # Once convergence has been reached compute the MCSE
                    if k_conv is not None and k - k_conv == W_check:
                        W = W_check
                        converged_iterates = np.array(variational_param_history[-W:])
                        iterate_average = np.mean(converged_iterates, axis=0)
                        if diagnostics and k not in iterate_average_k_history:
                            iterate_average_k_history.append(k)
                            iterate_average_history.append(iterate_average)
                        # compute MCSE
                        with Timer() as mcse_timer:
                            if isinstance(objective.approx, MFGaussian):
                                # For MF Gaussian, use MCSE(mu/sigma,log_sigma)
                                iterate_diff = converged_iterates[W-2,:] - converged_iterates[W-1,:]
                                iterate_diff_zero = iterate_diff == 0
                                # ignore constant variational parameters
                                if np.any(iterate_diff_zero):
                                    indices = np.argwhere(iterate_diff_zero)
                                    converged_iterates = np.delete(converged_iterates, indices, 1)
                                converged_log_sdevs = converged_iterates[:,-dim:]
                                mean_log_stdev = np.mean(converged_log_sdevs, axis=0)
                                ess, mcse  = MCSE(converged_iterates)
                                mcse_mean = mcse[:dim]/np.exp(mean_log_stdev)
                                mcse_stdev = mcse[-dim:]
                                mcse = np.concatenate((mcse_mean, mcse_stdev))
                            else:
                                ess, mcse = MCSE(converged_iterates)
                        if diagnostics:
                            ess_and_mcse_k_history.append(k)
                            ess_history.append(ess)
                            mcse_history.append(mcse)
                        if (np.max(mcse) < self._mcse_threshold and
                            np.min(ess)  > self._ESS_min):
                            k_stopped = k
                            break
                        else:
                            relative_mcse_time = mcse_timer.interval / W
                            relative_opt_time = total_opt_time / k
                            relative_time_ratio = relative_opt_time / relative_mcse_time
                            recheck_scale = max(1.05, 1 + 1/np.sqrt(1 + relative_time_ratio))
                            W_check = int(recheck_scale*W_check+1)
                    if k % self._k_check == 0:
                        avg_loss = np.mean(value_history[max(0, k-1000):k+1])
                        R_conv = 'converged' if k_conv is not None else 'not converged'
                        progress.set_description(
                            'average loss = {:,.5g} | R hat {}|'.format(avg_loss, R_conv))
            except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                progress.close()
        if k_stopped is None:
            if k_conv is None:
                print('WARNING: stationarity not reached after maximum number of iterations')
                print('WARNING: try incresing the learning rate or the maximum number of iterations')
            else:
                print('WARNING: stationarity reached but MCSE too large and/or ESS too small')
                print('WARNING: maximum MCSE = {:.3g}'.format(np.max(mcse)))
                print('WARNING: minimum ESS = {:.1f}'.format(np.min(ess)))
                print(ess)
        else:
            print('Convergence reached at iteration', k_stopped)
        return dict(opt_param = iterate_average,
                    k_conv = k_conv,
                    k_Rhat = k_Rhat,
                    k_stopped = k_stopped,
                    variational_param_history = np.array(variational_param_history),
                    value_history = np.array(value_history),
                    iterate_average_k_history = np.array(iterate_average_k_history),
                    iterate_average_history = np.array(iterate_average_history),
                    descent_dir_history = np.array(descent_dir_history),
                    ess_and_mcse_k_history = np.array(ess_and_mcse_k_history),
                    ess_history = np.array(ess_history),
                    mcse_history = np.array(mcse_history)
                    )
