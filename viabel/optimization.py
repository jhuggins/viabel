from abc import ABC, abstractmethod
from collections import defaultdict
import os

import numpy as np
import jax.numpy as jnp
import tqdm
import stan

from viabel._mc_diagnostics import MCSE, R_hat_convergence_check
from viabel._utils import Timer
from viabel.approximations import MFGaussian



__all__ = [
    'Optimizer',
    'StochasticGradientOptimizer',
    'RMSProp',
    'Adam',
    'Adagrad',
    'WindowedAdagrad',
    'AveragedRMSProp',
    'AveragedAdam',
    'FASO',
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
        results : `dict`
            Must contain at least `opt_param`, the estimate for the optimal
            variational parameter.
        """


class StochasticGradientOptimizer(Optimizer):
    """Stochastic gradient descent.
    """

    def __init__(self, learning_rate, *, weight_decay=0, iterate_avg_prop=0.2,
                 diagnostics=False):
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
        if iterate_avg_prop is not None and (iterate_avg_prop > 1.0
                                             or iterate_avg_prop <= 0.0):
            raise ValueError('"iterate_avg_prop" must be None or between 0 and 1')
        self._iterate_avg_prop = iterate_avg_prop
        self._diagnostics = diagnostics
        self.reset_state()

    def reset_state(self):
        """Reset internal state of the optimizer"""
        pass

    def optimize(self, n_iters, objective, init_param, init_hamflow_model_param=None,
                 init_hamflow_rho_param=None):
        variational_param = init_param.copy()
        iap = self._iterate_avg_prop
        results = defaultdict(list)
        # value_history = []
        # variational_param_history = []
        # descent_dir_history = []
        with tqdm.trange(n_iters) as progress:
            try:
                for k in progress:
                    # take step in descent direction
                    object_val, object_grad = objective(variational_param)
                    descent_dir = self.descent_direction(object_grad)
                    variational_param = objective.update(variational_param, 
                                             self._learning_rate * descent_dir)
                    if variational_param.ndim == 2:
                        variational_param *= (1 - self._weight_decay)
                    # record state information
                    results['value_history'].append(object_val)
                    if self._diagnostics or iap is not None:
                        results['variational_param_history'].append(variational_param.copy())
                        if (iap is not None and len(results['variational_param_history']) > iap * k):
                            results['variational_param_history'].pop(0)
                    if self._diagnostics:
                        results['descent_dir_history'].append(descent_dir)
                    if k % 10 == 0:
                        value_history = jnp.array(results['value_history'])
                        avg_loss = jnp.mean(value_history[max(0, k - 1000):k + 1])
                        progress.set_description(
                            'average loss = {:,.5g}'.format(avg_loss))
            except (KeyboardInterrupt, StopIteration):  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                progress.close()
        
        if iap is not None:
            window = max(1, int(k * iap))
            vph = jnp.array(results['variational_param_history'][-window:])
            results['opt_param'] = np.mean(vph, axis=0)
        else:
            results['opt_param'] = variational_param.copy()
        # if descent_dir_history is not None:
        #     results['descent_dir_history'] = descent_dir_history
        results_dict = {d: np.array(h) for d, h in results.items()}
        return results_dict

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
    """RMSProp optimization method (Hinton and Tieleman, 2012)
    
    Tracks the exponential moving average of squared gradient:
    
    .. math::
        \\nu^{(k+1)} = \\beta \\nu^{(k)} + (1-\\beta) \\hat{g}^{(k)} \\cdot  \\hat{g}^{(k)}
    
    and uses :math:`\\nu^{(k)}` to rescale the current stochastic gradient:
        
    .. math::
        \\hat{g}^{(k+1)}/\\sqrt{\\nu^{(k)}}.
    
    Parameters
    -----------
    beta : `float` optional
        Squared gradient moving average hyper parameter. The default is 0.9
    jitter: `float` optional
        Small value used for numerical stability. The default is 1e-8

    Returns
    ----------
    descent_dir : `numpy.ndarray`, shape(var_param_dim,)
        Descent direction of the optimization algorithm
        
    """

    def __init__(self, learning_rate, *, weight_decay=0, iterate_avg_prop=0.2, 
                 beta=0.9, jitter=1e-8, diagnostics=False):
        self._beta = beta
        self._jitter = jitter
        super().__init__(learning_rate, weight_decay=weight_decay,
                         iterate_avg_prop=iterate_avg_prop, 
                         diagnostics=diagnostics)

    def reset_state(self):
        """
        resetting :math:`\\nu`, the exponential moving average of squared gradient
        """
        self._avg_grad_sq = None

    def descent_direction(self, grad):
        if self._avg_grad_sq is None:
            avg_grad_sq = grad**2
        else:
            avg_grad_sq = self._avg_grad_sq
        avg_grad_sq *= self._beta
        avg_grad_sq += (1. - self._beta) * grad**2
        descent_dir = grad / jnp.sqrt(self._jitter + avg_grad_sq)
        self._avg_grad_sq = avg_grad_sq
        return descent_dir


class AveragedRMSProp(StochasticGradientOptimizer):
    """Averaged RMSProp optimization method (Mukkamala and Hein, 2017, §4)
    
    Uses averaged squared gradient by setting :math:`\\beta_k = 1-1/k` such that 
    
    .. math::
        \\nu^{(k+1)} = \\beta_k \\nu^{(k)} + (1-\\beta_k) \\hat{g}^{(k)} \\cdot  \\hat{g}^{(k)}.
    
    Then,
    
    .. math::
        \\nu^{(k+1)} = (k+1)^{-1} \\sum^k_{k^\\prime =0}\\hat{g}^{(k)} \\cdot  \\hat{g}^{(k)},
    
    where :math:`\\nu^{(k)}` converges to a constant almost surely under certain 
    conditions.
    
    Parameters
    -----------
    jitter: `float` optional
        Small value used for numerical stability. The default is 1e-8
    component_wise: `boolean` optional
        Indication of  component wise discent direction computation

    Returns
    ----------
    descent_dir : `numpy.ndarray`, shape(var_param_dim,)
        Descent direction of the optimization algorithm
    """
    def __init__(self, learning_rate, *, jitter=1e-8,
                 diagnostics=False, component_wise=True):
        self._jitter = jitter
        self._component_wise = component_wise
        super().__init__(learning_rate, diagnostics=diagnostics)

    def reset_state(self):
        """
        resetting :math:`\\nu` and k, the exponential moving average of squared 
        gradient and iteration respectively
        """
        self._avg_grad_sq = None
        self._t = None

    def descent_direction(self, grad):
        if self._avg_grad_sq is None:
            avg_grad_sq = grad**2
            t = 1
        else:
            avg_grad_sq = self._avg_grad_sq
            t = self._t + 1
        beta = 1 - 1/t
        avg_grad_sq *= beta
        avg_grad_sq += (1.-beta)*grad**2
        if self._component_wise:
            descent_dir = grad / jnp.sqrt(self._jitter+avg_grad_sq)
        else:
            descent_dir = grad / jnp.sqrt(self._jitter+jnp.sum(avg_grad_sq))
        self._avg_grad_sq = avg_grad_sq
        self._t = t
        return descent_dir
    
class Adam(StochasticGradientOptimizer):
    """Adam optimization method (Kingma and Ba, 2015)
    
    Tracks exponential moving average of the gradient as well as the 
    squared gradient: 
    
    .. math::    
        m^{(k+1)} &= \\beta_1 m^{(k)} + (1-\\beta_1) \\hat{g}^{(k)}\\\\
        \\nu^{(k+1)} &= \\beta_2 \\nu^{(k)} + (1-\\beta_2) \\hat{g}^{(k)} \\cdot \\hat{g}^{(k)}
    
    
    and uses :math:`m^{(k)}` and  :math:`\\nu^{(k)}` to rescale the current stochastic gradient:
    
    .. math::    
        m^{(k)}/\\sqrt{\\nu^{(k)}}.
        
    Parameters
    ----------
    beta1 : `float` optional
        Gradient moving average hyper parameter. The default is 0.9
    beta2 : `float` optional
        Squared gradient moving average hyper parameter. The default is 0.999
    jitter: `float` optional
        Small value used for numerical stability. The default is 1e-8
    component_wise: `boolean` optional
        Indicator for component-wise normalization of discent direction

    Returns
    ----------
    descent_dir : `numpy.ndarray`, shape(var_param_dim,)
        Descent direction of the optimization algorithm
    """
    def __init__(self, learning_rate, *, beta1=0.9, beta2=0.999, jitter=1e-8,
                 iterate_avg_prop=0.2, diagnostics=False):
        self._beta1 = beta1
        self._beta2 = beta2
        self._jitter = jitter
        super().__init__(learning_rate, iterate_avg_prop=iterate_avg_prop,
                         diagnostics=diagnostics)

    def reset_state(self):
        """
        resetting m and  :math:`\\nu`, the exponential moving average of 
        gradient and squared gradient respectively
        """
        self._momentum = None
        self._avg_grad_sq = None

    def descent_direction(self, grad):
        if self._avg_grad_sq is None:
            avg_grad_sq = grad**2
        else:
            avg_grad_sq = self._avg_grad_sq
        
        if self._momentum is None:
            momentum = grad
        else:
            momentum = self._momentum
        
        momentum *= self._beta1
        momentum += (1. - self._beta1) * grad
        avg_grad_sq *= self._beta2
        avg_grad_sq += (1. - self._beta2) * grad**2
        descent_dir = momentum / jnp.sqrt(self._jitter + avg_grad_sq)
        self._momentum = momentum
        self._avg_grad_sq = avg_grad_sq
        return descent_dir
    
class AveragedAdam(StochasticGradientOptimizer):
    """Averaged Adam optimization method (Mukkamala and Hein, 2017, §4)
    
    Uses averaged squared gradient by setting :math:`\\beta_k = 1-1/k` such that
    
    .. math::    
        \\nu^{(k+1)} = \\beta_k \\nu^{(k)} + (1-\\beta_k) \\hat{g}^{(k)} \\cdot \\hat{g}^{(k)}.
    Then,
    
    .. math::   
        \\nu^{(k+1)} = (k+1)^{-1} \\sum^k_{k^\\prime =0}\\hat{g}^{(k)} \\cdot  \\hat{g}^{(k)},
    where :math:`\\nu^{(k)}` converges to a constant almost surely under certain 
    conditions.
    
    Parameters
    ----------
    beta1 : `float` optional
        Gradient moving average hyper parameter. The default is 0.9
    jitter: `float` optional
        Small value used for numerical stability. The default is 1e-8
    component_wise: `boolean` optional
        Indicator for component-wise normalization of discent direction

    Returns
    ----------
    descent_dir : `numpy.ndarray`, shape(var_param_dim,)
        Descent direction of the optimization algorithm
    """
    def __init__(self, learning_rate, *, beta1=0.9, jitter=1e-8,
                 diagnostics=False, component_wise=True):
        self._beta1 = beta1
        self._jitter = jitter
        self._component_wise = component_wise
        super().__init__(learning_rate, diagnostics=diagnostics)

    def reset_state(self):
        """
        resetting m, :math:`\\nu` and, k, the exponential moving average of 
        gradient, squared gradient, and iteration respectively
        """
        self._momentum = None
        self._avg_grad_sq = None
        self._t = None

    def descent_direction(self, grad):
        if self._avg_grad_sq is None:
            avg_grad_sq = grad**2
            t = 1
        else:
            avg_grad_sq = self._avg_grad_sq
            t = self._t + 1
        if self._momentum is None:
            momentum = grad
        else:
            momentum = self._momentum
        
        momentum *= self._beta1
        momentum += (1. - self._beta1) * grad
        beta2 = 1 - 1/t
        avg_grad_sq *= beta2
        avg_grad_sq += (1. - beta2) * grad**2
        if self._component_wise:
            descent_dir = momentum / jnp.sqrt(self._jitter+avg_grad_sq)
        else:
            descent_dir = momentum / jnp.sqrt(self._jitter+jnp.sum(avg_grad_sq))
        self._momentum = momentum
        self._avg_grad_sq = avg_grad_sq
        self._t = t
        return descent_dir

class Adagrad(StochasticGradientOptimizer):
    """Adagrad optimization method (Duchi et al., 2011)
    
    Uses accumilated squared gradients to rescale the current stochastic
    gradient:   
        
    .. math::   
        \\frac{\\hat{g}^{(k+1)}}{\\sqrt{\\sum^k_{k^\\prime} \\hat{g}^{(k^\\prime)} \\cdot \\hat{g}^{(k^\\prime)}}}
    
    Parameters
    -----------
    jitter: `float` optional
        Small value used for numerical stability. The default is 1e-8

    Returns
    ----------
    descent_dir : `numpy.ndarray`, shape(var_param_dim,)
        Descent direction of the optimization algorithm
    """

    def __init__(self, learning_rate, *, weight_decay=0, jitter=1e-8, 
                 iterate_avg_prop=0.2, diagnostics=False):
        self._jitter = jitter
        super().__init__(learning_rate, weight_decay=weight_decay, 
                         iterate_avg_prop=iterate_avg_prop, diagnostics=diagnostics)

    def reset_state(self):
        """
        restting accumilated squared gradient
        """
        self._sum_grad_sq = 0

    def descent_direction(self, grad):
        self._sum_grad_sq += grad**2
        descent_dir = grad / jnp.sqrt(self._jitter + self._sum_grad_sq)
        return descent_dir

class WindowedAdagrad(StochasticGradientOptimizer):
    """Windowed Adagrad optimization method (Default optimizer in Pymc3)
    
    Uses a running window (w) to get the mean squared gradient to rescale
    the current stochastic gradient:
        
    .. math::
        \\frac{\\hat{g}^{(k+1)}}{\\sqrt{\\sum^k_{k^\\prime = k-w} \\hat{g}^{(k^\\prime)} \\cdot \\hat{g}^{(k^\\prime)}}}
    
    Parameters
    -----------
    window size : `int` optional
        Window size used to store the square of the gradients. The default is 10
    jitter: `float` optional
        Small value used for numerical stability. The default is 1e-8

    Returns
    ----------
    descent_dir : `numpy.ndarray`, shape(var_param_dim,)
        Descent direction of the optimization algorithm
    """

    def __init__(self, learning_rate, *, weight_decay=0, window_size=10, 
                 jitter=1e-8, diagnostics=False):
        self._window_size = window_size
        self._jitter = jitter
        super().__init__(learning_rate, weight_decay=weight_decay, 
                         diagnostics=diagnostics)

    def reset_state(self):
        """
        resetting the running squared gradients
        """
        self._history = []

    def descent_direction(self, grad):
        self._history.append(grad**2)
        if len(self._history) > self._window_size:
            self._history.pop(0)
        mean_grad_squared = jnp.mean(jnp.array(self._history), axis=0)
        descent_dir = grad / jnp.sqrt(self._jitter + mean_grad_squared)
        return descent_dir

        
class FASO(Optimizer):
    """Fixed-learning rate stochastic optimization meta-algorithm (FASO)
    
    This algorithm runs stochastic optimization with a fixed-learning rate using 
    a user specified optimization method. It determines the convergence at the 
    fixed-learning rate using the potential scale reduction factor \(\hat{R}\) and 
    combined with a Monte Carlo standard error cutoff.
    
    See more details: https://arxiv.org/pdf/2203.15945.pdf

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
        diagnostics = self._sgo._diagnostics
        k_conv = None  # Iteration number when reached convergence
        k_stopped = None  # Iteration number when MCSE/ESS conditions met
        k_Rhat = None  # Iteration number when R hat convergence criterion met
        learning_rate = self._sgo._learning_rate
        variational_param = init_param.copy()
        history = defaultdict(list)
        iterate_average = variational_param.copy()
        if diagnostics:
            history['iterate_average_k_history'].append(0)
            history['iterate_average_history'].append(iterate_average)
        total_opt_time = 0  # total time spent on optimization
        with tqdm.trange(n_iters) as progress:
            try:
                for k in progress:
                    # take step in descent direction
                    with Timer() as opt_timer:
                        object_val, object_grad = objective(variational_param)
                        history['value_history'].append(object_val)
                        history['grad_history'].append(object_grad)
                        descent_dir = self._sgo.descent_direction(object_grad)
                        variational_param = objective.update(variational_param, learning_rate * descent_dir)
                        history['variational_param_history'].append(variational_param.copy())
                        if diagnostics:
                            history['descent_dir_history'].append(descent_dir)
                    total_opt_time += opt_timer.interval
                    # If convergence has not been reached then check for
                    # convergence using R hat
                    if k_conv is None and k % self._k_check == 0:
                        W_upper = int(0.95 * k)
                        if W_upper > self._W_min:
                            windows = jnp.linspace(self._W_min, W_upper, num=5, dtype=int)
                            vph = jnp.array(history['variational_param_history'])
                            R_hat_success, best_W = R_hat_convergence_check(vph, windows)
                            iterate_average = jnp.mean(vph[-best_W:], axis=0)
                            if diagnostics:
                                history['iterate_average_k_history'].append(k)
                                history['iterate_average_history'].append(iterate_average)
                            if R_hat_success:
                                k_Rhat = k
                                k_conv = k - best_W
                                W_check = best_W  # immediately check MCSE

                    # Once convergence has been reached compute the MCSE
                    if k_conv is not None and k - k_conv == W_check:
                        W = W_check
                        converged_iterates = jnp.array(history['variational_param_history'][-W:])
                        iterate_average = jnp.mean(converged_iterates, axis=0)
                        if diagnostics and k not in history['iterate_average_k_history']:
                            history['iterate_average_k_history'].append(k)
                            history['iterate_average_history'].append(iterate_average)
                        # compute MCSE
                        with Timer() as mcse_timer:
                            if isinstance(objective.approx, MFGaussian):
                                dim = int(init_param.size/2)
                                # For MF Gaussian, use MCSE(mu/sigma,log_sigma)
                                iterate_diff = (converged_iterates[W - 2, :]
                                                - converged_iterates[W - 1, :])
                                iterate_diff_zero = iterate_diff == 0
                                # ignore constant variational parameters
                                if jnp.any(iterate_diff_zero):
                                    indices = jnp.argwhere(iterate_diff_zero)
                                    converged_iterates = jnp.delete(converged_iterates, indices, 1)
                                converged_log_sdevs = converged_iterates[:, -dim:]
                                mean_log_stdev = jnp.mean(converged_log_sdevs, axis=0)
                                ess, mcse = MCSE(converged_iterates)
                                mcse_mean = mcse[:dim] / jnp.exp(mean_log_stdev)
                                mcse_stdev = mcse[-dim:]
                                mcse = jnp.concatenate((mcse_mean, mcse_stdev))
                            else:
                                ess, mcse = MCSE(converged_iterates)
                        if diagnostics:
                            history['ess_and_mcse_k_history'].append(k)
                            history['ess_history'].append(ess)
                            history['mcse_history'].append(mcse)
                        if (jnp.max(mcse) < self._mcse_threshold and jnp.min(ess) > self._ESS_min):
                            k_stopped = k
                            break
                        else:
                            relative_mcse_time = mcse_timer.interval / W
                            relative_opt_time = total_opt_time / k
                            relative_time_ratio = relative_opt_time / relative_mcse_time
                            recheck_scale = max(1.05, 1 + 1 / jnp.sqrt(1 + relative_time_ratio))
                            W_check = int(recheck_scale * W_check + 1)
                    if k % self._k_check == 0:
                        value_history = jnp.array(history['value_history'])
                        avg_loss = jnp.mean(value_history[max(0, k - 1000):k + 1])
                        R_conv = 'converged' if k_conv is not None else 'not converged'
                        progress.set_description(
                            'average loss = {:,.5g} | R hat {}|'.format(avg_loss, R_conv))
            except (KeyboardInterrupt, StopIteration):  # pragma: no cover
                # do not print log on the same line
                progress.close()
            finally:
                progress.close()
        if k_stopped is None:
            if k_conv is None:
                print('WARNING: stationarity not reached after maximum number of iterations')
                print('WARNING: try incresing the learning rate or the maximum number of '
                      'iterations')
            else:
                print('WARNING: stationarity reached but MCSE too large and/or ESS too small')
                print('WARNING: maximum MCSE = {:.3g}'.format(jnp.max(mcse)))
                print('WARNING: minimum ESS = {:.1f}'.format(jnp.min(ess)))
                # print(ess)
        else:
            print('Convergence reached at iteration', k_stopped)
        results = {d: jnp.array(h) for d, h in history.items()}
        results['k_conv'] = k_conv
        results['k_Rhat'] = k_Rhat
        results['k_stopped'] = k_stopped
        results['opt_param'] = iterate_average
        return results

class RAABBVI(FASO):
    """A robust, automated, and accurate BBVI optimizer (RAABBVI)
    
    This algorithm combines the FASO algorithm with a termination rule to determine
    the appropriate point where the algorithm could terminate. The termination rule is
    based on the trade-off between improved accuracy of the variational approximation
    if the current learning rate is reduced by an adaptation factor :math:`\\rho \in (0,1)`  and 
    the time required to reach that improved accuracy. If the improved accuracy
    level is large compared to the runtime then this algorithm adaptively decrease
    the learning rate and if not algorithm will be terminated.
    
    See more details: https://arxiv.org/pdf/2203.15945.pdf

    Parameters
    ----------
    sgo : `StochasticGradientOptimizer` object
        Optimizer to use for computing descent direction.
    rho : `float`, optional
        Learning rate reducing factor. The default is 0.5
    iters0 : `int`, optional
        Small iteration number. The default is 1000
        for decreased learning rate.
    accuracy_threshold : `float`, optional
        Absolute SKL accuracy threshold 
    inefficiency_threshold : `float`, optional
        Threshold for the inefficiency index.
    init_rmpsprop : `Boolean`, optional
        Indicate whether to run using RMSProp optimization method for the initial learning rate
    **kwargs:
        Keyword arguments for `FASO`
    """
    def __init__(self, sgo, *, rho=0.5, iters0=1000, accuracy_threshold=0.1, inefficiency_threshold=1.0,
                 init_rmsprop=False, **kwargs):
        super().__init__(sgo, **kwargs)
        self._iters0 = iters0
        self._rho = rho
        self._accuracy_threshold = accuracy_threshold
        self._inefficiency_threshold = inefficiency_threshold
        self._init_rmsprop = init_rmsprop
        if rho < 0 or rho > 1:
            raise ValueError('"rho" must be between zero and one')

    def weighted_linear_regression(self, model_name, y, x, s=9, a=0.25, n_chains=4):
        """
        weighted regression with likelihood term having the weight
        Parameters
        ----------
        model : `pystan model`
            Pystan model to conduct the sampling
        y : `numpy_ndarray`
            Response variable
        x : `numpy_ndarray`
            Predictor variable
        s : `int`, optional
            Observation weight parameter. The default is 9.
        a : `int`, optional
            Power of the weight parameter. The default is 1/4.
        n_chains: `int`, optional
            Number of sampling chains. The default is 4.
            
        Returns
        -------
        kappa : `float`
            Power
        c : `float`
            Constant
        fit : `Pystan object`
            Pystan fit object if results = True
        """
        #defining initialization function
        def initfun(log_c, sigma, kappa=None, chain_id=1):
            if kappa is None:
                return dict(log_c=log_c, sigma=sigma)
            else:
                return dict(kappa=kappa, log_c=log_c, sigma=sigma)
        
        def _data_file_path(filename):
            """Returns the path to an internal file"""
            return os.path.abspath(os.path.join(__file__, '../data', filename))
        model_file = _data_file_path(model_name + '.stan')
        with open(model_file) as f:
            model_code = f.read()
        N = len(y)
        w = np.array(1/(1 + np.arange(N)[::-1]**2/s)**a) #weights
        data = dict(N=np.array(N), y=y, x=x, rho=np.array(self._rho), w=w) #data
        if isinstance(self._sgo, AveragedRMSProp) or isinstance(self._sgo, AveragedAdam):
              init = [initfun(100, 5, chain_id=i) for i in range(n_chains) ] #initial values
        else:
            init = [initfun(100, 5, 0.8, chain_id=i) for i in range(n_chains) ] #initial values
        model = stan.build(program_code=model_code, data=data)
        samples = model.sample(num_chains=n_chains, num_samples=1000,init = init)   # sampling from the model
        if isinstance(self._sgo, AveragedRMSProp) or isinstance(self._sgo, AveragedAdam):
            kappa = 1
        else:
            kappa = jnp.mean(samples['kappa'])
        log_c = jnp.mean(samples['log_c'])
        c = jnp.exp(log_c)
        return samples, kappa, c
        
    
    def wls(self, x, y, s=9, a=0.25):
        """
        weighted least squares

        Parameters
        ----------
        x : `numpy_ndarray`
            Predictor variable
        y : `numpy_ndarray`
            Response variable
        s : `int`, optional
            Observation weight parameter. The default is 9.
        a : `int`, optional
            Power of the weight parameter. The default is 1/4.

        Returns
        -------
        b0 : `float`
            Intercept
        b1 : `float`
            Slope
        """
        n = y.size
        x = jnp.column_stack((jnp.ones(n),x))
        w = jnp.diag(1/(1 + jnp.arange(n)[::-1]**2/s**2)**a) #weights
        y = jnp.reshape(y,(n,1))
        beta = jnp.linalg.inv(x.T @ w @ x) @ (x.T @ w @ y)
        return beta[0], beta[1]
        
    def convg_iteration_trend_detection(self, slope):
        """
        Detecting the relationship trend between learning
        rate and number of iterations to reach convergence

        Parameters
        ----------
        slope : `float`
            slope of the fitted regression model

        Returns
        -------
        bool
            Indicating having negative relationship or not

        """
        if slope < 0:
            return True
        else:
            return False
        

    def optimize(self, K_max, objective, init_param):
        """
        Parameters
        ----------
        K_max : `int`
            Number of iterations of the optimization
        objective: `function`
            Function for constructing the objective and gradient function
        init_param : `numpy.ndarray`, shape(var_param_dim,)
            Initial values of the variational parameters

        """
        if not objective.approx.supports_kl:
            print('WARNING: approximation family does not support KL. Using FASO.',
                  flush=True)
            return super().optimize(K_max, objective, init_param)
        k_new = -1 #Number of iterations at the given learning rate
        k = 0 #Number of times learning rate decreases
        k_total = 0 #Total number of iterations
        k_add = 0 #Iteration number when convergence reached for a fixed step size
        k_stopped_final = None #Iteration number when stopping rule criteria met
        sgo = self._sgo
        diagnostics = self._sgo._diagnostics
        if isinstance(self._sgo, AveragedRMSProp) or isinstance(self._sgo, AveragedAdam):
            reg_model = 'weighted_lin_regression_sgd'
        else:
            reg_model = 'weighted_lin_regression'
        iterate_average_curr = init_param.copy()
        history = defaultdict(list)
        history['iterate_average_curr_hist'].append(iterate_average_curr)
        history['k_mcse'].append(0)
        stopped = False
        try:
            while not stopped:
                K_max -= (k_new + 1)
                iterate_average_prev = iterate_average_curr
                if k == 0 and self._init_rmsprop: #Using RMSProp optimization initially if specified
                    rmsprop = RMSProp(learning_rate=sgo._learning_rate, diagnostics=diagnostics)
                    faso = FASO(sgo = rmsprop)
                    opt = faso.optimize(K_max, objective, iterate_average_curr)
                else:
                    opt = super().optimize(K_max, objective, iterate_average_curr)
                if opt['k_stopped'] is not None and k != 0: #removing the number of convergence iterations of initial learning rate
                    convg_iters = opt['k_stopped']
                    history['conv_iters_hist'].append(convg_iters)
                    # CI.append(convg_iters)
                iterate_average_curr = opt['opt_param']
                history['iterate_average_curr_hist'].append(iterate_average_curr)
                k_new = opt['k_stopped']
                
                # checking whether R hat convergence criteria reached and the FASO stopping criteria reached to add new number of iterations
                history['k_Rhat'].append(opt['k_Rhat'] + k_add if opt['k_Rhat'] is not None and k_new is not None else opt['k_Rhat']) 
                #checking whether R hat convergence criteria and the FASO stopping criteria reached to add new number of iterations
                history['k_conv'].append(opt['k_conv'] + k_add if opt['k_conv'] is not None and k_new is not None else opt['k_conv'])
                history['k_mcse'].append(k_new + k_add if k_new is not None else k_new)   
                history['variational_param_history'].extend(opt['variational_param_history'])
                history['value_history'].extend(opt['value_history'])
                history['grad_history'].extend(opt['grad_history'])
                
                #if user require diagnostics
                if diagnostics: 
                    history['descent_dir_history'].extend(opt['descent_dir_history'])
                    #checking convergence detection to store computed MCSE and ESS
                    if opt['k_conv'] is not None: 
                        history['ess_history'].extend(opt['ess_history'])
                        history['mcse_history'].extend(opt['mcse_history'])
                        if len(history['mcse_history']) > 0:
                            history['final_mcse_history'].append(history['mcse_history'][-1])
                        else:
                             history['final_mcse_history'].append(history['mcse_history'])
                    #saving the iteration histories at the initial learning rate
                    if k == 0: 
                        history['iterate_average_k_history'].extend(opt['iterate_average_k_history'])
                        history['iterate_average_history'].extend(opt['iterate_average_history'])
                    else: #adding previous number of iterations to the current iterations 
                        history['iterate_average_k_history'].extend(opt['iterate_average_k_history'][1:] + k_add)
                        history['iterate_average_history'].extend(opt['iterate_average_history'][1:,:])
               
                k_add = history['iterate_average_k_history'][-1]
                    
                if k_new is None:  # maximum number of iterations reached
                    break
                else: #compute the stopping criteria
                    k_total += k_new
                    sgo._learning_rate *= self._rho
                    self._mcse_threshold *= self._rho
                    if isinstance(self._sgo, AveragedRMSProp) or isinstance(self._sgo, AveragedAdam):
                        self._sgo.reset_state()
                    if len(history['learning_rate_hist']) > 0:
                        SKL = (objective.approx.kl(iterate_average_prev, iterate_average_curr) +
                               objective.approx.kl(iterate_average_curr, iterate_average_prev))
                        history['SKL_history'].append(SKL)
                        
                        # Conduct weighted linear regression to estimate parameters
                        # of SKL hat
                        if len(history['SKL_history']) > 0:
                            skl_history = np.array(history['SKL_history'])
                            y_wlr = np.log(skl_history)
                            learning_hist = np.array(history['learning_rate_hist'])
                            x_wlr = np.log(learning_hist)
                            fit, kappa, c = self.weighted_linear_regression(reg_model, y_wlr, x_wlr)
                            if diagnostics:
                                history['c_sample_hist'].append(jnp.exp(fit['log_c']))
                                if isinstance(self._sgo, AveragedRMSProp) or \
                                    isinstance(self._sgo, AveragedAdam):
                                    history['kappa_sample_hist'] = None
                                else:
                                    history['kappa_sample_hist'].append(fit['kappa'])
                            history['kappa_hist'].append(kappa)
                            history['c_hist'].append(c)
                            #computing the termination rule criteria 
                            if len(history['learning_rate_hist']) > 1:
                                relative_skl = (self._rho)**kappa + \
                                (self._accuracy_threshold/(jnp.sqrt(c) *
                                       history['learning_rate_hist'][-1]**kappa))
                                curr_iters = history['conv_iters_hist'][-1]
                                learning_hist = jnp.array(history['learning_rate_hist'])
                                conv_iter = jnp.array(history['conv_iters_hist'])
                                _, slope = self.wls(jnp.log(learning_hist),
                                                    jnp.log(conv_iter))
                                trend_check = self.convg_iteration_trend_detection(slope)
                                if trend_check: #if negative relationship use all observations
                                    y_wls = jnp.array(history['conv_iters_hist'])
                                    x_wls = jnp.array(history['learning_rate_hist'])
                                else: #remove the initial observation
                                    y_wls = jnp.array(history['conv_iters_hist'][1:])
                                    x_wls = jnp.array(history['learning_rate_hist'][1:])
                                b0, b1 = self.wls(jnp.log(x_wls), jnp.log(y_wls))
                                pred_iters = int(jnp.exp(b0) * \
                                        (self._rho * history['learning_rate_hist'][-1])**b1)
                                history['predicted_iters_hist'].append(pred_iters)
                                relative_iters = pred_iters/(curr_iters + self._iters0)
                                history['stopping_crt'].append(relative_skl * relative_iters)
                                #checking whether termination rule condition satisfied
                                if relative_skl * relative_iters > self._inefficiency_threshold:
                                    stopped = True
                                    k_stopped_final = k_total
                                    history['k_stopped_final_hist'].append(k_total)
                                    break 
                                
                    history['learning_rate_hist'].append(sgo._learning_rate)
                    # LR.append(sgo._learning_rate)
                k += 1
        except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
            pass
        if stopped:
            print('Termination rule reached at iteration', k_total)
            print('Inefficiency Index:', relative_skl * relative_iters)
        else:
            print('WARNING: maximum number of iterations reached before '
                  'stopping rule was triggered')
        results = {d: jnp.array(h) for d, h in history.items() if d!='k_Rhat' and d!='k_mcse' and d!='k_conv' }
        results['opt_param'] = iterate_average_curr
        results['k_stopped_final'] = k_stopped_final
        results['k_Rhat'] = history['k_Rhat']; results['k_mcse'] = history['k_mcse']
        results['k_conv'] = history['k_conv']
        return results
        
