import autograd.numpy as np

import tqdm

__all__ = [
    'adagrad_optimize',
]


def learning_rate_schedule(n_iters, learning_rate, learning_rate_end):
    """
    This function sets the schedule for the learning rate,
    the learning rate is kept constant for first 25% and last 25 % iterations,
    but it is linearly reduced from 25-75% iterations from learning_rate to learning_rate_end
    such that the 25th and 75th quantile are given by learning_rate_end and learning_rate respectively.

    Parameters
    ----------
    n_iters : number of iterations
    learning_rate : starting learning rate
    learning_rate_end : ending learning rate

    Returns
    -------
    learning_rate : generator for learning rate
    """

    if learning_rate < 0:
        raise ValueError('learning rate must be positive')
    if learning_rate_end is not None:
        if learning_rate <= learning_rate_end:
            raise ValueError('initial learning rate must be greater than final learning rate')
        # constant learning rate for first quarter, then decay like a/(b + i)
        # for middle half, then constant for last quarter
        b = n_iters*learning_rate_end/(2*(learning_rate - learning_rate_end))
        a = learning_rate*b
        start_decrease_at = n_iters//4
        end_decrease_at = 3*n_iters//4
    for i in range(n_iters):
        if learning_rate_end is None or i < start_decrease_at:
            yield learning_rate
        elif i < end_decrease_at:
            yield a / (b + i - start_decrease_at + 1)
        else:
            yield learning_rate_end


def adagrad_optimize(n_iters, objective, init_param,
                     has_log_norm=False, window=10, learning_rate=.01,
                     epsilon=.1, learning_rate_end=None):
    """
    This is adagrad optimizer without convergence diagnostics, we keep it as a baseline
    , and for cases where the dimensionality is so high that
    the optimizers may become too slow ....

    :param n_iters:
    :param objective:
    :param init_param:
    :param has_log_norm:
    :param window:
    :param learning_rate:
    :param epsilon:
    :param learning_rate_end:
    :return:
    """
    local_grad_history = []
    local_log_norm_history = []
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    variational_param_history = []
    with tqdm.trange(n_iters) as progress:
        try:
            schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
            for i, curr_learning_rate in zip(progress, schedule):
                prev_variational_param = variational_param
                if has_log_norm:
                    obj_val, obj_grad, log_norm = objective(variational_param)
                else:
                    obj_val, obj_grad = objective(variational_param)
                    log_norm = 0
                value_history.append(obj_val)
                local_grad_history.append(obj_grad)
                local_log_norm_history.append(log_norm)
                log_norm_history.append(log_norm)
                if len(local_grad_history) > window:
                    local_grad_history.pop(0)
                    local_log_norm_history.pop(0)
                grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                scaled_grads = grad_scale[:,np.newaxis]*np.array(local_grad_history)
                accum_sum = np.sum(scaled_grads**2, axis=0)
                variational_param = variational_param - curr_learning_rate*obj_grad/np.sqrt(epsilon + accum_sum)
                variational_param_history.append(variational_param.copy())
                # retain the last 25% of iterates
                if len(variational_param_history) > i // 4 + 1:
                    variational_param_history.pop()
                if i % 10 == 0:
                    avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                    progress.set_description(
                        'Average Loss = {:,.5g}'.format(avg_loss))
        except (KeyboardInterrupt, StopIteration) as e:  # pragma: no cover
            # do not print log on the same line
            progress.close()
        finally:
            progress.close()
    variational_param_history = np.array(variational_param_history)
    smoothed_opt_param = np.mean(variational_param_history, axis=0)
    return (smoothed_opt_param, variational_param_history,
            np.array(value_history), np.array(log_norm_history))
