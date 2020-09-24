import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.t as t_dist
from autograd.scipy.linalg import sqrtm

from functools import partial
import matplotlib.pyplot as plt

import tqdm
import scipy.stats as stats
from  .optimization_diagnostics import autocorrelation, monte_carlo_se_moving, compute_khat_iterates, \
    gpdfit, monte_carlo_se, compute_R_hat, compute_R_hat_window


__all__ = [
    'adagrad_optimize',
    'adagrad_workflow_optimize',
    'rmsprop_workflow_optimize',
    'adam_workflow_optimize'
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

def adagrad_optimize(n_iters, objective_and_grad, init_param,
                     has_log_norm=False, window=10,learning_rate=.01,
                     epsilon=.1, learning_rate_end=None):
    """
    This is adagrad optimizer without convergence diagnostics, we keep it as a baseline
    , and for cases where the dimensionality is so high that
    the optimizers may become too slow ....

    :param n_iters:
    :param objective_and_grad:
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
                    obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                else:
                    obj_val, obj_grad = objective_and_grad(variational_param)
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
                if i >= 3*n_iters // 4:
                    variational_param_history.append(variational_param.copy())
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


def adagrad_workflow_optimize(n_iters, objective_and_grad, init_param,K,
                     has_log_norm=False, window=10,learning_rate=.01, learning_rate_end=None,
                     epsilon=.1, tolerance=0.05, eval_elbo=100,  stopping_rule=1, n_optimizers=1,
                     r_mean_threshold=1.20, r_sigma_threshold=1.20,
                     tail_avg_iters=200, plotting= False, model_name=None):
    """
    stopping rule 1 means traditional ELBO stopping rule, while
    stopping rule 2 means MCSE stopping rule.

    The windowed Adagrad optimizer with convergence diagnostics and iterate averaging ...

    :param n_iters:
    :param objective_and_grad:
    :param init_param: initial params
    :param K:
    :param has_log_norm:
    :param window:
    :param learning_rate:
    :param epsilon:
    :param rhat_window:
    :param averaging:
    :param n_optimisers:
    :param r_mean_threshold:
    :param r_sigma_threshold:
    :param tail_avg_iters:
    :param eval_elbo:
    :param tolerance:
    :param stopping_rule:
    :param avg_grad_norm:
    :param learning_rate_end:
    :param plotting:
    :param model_name:
    :return:
    """

    log_norm_history = []
    variational_param = init_param.copy()
    prev_elbo = 0.
    pmz_size = init_param.size
    optimisation_log = {}
    variational_param_history_list = []
    variational_param_post_conv_history_list=[]
    # index for iters
    t=0
    # index for iters after convergence ..
    j=0
    N_overall= 50000
    sto_process_convergence = False
    sto_process_sigma_conv = False
    sto_process_mean_conv= False
    for o in range(n_optimizers):
        local_log_norm_history = []
        local_grad_history = []
        log_norm_history = []
        value_history = []
        elbo_diff_rel_med = 10.
        elbo_diff_rel_avg = 10.
        elbo_diff_rel_list = []
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.1
        schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
        t=0
        variational_param_history = []
        variational_param_post_conv_history = []
        mcse_all = np.zeros((pmz_size, 1))
        stop=False
        for curr_learning_rate in schedule:
            if t == N_overall:
                break

            if sto_process_convergence:
                j=j+1

            if has_log_norm == 1:
                obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
            else:
                obj_val, obj_grad = objective_and_grad(variational_param)
                log_norm= 0.

            if stopping_rule==1 and t > 1000 and t % eval_elbo == 0:
                elbo_diff_rel = np.abs(obj_val - prev_elbo) / (prev_elbo+1e-8)
                elbo_diff_rel_list.append(elbo_diff_rel)
                elbo_diff_rel_med = np.nanmedian(elbo_diff_rel_list)
                elbo_diff_rel_avg = np.nanmean(elbo_diff_rel_list)

            prev_elbo = obj_val
            start_stats = 1500
            if stopping_rule == 2 and t > 1500 and t % eval_elbo == 0:
                #print(np.nanmedian(mcse_all[:, -1]))
                mcse_se_combined_list = monte_carlo_se(np.array(variational_param_history)[None,:], 0)
                mcse_all = np.hstack((mcse_all, mcse_se_combined_list[:,None]))

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
            #if i >= 0:
            variational_param_history.append(variational_param.copy())

            if t % 10 == 0:
                avg_loss = np.mean(value_history[max(0, t - 1000):t + 1])

            t=t+1
            if stopping_rule==1 and stop== False and elbo_diff_rel_med  <= tolerance:
                N_overall = t + 100
                stop=True
            if stopping_rule == 1 and stop== False and elbo_diff_rel_avg <= tolerance:
                N_overall = t + 100
                stop=True

            if stopping_rule ==2 and stop == False and sto_process_convergence == True and t > 1500 and \
                    t % eval_elbo == 0 and (np.nanmedian(mcse_all[:,-1]) <= epsilon) and j > 300:
                print('Optimization stopping reliably!')
                stop=True
                break


            variational_param_history_array = np.array(variational_param_history)
            if stopping_rule ==2 and t % eval_elbo ==0 and t  > 800 and sto_process_convergence==False:
                variational_param_history_list.append(variational_param_history_array)
                variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
                variational_param_history_list.pop(0)
                rhats_halfway_last = compute_R_hat(variational_param_history_chains, warmup=0.5)[1]
                rhat_mean_halfway, rhat_sigma_halfway = rhats_halfway_last[:K], rhats_halfway_last[K:]

                if (rhat_mean_halfway < r_mean_threshold ).all() and sto_process_mean_conv == False:
                    start_swa_m_iters = t
                    print('Rhat- All means converged ...')
                    sto_process_mean_conv = True
                    start_stats = start_swa_m_iters

                if (rhat_sigma_halfway < r_sigma_threshold).all() and sto_process_sigma_conv == False:
                    start_swa_s_iters = t
                    print('Rhat- All sigmas converged ...')
                    sto_process_sigma_conv = True
                    start_stats = start_swa_s_iters


            if sto_process_mean_conv == True and sto_process_sigma_conv== True:
                sto_process_convergence= True
                start_stats = np.maximum(start_swa_m_iters, start_swa_s_iters)

            if sto_process_convergence:
                variational_param_post_conv_history.append(variational_param)

            if sto_process_convergence and j > 100 and t%(eval_elbo) ==0:
                variational_param_post_conv_history_array= np.array(variational_param_post_conv_history)
                variational_param_post_conv_history_list.append(variational_param_post_conv_history_array)
                variational_param_post_conv_history_chains = np.stack(variational_param_post_conv_history_list, axis=0)
                variational_param_post_conv_history_list.pop(0)
                pmz_size = variational_param_post_conv_history_chains.shape[2]
                Neff = np.zeros(pmz_size)
                Rhot = []
                khat_iterates = []
                khat_iterates2 = []
                # compute khat for iterates
                for k in range(pmz_size):
                    neff, rho_t_sum, autocov, rho_t = autocorrelation(variational_param_post_conv_history_chains, 0, k)
                    #mcse_se_combined = monte_carlo_se2(variational_param_history_chains, start_stats,i)
                    Neff[k] = neff
                    #mcmc_se2.append(mcse_se_combined)
                    Rhot.append(rho_t)
                    khat_i = compute_khat_iterates(variational_param_post_conv_history_chains, 0, k, increasing=True)
                    khat_iterates.append(khat_i)
                    khat_i2 = compute_khat_iterates(variational_param_post_conv_history_chains, 0, k, increasing=False)
                    khat_iterates2.append(khat_i2)

                rhot_array = np.stack(Rhot, axis=0)
                khat_combined = np.maximum(khat_iterates, khat_iterates2)


    if sto_process_convergence:

        optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
        optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters
        optimisation_log['r_hat_mean_halfway'] = rhat_mean_halfway
        optimisation_log['r_hat_sigma_halfway'] = rhat_sigma_halfway

        try:
            Neff
        except NameError:
            pass
        else:
            optimisation_log['neff'] = Neff
            optimisation_log['autocov'] = autocov
            optimisation_log['rhot'] = rhot_array
            optimisation_log['start_stats'] = start_stats
            # optimisation_log['mcmc_se2'] = mcmc_se2_array
            optimisation_log['khat_iterates_comb'] = khat_combined

    if stopping_rule == 1:
        start_stats = t - tail_avg_iters

    if stopping_rule ==1:
        smoothed_opt_param = np.mean(variational_param_history_array[start_stats:,:], axis=0)
    elif stopping_rule ==2 and sto_process_convergence== True:
        smoothed_opt_param = np.mean(variational_param_post_conv_history_chains[0, :], axis=0)

    if stopping_rule == 2 and sto_process_convergence == False:
        smoothed_opt_param = np.mean(variational_param_history_array[start_stats:,:], axis=0)

    if plotting:
        fig = plt.figure(figsize=(4.2, 2.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rhot_array[0, :100], label='loc-1')
        ax.plot(rhot_array[1, :100], label='loc-2')
        #ax.plot(rhot_array[2, :100], label='loc-3')
        plt.xlabel('Lags')
        plt.ylabel('autocorrelation')
        plt.legend()
        plt.savefig('autocor_model_adagrad_mean_mf.pdf')

        fig = plt.figure(figsize=(4.2, 2.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rhot_array[K, :100], label='sigma-1')
        ax.plot(rhot_array[K + 1, :100], label='sigma-2')
        #ax.plot(rhot_array[K + 2, :100], label='sigma-3')
        plt.xlabel('Lags')
        plt.ylabel('autocorrelation')
        plt.legend()
        plt.savefig('autocor_model_adagrad_sigma_mf.pdf')
        khat_array = optimisation_log['khat_iterates_comb']

    return (smoothed_opt_param, variational_param_history,
            np.array(value_history), np.array(log_norm_history), optimisation_log)



def rmsprop_workflow_optimize(n_iters, objective_and_grad, init_param, K,
                        has_log_norm=False, window=500, learning_rate=.01,
                        epsilon=.05, n_optimisers=1, r_mean_threshold=1.20,
                        r_sigma_threshold=1.20, tail_avg_iters=200,
                        eval_elbo=100, tolerance=0.01, stopping_rule=1, avg_grad_norm=False,
                        learning_rate_end=None, plotting=False, model_name=None):
    """
    stopping rule 1 means traditional ELBO stopping rule, while
    stopping rule 2 means MCSE stopping rule.

    The windowed RMSProp optimizer with the convergence diagnostics and iterate averaging ...

    :param n_iters:
    :param objective_and_grad:
    :param init_param: initial params
    :param K:
    :param has_log_norm:
    :param window:
    :param learning_rate:
    :param epsilon:
    :param rhat_window:
    :param averaging:
    :param n_optimisers:
    :param r_mean_threshold:
    :param r_sigma_threshold:
    :param tail_avg_iters:
    :param eval_elbo:
    :param tolerance:
    :param stopping_rule:
    :param avg_grad_norm:
    :param learning_rate_end:
    :param plotting:
    :param model_name:
    :return:
    """
    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    alpha = 0.99
    scaled_sum_grad_norm = 0.
    averaged_variational_mean_list = []
    averaged_variational_sigmas_list = []
    prev_elbo = 0.
    pmz_size = init_param.size
    optimisation_log = {}
    variational_param_history_list = []
    variational_param_post_conv_history_list=[]
    # index for iters
    t=0
    # index for iters after convergence ..
    j=0
    N_overall= 50000
    sto_process_convergence = False
    sto_process_sigma_conv = False
    sto_process_mean_conv= False
    last_i = np.maximum(int(0.1*n_iters/eval_elbo),2)

    for o in range(n_optimisers):
        variational_param_history = []
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.5
        mcse_all = np.zeros((pmz_size, 1))
        elbo_diff_rel_med = 10.
        elbo_diff_rel_avg = 10.

        elbo_diff_rel_avg2= 10.
        elbo_diff_rel_med2 = 10.
        local_grad_history = []
        local_log_norm_history = []
        value_history = []
        log_norm_history = []
        averaged_variational_mean_list = []
        averaged_variational_sigmas_list = []
        elbo_diff_rel_list = []
        variational_param = init_param.copy()
        t=0
        variational_param_history = []
        variational_param_post_conv_history = []
        mcse_all = np.zeros((pmz_size, 1))
        stop=False
        indices_list = []
        u=0
        Neff = np.ones((pmz_size,1))

        with tqdm.trange(n_iters) as progress:
            try:
                schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
                for i, curr_learning_rate in zip(progress, schedule):
                    if i == N_overall:
                        break

                    if sto_process_convergence:
                        j = j + 1
                    if has_log_norm == 1:
                        obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                    else:
                        obj_val, obj_grad = objective_and_grad(variational_param)
                        log_norm = 0

                    if stopping_rule == 1 and i > 1000 and i % eval_elbo == 0:
                        elbo_diff_rel = np.abs(obj_val - prev_elbo) / (prev_elbo + 1e-8)
                        elbo_diff_rel_list.append(elbo_diff_rel)
                        elbo_diff_rel_med = np.nanmedian(elbo_diff_rel_list)
                        elbo_diff_rel_avg = np.nanmean(elbo_diff_rel_list)
                        elbo_diff_rel_med2 = np.nanmedian(elbo_diff_rel_list[-last_i:])
                        elbo_diff_rel_avg2 = np.nanmean(elbo_diff_rel_list[-last_i:])

                    prev_elbo = obj_val
                    start_stats= 500
                    mcse_se_combined_list = np.zeros((pmz_size, 1))
                    if stopping_rule == 2 and i > 1000 and i % eval_elbo == 0:
                        mcse_se_combined_list = monte_carlo_se(np.array(variational_param_history)[None, :], 0)
                        mcse_all = np.hstack((mcse_all, mcse_se_combined_list[:, None]))
                    value_history.append(obj_val)
                    local_grad_history.append(obj_grad)
                    local_log_norm_history.append(log_norm)
                    log_norm_history.append(log_norm)
                    if len(local_grad_history) > window:
                        local_grad_history.pop(0)
                        local_log_norm_history.pop(0)
                    if has_log_norm:
                        grad_norm = np.exp(log_norm)
                    else:
                        grad_norm = np.sum(obj_grad ** 2, axis=0)
                    if i == 0:
                        if avg_grad_norm:
                            sum_grad_squared = grad_norm
                        else:
                            sum_grad_squared=obj_grad**2
                    else:
                        if avg_grad_norm:
                            sum_grad_squared = grad_norm * alpha + (1. - alpha) * grad_norm
                        else:
                            sum_grad_squared = sum_grad_squared*alpha + (1.-alpha)*obj_grad**2
                    grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                    scaled_grads = grad_scale[:, np.newaxis] * np.array(local_grad_history)
                    accum_sum = np.sum(scaled_grads ** 2, axis=0)
                    scaled_sum_grad_norm = scaled_sum_grad_norm * alpha + (1 - alpha) * accum_sum
                    variational_param = variational_param - curr_learning_rate * obj_grad / np.sqrt(
                        epsilon + sum_grad_squared)
                    # variational_param = variational_param - curr_learning_rate * obj_grad / np.sqrt(epsilon + scaled_sum_grad_norm)

                    variational_param_history.append(variational_param)
                    variational_param_history_array = np.array(variational_param_history)
                    if i % 100 == 0:
                        avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                        progress.set_description(
                            'Average Loss = {:,.6g}'.format(avg_loss))

                    t = t + 1
                    if stopping_rule == 1 and stop == False and elbo_diff_rel_med2 <= tolerance:
                        u= u+1
                        print('Convergence due to ELBO median')
                        if u == 1:
                            N_overall = i + 1
                        indices_list.append(i)
                        stop = True
                    if stopping_rule == 1 and stop == False and elbo_diff_rel_avg2 <= tolerance:
                        print('Convergence due to ELBO mean')
                        u=u+1
                        if u ==1:
                            N_overall = i + 1
                        indices_list.append(i)
                        stop = True

                    if stopping_rule == 2  and sto_process_convergence == True and i > 1500 and \
                            t % eval_elbo == 0 and (np.nanmedian(mcse_all[:, -1]) <= epsilon) \
                            and j > 400 and (np.nanquantile(Neff[:K], 0.50) > 20) and (np.nanquantile(Neff[K:], 0.50) > 5 ):
                        print('Optimization Stopping reliably!')
                        break

                    variational_param_history_array = np.array(variational_param_history)
                    if stopping_rule == 2 and t % eval_elbo == 0 and t > 1000 and sto_process_convergence == False:
                        variational_param_history_list.append(variational_param_history_array)
                        variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
                        variational_param_history_list.pop(0)
                        rhats_halfway_last = compute_R_hat(variational_param_history_chains, warmup=0.5)[1]
                        rhat_mean_halfway, rhat_sigma_halfway = rhats_halfway_last[:K], rhats_halfway_last[K:]

                        if (rhat_mean_halfway < r_mean_threshold).all() and sto_process_mean_conv == False:
                            start_swa_m_iters = i
                            print('Rhat- All means converged ...')
                            sto_process_mean_conv = True
                            start_stats = start_swa_m_iters

                        if (rhat_sigma_halfway < r_sigma_threshold).all() and sto_process_sigma_conv == False:
                            start_swa_s_iters = i
                            print('Rhat- All sigmas converged ...')
                            sto_process_sigma_conv = True
                            start_stats = start_swa_s_iters

                    if sto_process_mean_conv == True and sto_process_sigma_conv == True:
                        sto_process_convergence = True
                        start_stats = np.maximum(start_swa_m_iters, start_swa_s_iters)
                        optimisation_log['start_stats'] = start_stats.copy()

                    if sto_process_convergence:
                        variational_param_post_conv_history.append(variational_param)

                    if sto_process_convergence and j > 200 and t % eval_elbo == 0:
                        variational_param_post_conv_history_array = np.array(variational_param_post_conv_history)
                        variational_param_post_conv_history_list.append(variational_param_post_conv_history_array)
                        variational_param_post_conv_history_chains = np.stack(variational_param_post_conv_history_list,
                                                                              axis=0)
                        variational_param_post_conv_history_list.pop(0)
                        pmz_size = variational_param_post_conv_history_chains.shape[2]
                        Neff = np.zeros(pmz_size)
                        Rhot = []
                        khat_iterates = []
                        khat_iterates2 = []
                        # compute khat for iterates
                        for z in range(pmz_size):
                            neff, rho_t_sum, autocov, rho_t = autocorrelation(
                                variational_param_post_conv_history_chains, 0, z)
                            # mcse_se_combined = monte_carlo_se2(variational_param_history_chains, start_stats,i)
                            Neff[z] = neff
                            # mcmc_se2.append(mcse_se_combined)
                            Rhot.append(rho_t)
                            khat_i = compute_khat_iterates(variational_param_post_conv_history_chains, 0, z,
                                                           increasing=True)
                            khat_iterates.append(khat_i)
                            khat_i2 = compute_khat_iterates(variational_param_post_conv_history_chains, 0, z,
                                                            increasing=False)
                            khat_iterates2.append(khat_i2)

                        rhot_array = np.stack(Rhot, axis=0)
                        khat_combined = np.maximum(khat_iterates, khat_iterates2)

            except (KeyboardInterrupt, StopIteration) as e:
                progress.close()
            finally:
                progress.close()

    if sto_process_convergence:
        optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
        optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters
        optimisation_log['r_hat_mean_halfway'] = rhat_mean_halfway
        optimisation_log['r_hat_sigma_halfway'] = rhat_sigma_halfway
        try:
            Neff
        except NameError:
            pass
        else:
            optimisation_log['neff'] = Neff
            optimisation_log['autocov'] = autocov
            optimisation_log['rhot'] = rhot_array
            optimisation_log['start_stats'] = start_stats
            # optimisation_log['mcmc_se2'] = mcmc_se2_array
            optimisation_log['khat_iterates_comb'] = khat_combined


    if stopping_rule ==1:
        variational_param_history_list.append(variational_param_history_array)
        variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
        smoothed_opt_param = np.mean(variational_param_history_array[start_stats:,:], axis=0)
        averaged_variational_mean_list.append(smoothed_opt_param[:K])
        averaged_variational_sigmas_list.append(smoothed_opt_param[K:])
        optimisation_log['convergence_points'] = indices_list

    elif stopping_rule ==2 and sto_process_convergence== True:
        smoothed_opt_param = np.mean(variational_param_post_conv_history_chains[0, :,:], axis=0)
        averaged_variational_mean_list.append(smoothed_opt_param[:K])
        averaged_variational_sigmas_list.append(smoothed_opt_param[K:])

    if stopping_rule ==2 and sto_process_convergence== False:
        start_stats = t - tail_avg_iters
        variational_param_history_list.append(variational_param_history_array)
        variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
        #variational_param_history = np.stack(variational_param_history)
        #variational_param_history_chains = np.vstack(variational_param_history_chains, variational_param_post_conv_history)
        smoothed_opt_param = np.mean(variational_param_history_array[start_stats:,:], axis=0)
        averaged_variational_mean_list.append(smoothed_opt_param[:K])
        averaged_variational_sigmas_list.append(smoothed_opt_param[K:])

    if plotting:
        fig = plt.figure(figsize=(4.2, 2.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rhot_array[0, :100], label='loc-1')
        ax.plot(rhot_array[1, :100], label='loc-2')
        #ax.plot(rhot_array[2, :100], label='loc-3')
        plt.xlabel('Lags')
        plt.ylabel('autocorrelation')
        plt.legend()
        plt.savefig(f'autocor_{model_name}_rmsprop_loc_mf.pdf')

        fig = plt.figure(figsize=(4.2, 2.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rhot_array[K, :100], label='sigma-1')
        ax.plot(rhot_array[K + 1, :100], label='sigma-2')
        #ax.plot(rhot_array[K + 2, :100], label='sigma-3')
        plt.xlabel('Lags')
        plt.ylabel('autocorrelation')
        plt.legend()
        plt.savefig(f'autocor_{model_name}_rmsprop_sigma_mf.pdf')

    return (variational_param, variational_param_history_chains, averaged_variational_mean_list,
            averaged_variational_sigmas_list,
            np.array(value_history), np.array(log_norm_history), optimisation_log)


def adam_workflow_optimize(n_iters, objective_and_grad, init_param, K,
                        has_log_norm=False, window=100,  learning_rate=.01,learning_rate_end=None,
                        epsilon=.02, averaging=True, n_optimisers=1,
                        r_mean_threshold=1.20, r_sigma_threshold=1.20, tail_avg_iters=200,
                        eval_elbo=100, tolerance=0.01, stopping_rule=1, plotting=True, model_name=None):

    """
    stopping rule 1 means traditional ELBO stopping rule, while
    stopping rule 2 means MCSE stopping rule.

    The windowed ADAM optimizer with the convergence diagnostics and iterate averaging ...

    :param n_iters:
    :param objective_and_grad:
    :param init_param: initial params
    :param K:
    :param has_log_norm:
    :param window:
    :param learning_rate:
    :param epsilon:
    :param rhat_window:
    :param averaging:
    :param n_optimisers:
    :param r_mean_threshold:
    :param r_sigma_threshold:
    :param tail_avg_iters:
    :param eval_elbo:
    :param tolerance:
    :param stopping_rule:
    :param avg_grad_norm:
    :param learning_rate_end:
    :param plotting:
    :param model_name:
    :return:
    """

    optimisation_log = {}
    variational_param_post_conv_history_list=[]
    # index for iters
    t=0
    # index for iters after convergence ..
    j=0
    N_overall= 50000
    sto_process_convergence = False
    sto_process_sigma_conv = False
    sto_process_mean_conv= False

    value_history = []
    log_norm_history = []
    variational_param = init_param.copy()
    averaged_variational_param_history = []
    start_avg_iter = n_iters // 1.3
    variational_param_history_list = []
    averaged_variational_mean_list = []
    averaged_variational_sigmas_list = []
    grad_val= 0.
    grad_squared=0
    beta1=0.9
    beta2=0.999
    prev_elbo = 0.
    pmz_size = init_param.size
    mcse_all = np.zeros(pmz_size)

    for o in range(n_optimisers):
        np.random.seed(seed=o)
        if o >= 1:
            variational_param = init_param + stats.norm.rvs(size=len(init_param))*(o+1)*0.5
        elbo_diff_rel_med = 10.
        elbo_diff_rel_avg = 10.
        local_grad_history = []
        local_log_norm_history = []
        value_history = []
        log_norm_history = []
        averaged_variational_mean_list = []
        averaged_variational_sigmas_list = []
        elbo_diff_rel_list = []
        variational_param = init_param.copy()
        t=0
        variational_param_history = []
        variational_param_post_conv_history = []
        mcse_all = np.zeros((pmz_size, 1))
        stop=False

        with tqdm.trange(n_iters) as progress:
            try:
                schedule = learning_rate_schedule(n_iters, learning_rate, learning_rate_end)
                for i, curr_learning_rate in zip(progress, schedule):
                    if i == N_overall:
                        break

                    if sto_process_convergence:
                        j = j + 1
                    if has_log_norm == 1:
                        obj_val, obj_grad, log_norm = objective_and_grad(variational_param)
                    else:
                        obj_val, obj_grad = objective_and_grad(variational_param)
                        log_norm = 0

                    if stopping_rule == 1 and i > 1000 and i % eval_elbo == 0:
                        elbo_diff_rel = np.abs(obj_val - prev_elbo) / (prev_elbo + 1e-8)
                        elbo_diff_rel_list.append(elbo_diff_rel)
                        elbo_diff_rel_med = np.nanmedian(elbo_diff_rel_list)
                        elbo_diff_rel_avg = np.nanmean(elbo_diff_rel_list)

                    prev_elbo = obj_val
                    start_stats= 1000
                    mcse_se_combined_list = np.zeros((pmz_size, 1))
                    if stopping_rule == 2 and i > 1000 and i % eval_elbo == 0:
                        mcse_se_combined_list = monte_carlo_se(np.array(variational_param_history)[None, :], 0)
                        mcse_all = np.hstack((mcse_all, mcse_se_combined_list[:, None]))

                    value_history.append(obj_val)
                    local_grad_history.append(obj_grad)
                    local_log_norm_history.append(log_norm)
                    log_norm_history.append(log_norm)
                    if len(local_grad_history) > window:
                        local_grad_history.pop(0)
                        local_log_norm_history.pop(0)

                    if has_log_norm:
                        grad_norm = np.exp(log_norm)
                    else:
                        grad_norm = np.sum(obj_grad ** 2, axis=0)
                    if i == 0:
                        grad_squared = 0.9 * obj_grad ** 2
                        grad_val = 0.9 * obj_grad
                    else:
                        grad_squared = grad_squared * beta2 + (1. - beta2) * obj_grad ** 2
                        grad_val = grad_val * beta1 + (1. - beta1) * obj_grad
                    grad_scale = np.exp(np.min(local_log_norm_history) - np.array(local_log_norm_history))
                    scaled_grads = grad_scale[:, np.newaxis] * np.array(local_grad_history)
                    accum_sum = np.sum(scaled_grads ** 2, axis=0)
                    old_variational_param = variational_param.copy()
                    m_hat = grad_val / (1 - np.power(beta1, i + 2))
                    v_hat = grad_squared / (1 - np.power(beta2, i + 2))
                    variational_param = variational_param - curr_learning_rate * m_hat / np.sqrt(epsilon + v_hat)
                    if averaging is True and i > start_avg_iter:
                        averaged_variational_param = (variational_param + old_variational_param * (
                                    i - start_avg_iter)) / (i - start_avg_iter + 1)
                        averaged_variational_param_history.append(averaged_variational_param)

                    if i > 100:
                        variational_param_history.append(old_variational_param)

                    if len(variational_param_history) > 100 * window:
                        variational_param_history.pop(0)
                    if i % 100 == 0:
                        avg_loss = np.mean(value_history[max(0, i - 1000):i + 1])
                        #print(avg_loss)
                        progress.set_description(
                            'Average Loss = {:,.6g}'.format(avg_loss))

                    t = t + 1
                    if stopping_rule == 1 and stop == False and elbo_diff_rel_med <= epsilon:
                        print('Convergence achieved due to ELBO median')
                        N_overall = i + 100
                        stop = True
                    if stopping_rule == 1 and stop == False and elbo_diff_rel_avg <= epsilon:
                        print('Convergence achieved due to ELBO mean')
                        N_overall = i + 100
                        stop = True

                    if stopping_rule == 2 and stop == False and sto_process_convergence == True and i > 1500 and \
                            t % eval_elbo == 0 and (np.nanmedian(mcse_all[:, -1]) <= epsilon) and j > 500:
                        print('Optimization stopping reliably!')
                        stop = True
                        break

                    variational_param_history_array = np.array(variational_param_history)
                    if stopping_rule == 2 and t % eval_elbo == 0 and t > 1000 and sto_process_convergence == False:
                        variational_param_history_list.append(variational_param_history_array)
                        variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
                        variational_param_history_list.pop(0)
                        rhats_halfway_last = compute_R_hat(variational_param_history_chains, warmup=0.5)[1]
                        rhat_mean_halfway, rhat_sigma_halfway = rhats_halfway_last[:K], rhats_halfway_last[K:]
                        if (rhat_mean_halfway < r_mean_threshold).all() and sto_process_mean_conv == False:
                            start_swa_m_iters = i
                            print('Rhat- All mean converged ...')
                            sto_process_mean_conv = True
                            start_stats = start_swa_m_iters

                        if (rhat_sigma_halfway < r_sigma_threshold).all() and sto_process_sigma_conv == False:
                            start_swa_s_iters = i
                            print('Rhat- All sigmas converged ...')
                            sto_process_sigma_conv = True
                            start_stats = start_swa_s_iters

                    if sto_process_mean_conv == True and sto_process_sigma_conv == True:
                        sto_process_convergence = True
                        start_stats = np.maximum(start_swa_m_iters, start_swa_s_iters)

                    if sto_process_convergence:
                        variational_param_post_conv_history.append(variational_param)

                    if sto_process_convergence and j > 200 and t % eval_elbo == 0:
                        variational_param_post_conv_history_array = np.array(variational_param_post_conv_history)
                        variational_param_post_conv_history_list.append(variational_param_post_conv_history_array)
                        variational_param_post_conv_history_chains = np.stack(variational_param_post_conv_history_list,
                                                                              axis=0)
                        variational_param_post_conv_history_list.pop(0)
                        pmz_size = variational_param_post_conv_history_chains.shape[2]
                        Neff = np.zeros(pmz_size)
                        Rhot = []
                        khat_iterates = []
                        khat_iterates2 = []
                        # compute khat for iterates
                        for z in range(pmz_size):
                            neff, rho_t_sum, autocov, rho_t = autocorrelation(
                                variational_param_post_conv_history_chains, 0, z)
                            Neff[z] = neff
                            Rhot.append(rho_t)
                            khat_i = compute_khat_iterates(variational_param_post_conv_history_chains, 0, z,
                                                           increasing=True)
                            khat_iterates.append(khat_i)
                            khat_i2 = compute_khat_iterates(variational_param_post_conv_history_chains, 0, z,
                                                            increasing=False)
                            khat_iterates2.append(khat_i2)

                        rhot_array = np.stack(Rhot, axis=0)
                        khat_combined = np.maximum(khat_iterates, khat_iterates2)

            except (KeyboardInterrupt, StopIteration) as e:
                progress.close()
            finally:
                progress.close()

    if sto_process_convergence:
        optimisation_log['start_avg_mean_iters'] = start_swa_m_iters
        optimisation_log['start_avg_sigma_iters'] = start_swa_s_iters
        optimisation_log['r_hat_mean_halfway'] = rhat_mean_halfway
        optimisation_log['r_hat_sigma_halfway'] = rhat_sigma_halfway
        try:
            Neff
        except NameError:
            pass
        else:
            optimisation_log['neff'] = Neff
            optimisation_log['autocov'] = autocov
            optimisation_log['rhot'] = rhot_array
            optimisation_log['start_stats'] = start_stats
            # optimisation_log['mcmc_se2'] = mcmc_se2_array
            optimisation_log['khat_iterates_comb'] = khat_combined

    if stopping_rule == 1:
        start_stats = i - tail_avg_iters

    if stopping_rule == 1:
        variational_param_history_list.append(variational_param_history_array)
        variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
        smoothed_opt_param = np.mean(variational_param_history_array[start_stats:,:], axis=0)
        averaged_variational_mean_list.append(smoothed_opt_param[:K])
        averaged_variational_sigmas_list.append(smoothed_opt_param[K:])

    elif stopping_rule ==2 and sto_process_convergence==True:
        smoothed_opt_param = np.mean(variational_param_post_conv_history_chains[0, :,:], axis=0)
        averaged_variational_mean_list.append(smoothed_opt_param[:K])
        averaged_variational_sigmas_list.append(smoothed_opt_param[K:])


    if stopping_rule ==2 and sto_process_convergence== False:
        start_stats = t - tail_avg_iters
        variational_param_history_list.append(variational_param_history_array)
        variational_param_history_chains = np.stack(variational_param_history_list, axis=0)
        smoothed_opt_param = np.mean(variational_param_history_array[start_stats:,:], axis=0)
        averaged_variational_mean_list.append(smoothed_opt_param[:K])
        averaged_variational_sigmas_list.append(smoothed_opt_param[K:])

    if plotting:
        fig = plt.figure(figsize=(4.2, 2.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rhot_array[0, :100], label='loc-1')
        ax.plot(rhot_array[1, :100], label='loc-2')
        #ax.plot(rhot_array[2, :100], label='loc-3')
        plt.xlabel('Lags')
        plt.ylabel('autocorrelation')
        plt.legend()
        plt.savefig('autocor_model_adam_mean_mf.pdf')

        fig = plt.figure(figsize=(4.2, 2.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rhot_array[K, :100], label='sigma-1')
        ax.plot(rhot_array[K + 1, :100], label='sigma-2')
        #ax.plot(rhot_array[K + 2, :100], label='sigma-3')
        plt.xlabel('Lags')
        plt.ylabel('autocorrelation')
        plt.legend()
        plt.savefig('autocor_model_adam_sigma_mf.pdf')

    return (variational_param, variational_param_history_chains, averaged_variational_mean_list,
            averaged_variational_sigmas_list,
            np.array(value_history), np.array(log_norm_history), optimisation_log)
