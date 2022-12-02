import autograd.numpy as anp
import numpy as np
import scipy.stats
from autograd.scipy.stats import norm

from viabel.approximations import MFGaussian, MFStudentT
from viabel.objectives import AlphaDivergence, DISInclusiveKL, ExclusiveKL
from viabel.optimization import RMSProp


def _test_objective(objective_cls, num_mc_samples, **kwargs):
    np.random.seed(851)
    # mean = np.random.randn(1,dimension)
    # stdev = np.exp(np.random.randn(1,dimension))
    mean = np.array([1., -1.])[np.newaxis, :]
    stdev = np.array([2., 5.])[np.newaxis, :]

    def log_p(x):
        return anp.sum(norm.logpdf(x, loc=mean, scale=stdev), axis=1)
    approx = MFStudentT(2, 100)
    objective = objective_cls(approx, log_p, num_mc_samples, **kwargs)
    # large number of MC samples and smaller epsilon and learning rate to ensure accuracy
    init_param = np.array([0, 0, 1, 1], dtype=np.float32)
    opt = RMSProp(0.1)
    opt_results = opt.optimize(1000, objective, init_param)
    # iterate averaging introduces some bias, so use last iterate
    est_mean, est_cov = approx.mean_and_cov(opt_results['opt_param'])
    est_stdev = np.sqrt(np.diag(est_cov))
    print(est_stdev, stdev)
    np.testing.assert_almost_equal(mean.squeeze(), est_mean, decimal=1)
    np.testing.assert_almost_equal(stdev.squeeze(), est_stdev, decimal=1)


def test_ExclusiveKL():
    _test_objective(ExclusiveKL, 100)


def test_ExclusiveKL_path_deriv():
    _test_objective(ExclusiveKL, 100, use_path_deriv=True)

    # def __init__(self, approx, model, num_mc_samples, ess_target,
    #              temper_model, temper_model_sampler, temper_fn, temper_eps_init,
    #              use_resampling=True, num_resampling_batches=1, w_clip_threshold=10,
    #              pretrain_batch_size=100):

def test_DISInclusiveKL():
    dim = 2

    _test_objective(DISInclusiveKL, 100,
                    temper_model=scipy.stats.multivariate_normal(mean=[0]*dim, cov=np.diag([1]*dim)).logpdf,
                    temper_model_sampler=lambda n: np.random.multivariate_normal([[0] * dim, [1] * dim], size=n),
                    temper_fn=lambda model_logpdf, temper_logpdf, eps: temper_logpdf * eps + model_logpdf * (1 - eps),
                    temper_eps_init=1,
                    ess_target=50)


def test_AlphaDivergence():
    _test_objective(AlphaDivergence, 100, alpha=2)
