import os
import pickle
import shutil
import time
from hashlib import md5

import autograd.numpy as np
import pystan


def vectorize_if_needed(f, a, axis=-1):
    if a.ndim > 1:
        return np.apply_along_axis(f, axis, a)
    else:
        return f(a)


def ensure_2d(a):
    if a.ndim == 0:
        return a
    while a.ndim < 2:
        a = a[:, np.newaxis]
    return a


class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def _data_file_path(filename):
    """Returns the path to an internal file"""
    return os.path.abspath(os.path.join(__file__, '../stan_models', filename))


def _stan_model_cache_dir():
    return _data_file_path('cached-stan-models')


def clear_stan_model_cache():
    stan_model_dir = _stan_model_cache_dir()
    if os.path.exists(stan_model_dir):
        shutil.rmtree(stan_model_dir)


def StanModel_cache(model_code=None, model_name=None, **kwargs):
    """Use just as you would `StanModel`"""
    if model_code is None:
        if model_name is None:
            raise ValueError('Either model_code or model_name must be provided')
        model_file = _data_file_path(model_name + '.stan')
        if not os.path.isfile(model_file):
            raise ValueError('invalid model "{}"'.format(model_name))
        with open(model_file) as f:
            model_code = f.read()
    stan_model_dir = _stan_model_cache_dir()
    os.makedirs(stan_model_dir, exist_ok=True)
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pck'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pck'.format(model_name, code_hash)
    cache_file = os.path.join(stan_model_dir, cache_fn)
    if os.path.exists(cache_file):
        print('Using cached StanModel{}'.format('' if model_name is None
                                                else ' for ' + model_name))
        with open(cache_file, 'rb') as f:
            sm = pickle.load(f)
    else:
        sm = pystan.StanModel(model_code=model_code, model_name=model_name)
        with open(cache_file, 'wb') as f:
            pickle.dump(sm, f)

    return sm
