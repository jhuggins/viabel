import os
import pickle
import shutil
import time
from hashlib import md5

import jax.numpy as np
import stan
import bridgestan as bs


def vectorize_if_needed(f, a, axis=-1):
    a = np.asarray(a, dtype='float64')
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


def StanModel_cache(model_name=None, data=None):
    """Get or compile a BridgeStan model with caching functionality."""

    if not model_name:
        raise ValueError("Model name must be provided.")

    stan_file = _data_file_path(f"{model_name}.stan")
    with open(stan_file, 'r') as f:
        model_code = f.read()
    code_hash = md5(model_code.encode('ascii')).hexdigest()

    model_lib_path = os.path.join(_stan_model_cache_dir(), f"{model_name}-{code_hash}.so")
    pickle_path = os.path.join(_stan_model_cache_dir(), f"{model_name}-{code_hash}.pck")

    if os.path.exists(pickle_path):
        print('Using cached StanModel{}'.format('' if model_name is None
                                        else ' for ' + model_name))
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:    
        model = bs.StanModel(model_lib=model_lib_path, model_data=data)
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)

    return model
