import os
import pickle
import shutil
import time
from hashlib import md5

import jax.numpy as np
import stan


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
    """Get or compile a BridgeStan model."""

    if not model_name:
        raise ValueError("Model name must be provided.")

    stan_file = _data_file_path(f"{model_name}.stan")
    model_lib_path = os.path.join(_stan_model_cache_dir(), f"{model_name}.so")
    if not os.path.exists(model_lib_path):
        compiled_path = bs.compile_model(stan_file)
        os.rename(compiled_path, model_lib_path)
    if data:
        model = bs.StanModel(model_lib=model_lib_path, model_data=data)
    else:
        model = bs.StanModel(model_lib=model_lib_path)
    return model
