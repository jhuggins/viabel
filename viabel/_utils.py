import time
import numpy as np


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



