import autograd.numpy as np


def vectorize_if_needed(f, a, axis=-1):
    if a.ndim > 1:
        return np.apply_along_axis(f, axis, a)
    else:
        return f(a)


def ensure_2d(a):
    if a.ndim == 0:
        return a
    while a.ndim < 2:
        a = a[:,np.newaxis]
    return a
