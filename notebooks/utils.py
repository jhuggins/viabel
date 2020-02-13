from time import perf_counter

__all__ = ['Timer']

def format_seconds(secs):
    if secs < 1e-3:
        t, u = secs * 1e6, 'microsec'
    elif secs < 1e0:
        t, u = secs * 1e3, 'millisec'
    else:
        t, u = secs, 'sec'
    return '{:.03f} {}'.format(t, u)


class Timer:
    def __init__(self, descr=None):
        self.description = descr

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        self.end = perf_counter()
        self.interval = self.end - self.start
        if self.description is not None:
            time_str = format_seconds(self.interval)
            print('{} took {} to run'.format(self.description, time_str))
