import logging as log
from time import sleep


class except_safe:
    def __init__(self, *exceptions, tries=10):
        self.exceptions = exceptions
        self.tries = tries

    def __call__(self, f):
        def safe_f(*args, **kwargs):
            for i in range(1, self.tries+1):
                try:
                    return f(*args, **kwargs)
                except self.exceptions as e:
                    if i == self.tries:
                        raise e
                    log.warning('except_safe decorator: try #{} (next delay {}s): {}: {}'.format(i, i-1, f.__name__, e))
                    sleep(i-1)
        return safe_f


@except_safe(Exception, tries=4)
def test_except_safe(n_exceptions, buf=[]):
    l = len(buf)
    print(test_except_safe.__name__, l)
    if l < n_exceptions:
        buf.append(l)
        raise Exception('dummy exception')


if __name__ == "__main__":
    test_except_safe(3)  # no exception is thrown
    test_except_safe(4)  # exception is thrown on 4th execution

