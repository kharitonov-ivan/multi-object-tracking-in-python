import time
import logging


def timer(func):
    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        logging.info(f"Runtime of {func} took {(t2 - t1)} seconds")
        return f

    return inner
