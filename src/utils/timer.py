import logging
import time

from contextlib import contextmanager
from typing import Optional


@contextmanager
def timer(name: str, logger: Optional[logging.Logger]):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.3f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
