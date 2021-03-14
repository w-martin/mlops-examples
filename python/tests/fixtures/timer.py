import logging
import sys
import time
from contextlib import AbstractContextManager


class Timer(AbstractContextManager):

    def __init__(self, name: str):
        super().__init__()
        self.__logger = logging.getLogger('Timer')
        # self.__logger.addHandler(logging.StreamHandler(sys.stdout))
        self.__logger.setLevel(logging.INFO)
        self.__start_time_s: float
        self.__name = name

    def __enter__(self):
        self.__start_time_s = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__logger.info(f"{self.__name} took {time.time() - self.__start_time_s:.5f}s")
