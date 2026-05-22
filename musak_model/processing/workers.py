import multiprocessing
from multiprocessing.context import BaseContext


def process_pool_context() -> BaseContext:
    available_methods = multiprocessing.get_all_start_methods()
    if "forkserver" in available_methods:
        return multiprocessing.get_context("forkserver")

    if "spawn" in available_methods:
        return multiprocessing.get_context("spawn")

    return multiprocessing.get_context()
