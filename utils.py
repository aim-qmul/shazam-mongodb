from functools import reduce
from typing import List, Tuple


def chain_functions(*functions):
    return lambda *initial: reduce(lambda x, f: f(*x), functions, initial)