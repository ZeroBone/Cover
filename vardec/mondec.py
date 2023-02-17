from typing import Callable

from vardec_binary import vardec_binary
from z3_utils import get_formula_variables


def _split_arr_by_index(arr, index_pred: Callable[[int], bool]) -> tuple:
    first = []
    second = []
    for i, v in enumerate(arr):
        if index_pred(i):
            first.append(v)
        else:
            second.append(v)
    return first, second


def mondec(phi, phi_vars: list = None) -> bool:

    if phi_vars is None:
        phi_vars = [v.unwrap() for v in get_formula_variables(phi)]

    var_count = len(phi_vars)

    if var_count < 2:
        return True

    q = (var_count - 1).bit_length()

    for i in range(q):

        x, y = _split_arr_by_index(phi_vars, lambda j: (j >> i) & 1 == 0)

        print("Decomposing on partition Pi = {%s, %s}" % (x, y))

        result = vardec_binary(phi, x, y)

        if result is None:
            # the formula is not decomposable
            return False

    return True
