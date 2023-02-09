import logging
import os
import sys
from pathlib import Path

import z3 as z3

from lindep_graph import build_lindep_graph
from vardec import vardec
from z3_utils import is_valid, is_sat, is_unsat


_logger = logging.getLogger("vardec")


def _resolve_base_path():
    base_path = Path(__file__).parent
    return (base_path / "../").resolve()


def _user_main():
    x_1, x_2, x_3, y_1, y_2, y_3 = z3.Reals("x_1 x_2 x_3 y_1 y_2 y_3")

    phi_1_eq = 12 * x_1 - 2 * x_2 + 16 * x_3 + 28 * y_1 + 35 * y_2 - 7 * y_3 == 0
    phi_1_lt = 12 * x_1 - 2 * x_2 + 16 * x_3 + 28 * y_1 + 35 * y_2 - 7 * y_3 < 0
    phi_1_gt = 12 * x_1 - 2 * x_2 + 16 * x_3 + 28 * y_1 + 35 * y_2 - 7 * y_3 > 0

    phi_2_eq = 6 * x_1 - x_2 + 8 * x_3 + 16 * y_1 + 20 * y_2 - 4 * y_3 == 0
    phi_2_lt = 6 * x_1 - x_2 + 8 * x_3 + 16 * y_1 + 20 * y_2 - 4 * y_3 < 0
    phi_2_gt = 6 * x_1 - x_2 + 8 * x_3 + 16 * y_1 + 20 * y_2 - 4 * y_3 > 0

    a_space = z3.And(phi_1_eq, phi_2_eq)

    a_space_dec = z3.And(
        6 * x_1 - x_2 + 8 * x_3 == 0,
        4 * y_1 + 5 * y_2 - y_3 == 0
    )

    assert is_valid(a_space == a_space_dec)

    b_space = z3.And(
        6 * x_1 - x_2 + 8 * x_3 == 0,
        2 * x_1 - x_2 + 2 * y_1 + 2 * y_2 == 0,
        10 * x_1 - 5 * x_2 + 2 * y_1 + 2 * y_3 == 0
    )

    assert is_valid(z3.Implies(b_space, a_space))

    # corresponds to p
    p_1 = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 == 0

    assert is_valid(b_space == z3.And(a_space, p_1))

    p_2 = 6 * x_1 - 3 * x_2 + 2 * y_1 + y_2 + y_3 == 0

    assert is_valid(b_space == z3.And(a_space, p_2))

    p_1_opposite = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 < 0
    p_2_opposite = 6 * x_1 - 3 * x_2 + 2 * y_1 + y_2 + y_3 >= 0

    assert is_unsat(z3.And(phi_1_eq, phi_2_eq, p_1_opposite, p_2_opposite))
    assert is_valid(z3.And(phi_1_eq, phi_2_eq) == z3.And(phi_1_eq, phi_2_eq, z3.Or(p_1_opposite, p_2_opposite)))
    assert not is_valid(z3.Or(p_1_opposite, p_2_opposite))
    assert not is_valid(z3.And(phi_1_eq) == z3.And(phi_1_eq, z3.Or(p_1_opposite, p_2_opposite)))
    assert not is_valid(z3.And(phi_2_eq) == z3.And(phi_2_eq, z3.Or(p_1_opposite, p_2_opposite)))

    phi_3_lt = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 < 1
    phi_3_gt = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 > 1
    phi_3_eq = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 == 1

    phi_4_gt = 6 * x_1 - 3 * x_2 + 2 * y_1 + y_2 + y_3 > 0
    phi_4_lt = 6 * x_1 - 3 * x_2 + 2 * y_1 + y_2 + y_3 < 0
    phi_4_eq = 6 * x_1 - 3 * x_2 + 2 * y_1 + y_2 + y_3 == 0

    assert is_valid(z3.And(phi_1_eq, phi_2_eq) == z3.And(phi_1_eq, phi_2_eq, z3.Or(phi_3_lt, phi_4_gt)))
    assert not is_valid(z3.And(phi_1_eq) == z3.And(phi_1_eq, z3.Or(phi_3_lt, phi_4_gt)))
    assert not is_valid(z3.And(phi_2_eq) == z3.And(phi_2_eq, z3.Or(phi_3_lt, phi_4_gt)))

    assert is_sat(z3.And(phi_1_eq, phi_2_eq, phi_3_lt, phi_4_gt))
    assert is_sat(z3.And(phi_1_eq, phi_2_eq, phi_3_lt, z3.Not(phi_4_gt)))
    assert is_sat(z3.And(phi_1_eq, phi_2_eq, z3.Not(phi_3_lt), phi_4_gt))

    assert is_sat(z3.And(phi_1_eq, z3.Not(phi_3_lt), z3.Not(phi_4_gt)))
    assert is_sat(z3.And(phi_2_eq, z3.Not(phi_3_lt), z3.Not(phi_4_gt)))
    assert is_unsat(z3.And(phi_1_eq, phi_2_eq, z3.Not(phi_3_lt), z3.Not(phi_4_gt)))

    # c space

    c_space = z3.And(
        2 * x_1 - x_2 == 0,
        x_1 + 2 * x_3 == 0,
        y_1 + y_2 == 0,
        y_1 + y_3 == 0
    )

    p_C = 3 * x_1 - x_2 + 2 * x_3 + 2 * y_1 + y_2 + y_3 == 0

    assert is_valid(c_space == z3.And(b_space, p_C))
    assert is_valid(c_space == z3.And(a_space, p_2, p_C))

    phi_5_neq = 3 * x_1 - x_2 + 2 * x_3 + 2 * y_1 + y_2 + y_3 != 0
    phi_5_lt = 3 * x_1 - x_2 + 2 * x_3 + 2 * y_1 + y_2 + y_3 < 0
    phi_5_gt = 3 * x_1 - x_2 + 2 * x_3 + 2 * y_1 + y_2 + y_3 > 0
    phi_5_eq = 3 * x_1 - x_2 + 2 * x_3 + 2 * y_1 + y_2 + y_3 == 0

    # put everything together

    phi = z3.And(phi_1_eq, phi_2_eq, z3.Or(phi_3_lt, phi_4_gt), z3.Implies(phi_3_eq, phi_5_neq))

    if "--lindep-graph" in sys.argv[1:]:
        build_lindep_graph(phi, [x_1, x_2, x_3], [y_1, y_2, y_3])
    else:
        vardec(phi, [x_1, x_2, x_3], [y_1, y_2, y_3])


def _main():
    # initialize the logger

    verbose_mode = "--verbose" in sys.argv[1:]
    safe_mode = "--unsafe" not in sys.argv[1:]

    print("Verbose mode: %s" % ("enabled" if verbose_mode else "disabled"))
    print("Safe mode: %s" % ("enabled" if safe_mode else "disabled"))

    _logger.setLevel(logging.DEBUG if verbose_mode else logging.INFO)

    file_handler = logging.FileHandler(os.path.join(_resolve_base_path(), "vardec.log"), mode="w")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s %(levelname)7s]: %(message)s")
    file_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)

    _user_main()


if __name__ == '__main__':
    _main()
