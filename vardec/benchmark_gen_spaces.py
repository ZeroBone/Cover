import os
from fractions import Fraction
from pathlib import Path

import numpy as np
# noinspection PyPackageRequirements
import z3 as z3
import sys
import random

from gauss import compute_kernel


def _resolve_formula_class_dir():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark/data/spaces").resolve()


def _generate_random_fraction() -> Fraction:

    num = random.randrange(100)

    if random.choice([True, False]):
        num = -num

    return Fraction(num)


def _generate_ker_mat(dim: int, /) -> np.ndarray:

    ker_mat = np.zeros(dim, dtype=Fraction)

    assert not np.any(ker_mat)

    # fill ker_mat with random elements
    for d in range(dim):
        ker_mat[d] = _generate_random_fraction()

    return ker_mat


def _coeffs_to_z3_expr(v, coeffs: np.ndarray, /):
    return z3.Sum(
        *(
            coeffs[i] * v
            for i, v in enumerate(v) if coeffs[i] != 0
        )
    )


def _main():
    np.set_printoptions(formatter={"object": lambda _s: "%9s" % _s})

    space_count = int(sys.argv[1])

    random.seed(0xdeadbeef)

    print("Space count: %d" % space_count)

    affine_offset_scale_factor = 10

    v = z3.Reals("x y_1 y_2")

    disjuncts = []

    coeff_x = _generate_ker_mat(3)

    coeff_y = _generate_ker_mat(2)
    coeff_y_full = np.concatenate([np.array([0], dtype=Fraction), coeff_y])

    coeff_y_ker = compute_kernel(np.array([coeff_y]))
    additional_y_coeffs = np.concatenate([np.array([1], dtype=Fraction), coeff_y_ker.T[0]])

    for _ in range(space_count):

        affine_offset = np.array([
            affine_offset_scale_factor * _generate_random_fraction() for _ in range(3)
        ], dtype=Fraction)

        disjunct = z3.And(
            _coeffs_to_z3_expr(v, coeff_y_full) == np.dot(affine_offset, coeff_y_full),
            z3.Not(z3.And(
                _coeffs_to_z3_expr(v, additional_y_coeffs) == np.dot(affine_offset, additional_y_coeffs),
                _coeffs_to_z3_expr(v, coeff_x) == np.dot(affine_offset, coeff_x)
            ))
        )

        disjuncts.append(disjunct)

    # generate formula

    solver = z3.Solver()
    solver.add(z3.Or(*disjuncts))

    os.makedirs(_resolve_formula_class_dir(), exist_ok=True)
    output_file_name = os.path.join(
        _resolve_formula_class_dir(),
        "spaces_spaces%03d.smt2" % space_count
    )

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)


if __name__ == "__main__":
    _main()
