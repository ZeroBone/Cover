import os
from fractions import Fraction
from pathlib import Path

import numpy as np
# noinspection PyPackageRequirements
import z3 as z3
import sys
import random

from z3_utils import is_valid


def _resolve_formula_class_dir():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark/data/spaces").resolve()


def _generate_random_fraction() -> Fraction:

    num = random.randrange(1000)

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


def _coeffs_to_z3_expr(v_x, v_y, coeffs: np.ndarray, /):
    return z3.Sum(
        *(
            coeffs[i] * v
            for i, v in enumerate(v_x) if coeffs[i] != 0
        ),
        *(
            coeffs[i + len(v_x)] * v
            for i, v in enumerate(v_y) if coeffs[i + len(v_x)] != 0
        )
    )


def _main():
    np.set_printoptions(formatter={"object": lambda _s: "%9s" % _s})

    dim = int(sys.argv[1])
    space_count = int(sys.argv[2])

    random.seed(0xdeadbeef)

    print("Dimension: %d Space count: %d" % (dim, space_count))

    affine_offset_scale_factor = 10

    assert dim >= 2

    x_fragment = dim // 2
    y_fragment = dim - x_fragment

    v_x = z3.Reals(" ".join("x_%d" % (i + 1) for i in range(x_fragment)))
    v_y = z3.Reals(" ".join("y_%d" % (i + 1) for i in range(y_fragment)))

    disjuncts = []

    for _ in range(space_count):
        coeff_x = _generate_ker_mat(x_fragment)
        coeff_y = _generate_ker_mat(y_fragment)

        x_pred_coeffs = np.concatenate([
            coeff_x,
            coeff_y
        ])

        y_pred_coeffs = np.concatenate([
            coeff_x,
            2 * coeff_y
        ])

        # print(x_pred_coeffs, y_pred_coeffs)

        coeff_x = np.concatenate([coeff_x, np.zeros(y_fragment, dtype=Fraction)])
        coeff_y = np.concatenate([np.zeros(x_fragment, dtype=Fraction), coeff_y])

        # print(coeff_x, coeff_y)

        affine_offset = np.array([
            affine_offset_scale_factor * _generate_random_fraction() for _ in range(dim)
        ], dtype=Fraction)

        # print("Affine offset: %s" % affine_offset)

        disjunct_pi_respecting = z3.And(
            # x
            _coeffs_to_z3_expr(v_x, v_y, coeff_x) == np.dot(affine_offset, coeff_x),
            # y
            _coeffs_to_z3_expr(v_x, v_y, coeff_y) == np.dot(affine_offset, coeff_y),
        )

        disjunct = z3.And(
            # x
            _coeffs_to_z3_expr(v_x, v_y, x_pred_coeffs) == np.dot(affine_offset, x_pred_coeffs),
            # y
            _coeffs_to_z3_expr(v_x, v_y, y_pred_coeffs) == np.dot(affine_offset, y_pred_coeffs)
        )

        assert is_valid(disjunct == disjunct_pi_respecting)

        disjuncts.append(disjunct)

    # generate formula

    solver = z3.Solver()
    solver.add(z3.Or(*disjuncts))

    os.makedirs(_resolve_formula_class_dir(), exist_ok=True)
    output_file_name = os.path.join(
        _resolve_formula_class_dir(),
        "spaces_dim%03d_spaces%03d.smt2" % (dim, space_count)
    )

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)


if __name__ == "__main__":
    _main()
