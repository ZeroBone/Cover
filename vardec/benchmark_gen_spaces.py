from fractions import Fraction
from pathlib import Path

import numpy as np
# noinspection PyPackageRequirements
import z3 as z3
import sys
import random

import gauss
from z3_utils import is_sat, is_valid


def _resolve_formula_class_dir():
    base_path = Path(__file__).parent
    return (base_path / "data/spaces").resolve()


def _generate_random_fraction() -> Fraction:

    num = random.randrange(1000)

    if random.choice([True, False]):
        num = -num

    return Fraction(num, 10)


def _generate_space_mat(dim: int, /) -> np.ndarray:
    # generate matrix whose image is of (dim - 1) dimension

    ker_mat = np.zeros((dim, 1), dtype=Fraction)

    assert not np.any(ker_mat)

    # fill ker_mat with random elements
    while not np.any(ker_mat):
        for d in range(dim):
            ker_mat[d][0] = _generate_random_fraction()

    space_mat = gauss.compute_kernel(np.transpose(ker_mat.copy()))

    return space_mat, ker_mat


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
    # random.seed(0xdeadbeef)
    np.set_printoptions(formatter={"object": lambda _s: "%9s" % _s})

    dim = int(sys.argv[1])

    polyhedron_width = 1
    excluded_disjuncts_per_polyhedron = 10

    assert dim >= 2

    x_fragment = dim // 2
    y_fragment = dim - x_fragment

    v_x = z3.Reals(" ".join("x_%d" % (i + 1) for i in range(x_fragment)))
    v_y = z3.Reals(" ".join("y_%d" % (i + 1) for i in range(y_fragment)))

    space_mat_x, ker_mat_x = _generate_space_mat(x_fragment)
    space_mat_y, ker_mat_y = _generate_space_mat(y_fragment)

    coeff_x = np.transpose(ker_mat_x)[0]
    coeff_y = np.transpose(ker_mat_y)[0]

    x_pred_coeffs = np.concatenate([
        coeff_x,
        coeff_y
    ])

    y_pred_coeffs = np.concatenate([
        coeff_x,
        2 * coeff_y
    ])

    print(x_pred_coeffs, y_pred_coeffs)

    coeff_x = np.concatenate([coeff_x, np.zeros(y_fragment, dtype=Fraction)])
    coeff_y = np.concatenate([np.zeros(x_fragment, dtype=Fraction), coeff_y])

    print(coeff_x, coeff_y)

    affine_offset = np.array([
        64 * _generate_random_fraction() for _ in range(dim)
    ], dtype=Fraction)

    print("Affine offset: %s" % affine_offset)

    # define the polyhedron using Pi-respecting predicates
    polyhedron_def = z3.And(
        # x
        _coeffs_to_z3_expr(v_x, v_y, coeff_x) < np.dot(affine_offset, coeff_x) + polyhedron_width,
        _coeffs_to_z3_expr(v_x, v_y, coeff_x) > np.dot(affine_offset, coeff_x) - polyhedron_width,
        # y
        _coeffs_to_z3_expr(v_x, v_y, coeff_y) < np.dot(affine_offset, coeff_y) + polyhedron_width,
        _coeffs_to_z3_expr(v_x, v_y, coeff_y) > np.dot(affine_offset, coeff_y) - polyhedron_width
    )

    print(polyhedron_def)

    assert is_sat(polyhedron_def)

    exclusion_def_pi_respecting = z3.And(
        # x
        _coeffs_to_z3_expr(v_x, v_y, coeff_x) == np.dot(affine_offset, coeff_x),
        # y
        _coeffs_to_z3_expr(v_x, v_y, coeff_y) == np.dot(affine_offset, coeff_y),
    )

    assert is_valid(z3.Implies(exclusion_def_pi_respecting, polyhedron_def))

    first_exclusion = z3.And(
        # x
        _coeffs_to_z3_expr(v_x, v_y, x_pred_coeffs) == np.dot(affine_offset, x_pred_coeffs),
        # y
        _coeffs_to_z3_expr(v_x, v_y, y_pred_coeffs) == np.dot(affine_offset, y_pred_coeffs)
    )

    assert is_valid(first_exclusion == exclusion_def_pi_respecting)

    exclusions = [first_exclusion]

    # sanity check
    x_upper_bound = np.dot(affine_offset, coeff_x + coeff_y) + 2 * polyhedron_width
    x_lower_bound = np.dot(affine_offset, coeff_x + coeff_y) - 2 * polyhedron_width
    y_lower_bound = np.dot(affine_offset, coeff_x) + 2 * np.dot(affine_offset, coeff_y) - 3 * polyhedron_width
    y_upper_bound = np.dot(affine_offset, coeff_x) + 2 * np.dot(affine_offset, coeff_y) + 3 * polyhedron_width

    assert is_valid(z3.Implies(
        polyhedron_def,
        z3.And(
            # x
            _coeffs_to_z3_expr(v_x, v_y, x_pred_coeffs) < x_upper_bound,
            _coeffs_to_z3_expr(v_x, v_y, x_pred_coeffs) > x_lower_bound,
            # y
            _coeffs_to_z3_expr(v_x, v_y, y_pred_coeffs) > y_lower_bound,
            _coeffs_to_z3_expr(v_x, v_y, y_pred_coeffs) < y_upper_bound
        )
    ))

    # change basis
    new_basis_x_upper_bound = np.dot(affine_offset, 2 * x_pred_coeffs - y_pred_coeffs) + polyhedron_width
    new_basis_x_lower_bound = np.dot(affine_offset, 2 * x_pred_coeffs - y_pred_coeffs) - polyhedron_width

    new_basis_y_upper_bound = np.dot(affine_offset, y_pred_coeffs - x_pred_coeffs) + polyhedron_width
    new_basis_y_lower_bound = np.dot(affine_offset, y_pred_coeffs - x_pred_coeffs) - polyhedron_width

    new_basis_x_pred_coeffs = 2 * x_pred_coeffs - y_pred_coeffs
    new_basis_y_pred_coeffs = y_pred_coeffs - x_pred_coeffs

    assert is_valid(polyhedron_def == z3.And(
        # x
        _coeffs_to_z3_expr(v_x, v_y, new_basis_x_pred_coeffs) < new_basis_x_upper_bound,
        _coeffs_to_z3_expr(v_x, v_y, new_basis_x_pred_coeffs) > new_basis_x_lower_bound,
        # y
        _coeffs_to_z3_expr(v_x, v_y, new_basis_y_pred_coeffs) < new_basis_y_upper_bound,
        _coeffs_to_z3_expr(v_x, v_y, new_basis_y_pred_coeffs) > new_basis_y_lower_bound
    ))

    for _ in range(excluded_disjuncts_per_polyhedron - 1):
        pass

    # generate formula

    """
    solver = z3.Solver()
    solver.add(z3.And(
        z3.Or(*grid_predicates),
        *nondec_plane_predicates
    ))

    os.makedirs(_resolve_formula_class_dir(), exist_ok=True)
    output_file_name = os.path.join(
        _resolve_formula_class_dir(),
        "spaces_dim%03d_apc%03d_napc%03d.smt2" % (dim, aligned_plane_count, nonaligned_plane_count)
    )

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)
    """


if __name__ == "__main__":
    _main()
