import os
from fractions import Fraction
from pathlib import Path

# noinspection PyPackageRequirements
import z3 as z3
import sys
import random


def _resolve_formula_class_dir(_suffix: str):
    base_path = Path(__file__).parent
    return (base_path / ("../benchmark/data/grid%s" % _suffix)).resolve()


if __name__ == "__main__":

    random.seed(0xdeadbeef)

    dim = int(sys.argv[1])
    aligned_plane_count = int(sys.argv[2])
    nonaligned_plane_count = int(sys.argv[3])

    n_dim_mode = "--ndim" in sys.argv

    assert dim >= 2

    print("Dimension: %d Aligned plane count: %d Non-aligned plane count: %d" %
          (dim, aligned_plane_count, nonaligned_plane_count))

    v = z3.Reals("x " + " ".join("y_%d" % i for i in range(1, dim)))

    # generate grid

    grid_predicates = []

    for grid_offset in range(aligned_plane_count):
        grid_predicates.append(v[0] == (grid_offset // 2 if grid_offset % 2 == 0 else -(grid_offset // 2) - 1))

    # generate non-decomposable hyperplanes intersecting the grid

    nondec_plane_predicates = []

    dim_absent = 1

    for _ in range(nonaligned_plane_count):

        if dim_absent < dim:
            beta_var = dim_absent
            dim_absent += 1
        else:
            beta_var = random.randrange(1, dim)

        alpha = Fraction(
            random.randrange(10000),
            100
        )

        if random.choice([True, False]):
            alpha = -alpha

        beta = Fraction(
            random.randrange(10000),
            100
        )

        if random.choice([True, False]):
            beta = -beta

        predicate = alpha * v[0] + beta * v[beta_var] != 0
        nondec_plane_predicates.append(predicate)

    # put everything together

    solver = z3.Solver()
    solver.add(z3.And(
        z3.Or(*grid_predicates),
        *nondec_plane_predicates
    ))

    _suffix = "ndim" if n_dim_mode else ""

    os.makedirs(_resolve_formula_class_dir(_suffix), exist_ok=True)
    output_file_name = os.path.join(
        _resolve_formula_class_dir(_suffix),
        "grid%s_dim%03d_apc%03d_napc%03d.smt2" % (_suffix, dim, aligned_plane_count, nonaligned_plane_count)
    )

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)