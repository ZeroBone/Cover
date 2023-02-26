from fractions import Fraction

# noinspection PyPackageRequirements
import z3 as z3
import sys
import random


if __name__ == "__main__":

    random.seed(0xdeadbeef)

    dimension = int(sys.argv[1])
    grid_plane_count = int(sys.argv[2])
    nondec_plane_count = int(sys.argv[3])

    assert dimension >= 2

    print("Dimension: %d Grid hyperplane count: %d Non-decomposable hyperplane count: %d" %
          (dimension, grid_plane_count, nondec_plane_count))

    v = z3.Reals(" ".join("x_%d" % (i + 1) for i in range(dimension)))

    # generate grid

    grid_predicates = []

    for grid_offset in range(grid_plane_count):
        grid_predicates.append(v[0] == (grid_offset if grid_offset % 2 == 0 else -grid_offset))

    # generate non-decomposable hyperplanes intersecting the grid

    nondec_plane_predicates = []

    for _ in range(nondec_plane_count):
        beta_var = random.randrange(dimension)

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

    output_file_name = "data/grid_%04d_%04d_%04d.smt2" % (dimension, grid_plane_count, nondec_plane_count)

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)
