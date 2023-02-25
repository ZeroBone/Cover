# noinspection PyPackageRequirements
import z3 as z3
import sys


if __name__ == "__main__":
    n = int(sys.argv[1])

    xy = z3.Reals("x " + " ".join("y_%d" % (i + 1) for i in range(n)))

    phi = z3.And(
        z3.Sum(*[xy[i + 1] for i in range(n)]) - xy[0] < 0,
        -1 < z3.Sum(*[xy[i + 1] for i in range(n)]) - xy[0],
        *[
            z3.Or(xy[i] == 0, xy[i] == 2 ** i)
            for i in range(1, n + 1)
        ]
    )

    solver = z3.Solver()
    solver.add(phi)

    output_file_name = "data/add_%03d.smt2" % n

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)
