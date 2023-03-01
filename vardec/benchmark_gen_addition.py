import os.path
from pathlib import Path

# noinspection PyPackageRequirements
import z3 as z3
import sys


def _resolve_formula_class_dir():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark/data/add").resolve()


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

    os.makedirs(_resolve_formula_class_dir(), exist_ok=True)
    output_file_name = os.path.join(_resolve_formula_class_dir(), "add_n%03d.smt2" % n)

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)
