import os
from pathlib import Path

# noinspection PyPackageRequirements
import z3 as z3
import sys


def _resolve_formula_class_dir():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark/data/xorspace").resolve()


if __name__ == "__main__":

    dim = int(sys.argv[1])

    assert dim >= 2

    print("Dimension: %d" % dim)

    v = z3.Reals(" ".join("x_%d" % (i + 1) for i in range(dim)))

    phi = v[0] < 0

    for d in range(1, dim):
        phi = z3.Xor(phi, v[d] < 0)

    # put everything together

    solver = z3.Solver()
    solver.add(phi)

    os.makedirs(_resolve_formula_class_dir(), exist_ok=True)
    output_file_name = os.path.join(
        _resolve_formula_class_dir(),
        "xorspace_dim%03d.smt2" % dim
    )

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)
