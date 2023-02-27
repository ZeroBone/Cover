# noinspection PyPackageRequirements
import z3 as z3
import sys


if __name__ == "__main__":
    width = int(sys.argv[1])
    count = int(sys.argv[2])

    print("Generating %d cubes of width %d" % (count, width))

    x, y = z3.Reals("x y")
    disjuncts = []
    for left_bottom_corner in range(count):
        disjuncts.append(z3.And([
            left_bottom_corner <= x, x <= (left_bottom_corner + width),
            left_bottom_corner <= y, y <= (left_bottom_corner + width)
        ]))

    solver = z3.Solver()
    solver.add(z3.Or(disjuncts))

    output_file_name = "data/cube_width%03d_count%03d.smt2" % (width, count)

    fh = open(output_file_name, "w")
    fh.write(solver.sexpr())
    fh.close()

    print("Generated .smt2 file '%s'." % output_file_name)
