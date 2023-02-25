"""
Authors: Margus Veanes, Nikolaj Bjørner, Lev Nachmanson & Sergey Bereg
"""
import time

# noinspection PyPackageRequirements
from z3 import *


def nu_ab(R, x, y, a, b):
    x_ = [Const("x_%d" % i, x[i].sort()) for i in range(len(x))]
    y_ = [Const("y_%d" % i, y[i].sort()) for i in range(len(y))]
    return Or(Exists(y_, R(x + y_) != R(a + y_)), Exists(x_, R(x_ + y) != R(x_ + b)))


def is_unsat(fml):
    s = Solver()
    s.add(fml)
    return unsat == s.check()


def last_sat(s, m, fmls):
    if len(fmls) == 0:
        return m
    s.push()
    s.add(fmls[0])
    if s.check() == sat:
        m = last_sat(s, s.model(), fmls[1:])
    s.pop()
    return m


def mondec_veanes(R, variables):
    # print(variables)
    phi = R(variables)
    if len(variables) == 1:
        return phi
    l = int(len(variables) / 2)
    x, y = variables[0:l], variables[l:]

    def dec(nu, pi):
        if is_unsat(And(pi, phi)):
            return BoolVal(False)
        if is_unsat(And(pi, Not(phi))):
            return BoolVal(True)
        fmls = [BoolVal(True), phi, pi]
        # try to extend nu
        m = last_sat(nu, None, fmls)
        # nu must be consistent
        assert (m is not None)
        a = [m.evaluate(z, True) for z in x]
        b = [m.evaluate(z, True) for z in y]
        psi_ab = And(R(a + y), R(x + b))
        phi_a = mondec_veanes(lambda z: R(a + z), y)
        phi_b = mondec_veanes(lambda z: R(z + b), x)
        nu.push()
        # exclude: x~a and y~b
        nu.add(nu_ab(R, x, y, a, b))
        t = dec(nu, And(pi, psi_ab))
        f = dec(nu, And(pi, Not(psi_ab)))
        nu.pop()
        return If(And(phi_a, phi_b), t, f)

    # nu is initially true
    return dec(Solver(), BoolVal(True))


def test_mondec(k):
    R = lambda v: And(v[1] > 0, (v[1] & (v[1] - 1)) == 0,
                      (v[0] & (v[1] % ((1 << k) - 1))) != 0)
    bvs = BitVecSort(2 * k)  # use 2k-bit bitvectors
    x, y = Consts('x y', bvs)
    res = mondec_veanes(R, [x, y])
    assert (is_unsat(res != R([x, y])))  # check correctness
    print("mondec1(", R([x, y]), ") =", res)


def test_mondec1(k):
    R = lambda v: And(v[0] + v[1] >= k, v[0] >= 0, v[1] >= 0)
    x, y = Consts('x y', IntSort())
    res = mondec_veanes(R, [x, y])
    assert (is_unsat(res != R([x, y])))  # check correctness
    print("mondec1(", R([x, y]), ") =", res)


def test_mondec2(k):
    R = lambda v: Or(Or([And(v[0] <= i + 2, v[0] >= i, v[1] >= i, v[1] <= i + 2) for
                         i in range(1, k)]),
                     And(v[0] + v[1] == k, v[0] >= 0, v[1] >= 0))
    x, y = Consts('x y', IntSort())
    res = mondec_veanes(R, [x, y])
    assert (is_unsat(res != R([x, y])))  # check correctness
    print("mondec1(", R([x, y]), ") =", res)


def test_mondec3(k):
    R = lambda v: Or([And(v[0] <= i + 2, v[0] >= i, v[1] >= i, v[1] <= i + 2) for
                      i in range(1, k)])
    x, y = Consts('x y', IntSort())
    res = mondec_veanes(R, [x, y])
    assert (is_unsat(res != R([x, y])))  # check correctness
    print("mondec1(", R([x, y]), ") =", res)


def test_mondec4():
    # not monadically decomposable, will not terminate
    R = lambda v: v[0] == v[1]
    x, y = Ints("x y")
    mondec_veanes(R, [x, y])


def test_mondec5():
    R = lambda v: And(v[1] <= -2 * v[0] + 5, v[0] >= 0, v[1] >= 0)
    x, y = Ints("x y")
    mondec_veanes(R, [x, y])


def _load_formula(phi):

    formula_vars = z3util.get_vars(phi)

    def lambda_model(value):
        """The mondec algorithm requires a lambda function as an input, instantiating
        the formula with any list of free variables.
        Note that substitue() may have an overhead (direct lambda specification
        may be faster)
        """
        return substitute(phi, *((vname, value[i]) for (i, vname) in enumerate(formula_vars)))

    return phi, formula_vars, lambda_model


def _load_smt(file_name):
    target_vector = parse_smt2_file(file_name)

    # take the formulas from the file in conjunction
    phi = z3.And([form for form in target_vector])

    return _load_formula(phi)


def run_veanes_benchmark(phi):
    _start = time.perf_counter()
    formula, var, lambda_model = _load_formula(phi)
    result = mondec_veanes(lambda_model, var)
    _end = time.perf_counter()
    return _end - _start, len(result.sexpr())


def _main():
    if len(sys.argv) < 2:
        print("Usage: %s [filename.smt2]")
        sys.exit(0)

    file_name = sys.argv[-1]

    formula, var, lambda_model = _load_smt(file_name)

    start = time.perf_counter()
    result = mondec_veanes(lambda_model, var)
    end = time.perf_counter()

    if "-q" not in sys.argv:
        print("Res: %r" % result)

    print(f"Successfully decomposed (time: {end - start}s size: {len(result.sexpr())})")


if __name__ == '__main__':
    _main()
