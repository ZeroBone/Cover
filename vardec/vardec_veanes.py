"""
Original authors: Margus Veanes, Nikolaj Bjørner, Lev Nachmanson & Sergey Bereg
This version was adapted to work for arbitrary partitions by Daniel Stan
"""
import argparse

from z3 import *
import time

from z3_utils import get_formula_ast_node_count


def _fresh_vars(pref, l):
    """Returns a pair (vlist, vpack)"""
    vlist = []
    vpack = []
    for (i, part) in enumerate(l):
        vpack.append([])
        for (j, var) in enumerate(part):
            new = Const("%s_%d_%d" % (pref, i, j), var.sort())
            vlist.append(new)
            vpack[-1].append(new)
        vpack[-1] = tuple(vpack[-1])
    return vlist, vpack


def nu_ab(R, x, y, a, b):
    x_list, x_pack = _fresh_vars("x", x)
    y_list, y_pack = _fresh_vars("y", y)
    return Or(Exists(y_list, R(x + y_pack) != R(a + y_pack)),
              Exists(x_list, R(x_pack + y) != R(x_pack + b)))


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


def evaluate(m, part):
    """Given a model on a part (tuple), return a tuple of values"""
    return tuple(m.evaluate(var, True) for var in part)


def vardec_veanes(R, partition):
    phi = R(partition)
    if len(partition) == 1:
        return phi
    l = int(len(partition) / 2)
    x, y = partition[0:l], partition[l:]

    def dec(nu, pi):
        if is_unsat(And(pi, phi)):
            return BoolVal(False)
        if is_unsat(And(pi, Not(phi))):
            return BoolVal(True)
        fmls = [BoolVal(True), phi, pi]
        # try to extend nu
        m = last_sat(nu, None, fmls)
        # nu must be consistent
        assert m is not None
        a = [evaluate(m, part) for part in x]
        b = [evaluate(m, part) for part in y]
        psi_ab = And(R(a + y), R(x + b))
        phi_a = vardec_veanes(lambda z: R(a + z), y)
        phi_b = vardec_veanes(lambda z: R(z + b), x)
        nu.push()
        # exclude: x~a and y~b
        nu.add(nu_ab(R, x, y, a, b))
        t = dec(nu, And(pi, psi_ab))
        f = dec(nu, And(pi, Not(psi_ab)))
        nu.pop()
        return If(And(phi_a, phi_b), t, f)

    # nu is initially true
    return dec(Solver(), BoolVal(True))


def build_lambda(formula, partition):
    """The mondec algorithm requires a lambda function as an input, instantiating
    the formula with any list of free variables.
    Note that substitue() may have an overhead (direct lambda specification
    may be faster)
    """

    def function(value):
        args = []
        assert len(value) == len(partition)
        for (i, part) in enumerate(partition):
            assert len(part) == len(value[i])
            for assi in zip(part, value[i]):
                args.append(assi)
        return substitute(formula, *tuple(args))

    return function


def _main():
    parser = argparse.ArgumentParser(
        prog='mondec-classic',
        description='monadic or variadic decomposition (classic algorithm)',
        epilog='See Monadic Decomposition, by Margus Veanes, Nikolaj Bjørner, Lev Nachmanson & Sergey Bereg (CAV 2014)')
    parser.add_argument('filename')
    parser.add_argument('-p', '--part',
                        help="Specify part of the partition of variables. Default = one variable per part (monadic decomposition)",
                        action="extend", nargs="+",
                        default=[])
    parser.add_argument('-t', '--time', action='store_true',
                        help="Print execution time",
                        default=False)
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Do not print the formula",
                        default=False)
    parser.add_argument('-s', '--sanity', action='store_true',
                        help="Sanity check",
                        default=False)

    args = parser.parse_args()

    # Load formula from SMT2 file
    target_vector = parse_smt2_file(args.filename)
    # iterate the vector and from it build a AND formula
    formula = z3.And([form for form in target_vector])

    var = z3util.get_vars(formula)
    if not args.part:
        partition = [(v,) for v in var]
    else:
        # dict of str -> variable object
        dicto = {v.decl().name(): v for v in var}
        partition = []
        seen = set()
        for p_names in args.part:
            p_names = p_names.split(',')
            if seen.intersection(p_names):
                print("-p should not intersect with each others")
                sys.exit(1)
            seen.update(p_names)
            partition.append(tuple(dicto[v_name] for v_name in p_names))
        if len(var) != len(seen):
            print("Partition given but not all variable set was partitionned")
            sys.exit(2)

    start = time.perf_counter()
    result = vardec_veanes(build_lambda(formula, partition), partition)
    total = time.perf_counter() - start

    decomposition_size = get_formula_ast_node_count(result)

    print(f"R: ✓ Successfully decomposed (time: {total}s size: {decomposition_size})")
    if not args.quiet:
        print(result.sexpr())
    if args.sanity:
        solver = Solver()
        solver.add(formula != result)
        if solver.check() == sat:
            print("Result is wrong for %r" % solver.model())
            sys.exit(1)


if __name__ == '__main__':
    _main()
