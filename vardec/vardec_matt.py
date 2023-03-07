
import argparse
import logging
import time

from itertools import chain
from typing import List

from formula_context import FormulaContext
from linear_constraint import LinearConstraint
from partition import Partition, PartitionException
from partition import get_singleton_partition, parse_formula_variable_partition
from presvardec import _load_formula_from_smt2, _load_formula_from_repl
from vardec_context import VarDecContext

from z3 import \
    And, ArithRef, BoolRef, Const, Not, Or, Solver, sat, substitute, unsat, z3util

_logger = logging.getLogger("vardec")

class CheckFailureException(Exception):
    pass

def constraint_expr(
    constraint : LinearConstraint, context : VarDecContext
) -> ArithRef:
    """Convert f(x, y) = c to f(x, y) - c expression
    Where x, y instantiated from context"""
    return constraint.get_lhs_linear_combination_expr(context) \
        - constraint.get_rhs_constrant()

def fmla_same_profile(
    context1 : VarDecContext, context2 : VarDecContext, fmla_context : FormulaContext
) -> BoolRef:
    """Get fmla asserting variables in both contexts have same profile
    That is, agree on sign of all LHS of constraints in fmla_context"""
    conjuncts = []
    for constraint in fmla_context.constraints:
        expr1 = constraint_expr(constraint, context1)
        expr2 = constraint_expr(constraint, context2)

        conjuncts.append(Or(
            And(expr1 < 0, expr2 < 0),
            And(expr1 == 0, expr2 == 0),
            And(expr1 > 0, expr2 > 0)
        ))

    return And(conjuncts)

def fmla_context1_strict_neighbour_context2(
    context1 : VarDecContext, context2 : VarDecContext, fmla_context : FormulaContext
) -> BoolRef:
    """Get fmla asserting context2 is just off the lines of context2
    I.e. both agree on < and > signs, but if context1 is ==, context2
    can be <, =, or >. This is does not prevent both having same
    profile."""
    conjuncts = []
    for constraint in fmla_context.constraints:
        expr1 = constraint_expr(constraint, context1)
        expr2 = constraint_expr(constraint, context2)

        conjuncts.append(Or(
            And(expr1 < 0, expr2 < 0),
            And(expr1 == 0), # expr2 can be anything!
            And(expr1 > 0, expr2 > 0)
        ))

    return And(conjuncts)

def fmla_has_equality(
    context : VarDecContext, fmla_context : FormulaContext
) -> BoolRef:
    """Get fmla asserting that at least one constraint has = sign"""
    return Or([
        constraint_expr(constraint, context) == 0
        for constraint in fmla_context.constraints
    ])

def fmla_vardec_test(
    phi, xs : List[Const], ys : List[Const], context : VarDecContext
) -> BoolRef:
    """Build the fmla testing non-decomposability.
    I.e. true if not decomposable"""
    # formula is
    # (E)(x1,y1)(x2,y2) such that
    #     Profile(x1,y1) = Profile(x2,y2) and has at least one equality
    #     and Profile(x1,y2) disagrees with Profile(x1,y1) only on
    #           equalities in Profile(x1,y1)
    #     and phi(x1,y1) = phi(x2,y2) != phi(x1,y2)

    fmla_context = FormulaContext(phi, context)

    xs1 = [ Const("mx1_" + str(x), x.sort()) for x in xs ]
    ys1 = [ Const("my1_" + str(y), y.sort()) for y in ys ]
    xs2 = [ Const("mx2_" + str(x), x.sort()) for x in xs ]
    ys2 = [ Const("my2_" + str(y), y.sort()) for y in ys ]

    context1 = VarDecContext(xs1, ys1)
    context2 = VarDecContext(xs2, ys2)
    context_swapped = VarDecContext(xs1, ys2)

    same_profile = fmla_same_profile(context1, context2, fmla_context)
    has_equality = fmla_has_equality(context1, fmla_context)
    neighbour = fmla_context1_strict_neighbour_context2(
        context1, context_swapped, fmla_context
    )

    phi1 = substitute(phi, *zip(chain(xs, ys), chain(xs1, ys1)))
    phi2 = substitute(phi, *zip(chain(xs, ys), chain(xs2, ys2)))
    phi_swapped = substitute(phi, *zip(chain(xs, ys), chain(xs1, ys2)))

    agree_disagree = Or(
        And(phi1, phi2, Not(phi_swapped)),
        And(Not(phi1), Not(phi2), phi_swapped)
    )

    return And(same_profile, has_equality, neighbour, agree_disagree)

def vardec_matt_binary(
    phi, xs : List[Const], ys : List[Const], context : VarDecContext
) -> bool:
    #print(phi, xs, ys)
    s = Solver()
    fmla_test = fmla_vardec_test(phi, xs, ys, context)
    s.add(fmla_test)
    result = s.check()
    if result == sat:
        #print(fmla_test)
        #print(s.model())
        return False
    elif result == unsat:
        return True
    else:
        raise CheckFailureException("Could not determine value of test formula")

def vardec_matt(phi, pi: Partition) -> bool:
    if pi.is_unary():
        return True

    eq_binary_partitions = pi.get_equivalent_list_of_binary_partitions()

    _logger.info("List of partitions equivalent to Pi:\n%s", eq_binary_partitions)

    assert len(eq_binary_partitions) > 0

    for xs, ys in eq_binary_partitions:
        context = VarDecContext(xs, ys)
        result = vardec_matt_binary(phi, xs, ys, context)
        if result is False:
            # the formula is not decomposable
            return False

    return True

def run_matt_vardec_benchmark(phi, pi: Partition):
    _start = time.perf_counter()
    result = vardec_matt(phi, pi)
    _end = time.perf_counter()
    return _end - _start, result

def _main():
    parser = argparse.ArgumentParser(
        prog='vardec-matt',
        description='variadic decomposition (matt\'s test)',
        epilog='See matt\'s note')
    parser.add_argument("-f", "--formula", metavar="FILE",
                        help="path to the .smt2 file containing the formula",
                        type=argparse.FileType("r", encoding="UTF-8"))
    parser.add_argument('-p', '--pi',
                        help="Specify part of the partition of variables. Default = singleton partition",
                        action="extend", nargs="+",
                        default=[])

    args = parser.parse_args()

    if args.formula is not None:
        phi = _load_formula_from_smt2(args)
        pi = None
    else:
        phi, pi = _load_formula_from_repl()

    phi_vars = z3util.get_vars(phi)

    # did the user specify a partition?
    # if yes, then use it
    if pi is None and args.pi is not None:
        try:
            pi = parse_formula_variable_partition(phi_vars, args.pi)
        except PartitionException as e:
            print("Partition is not specified correctly: %s" % e)
            return

    if pi is None:
        # still the partition is missing
        # fall back to the singleton partition
        pi = get_singleton_partition(phi_vars)

    assert isinstance(pi, Partition)

    start = time.perf_counter()
    result = vardec_matt(phi, pi)
    total = time.perf_counter() - start

    if result:
        print(f"R: decomposable (time: {total}s)")
    else:
        print(f"R: not decomposable (time: {total}s)")

if __name__ == '__main__':
    _main()
