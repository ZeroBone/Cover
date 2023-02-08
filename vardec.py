from context import VarDecContext
from linconstraint import predicate_to_linear_constraint
from z3_utils import get_formula_predicates


def vardec(phi, x: list, y: list):

    context = VarDecContext(x, y)

    print("Formula:", phi)

    for predicate in get_formula_predicates(phi):
        print("Predicate: %s" % predicate)

        constraint = predicate_to_linear_constraint(context, predicate)

        print("Constraint: %s" % constraint)

