from typing import List

import z3

from formula_context import FormulaContext
from linconstraint import LinearConstraint


def compute_all_disjuncts(phi_context: FormulaContext, domain_lin_constraints: List[LinearConstraint], domain_formula):

    disjunct_solver = z3.Solver()
    disjunct_solver.add(domain_formula)

    while disjunct_solver.check() == z3.sat:
        model = disjunct_solver.model()
