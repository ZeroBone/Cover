from typing import List

import z3

from formula_context import FormulaContext
from linconstraint import LinearConstraint
from vardec_context import VarDecContext
from z3_utils import is_sat


def compute_all_disjuncts(
    context: VarDecContext,
    phi_context: FormulaContext,
    domain_lin_constraints: List[LinearConstraint],
    domain_formula
):

    disjunct_solver = z3.Solver()
    disjunct_solver.add(domain_formula)

    disj_count = 0

    while disjunct_solver.check() == z3.sat:
        disj_model_vec = context.model_to_vec(disjunct_solver.model())
        disjunct = [
            *(ct.get_version_satisfying_model(context, disj_model_vec) for ct in phi_context.constraints),
            *(ct.get_version_satisfying_model(context, disj_model_vec) for ct in domain_lin_constraints),
        ]

        assert is_sat(z3.And(*disjunct))

        print(disjunct)
        disj_count += 1

        disjunct_solver.add(z3.Or(
            *(z3.Not(d) for d in disjunct)
        ))

    print("Disjunct count: %d" % disj_count)
