import logging
from fractions import Fraction

import numpy as np
import z3

from context import VarDecContext
from gauss import compute_kernel, check_image_space_inclusion
from linconstraint import predicate_to_linear_constraint
from z3_utils import is_sat, is_valid, get_formula_predicates

_logger = logging.getLogger("vardec")


class FormulaContext:

    def __init__(self, phi, context: VarDecContext, /):
        self.phi = phi

        _logger.info("Formula:\n%s", phi)

        self.constraints = []

        for predicate in get_formula_predicates(phi):
            _logger.info("Predicate: %s", predicate)

            constraint = predicate_to_linear_constraint(context, predicate)

            if constraint not in self.constraints:
                self.constraints.append(constraint)

        _logger.info("Constraints:\n%s", ",\n".join((str(c) for c in self.constraints)))

    def get_constraint_count(self) -> int:
        return len(self.constraints)


def vardec(phi, x: list, y: list):

    context = VarDecContext(x, y)
    phi_context = FormulaContext(phi, context)

    global_solver = z3.Solver()

    global_solver.add(phi)

    print(global_solver.check())

    model = global_solver.model()
    model_vec = context.model_to_vec(model)

    model_vec_x = context.select_entries_corresp_x(model_vec)
    model_vec_y = context.select_entries_corresp_y(model_vec)

    print("Model: %s Projected onto: x: %s y: %s" % (model_vec, model_vec_x, model_vec_y))

    gamma = [constraint.get_version_satisfying_model(context, model_vec) for constraint in phi_context.constraints]
    gamma_eq_constraint_indices = []
    for i, constraint in enumerate(phi_context.constraints):
        if constraint.model_satisfies_equality_version(model_vec):
            gamma_eq_constraint_indices.append(i)

    assert is_sat(z3.And(*gamma))
    assert is_valid(z3.Implies(z3.And(*(v == model[v] for v in model)), z3.And(*gamma)))

    print("Gamma: %s" % gamma)
    print("Equality constraint id's: %s" % gamma_eq_constraint_indices)

    eq_constraint_mat_ker = kernel_table[frozenset(gamma_eq_constraint_indices)]

    print(eq_constraint_mat_ker)

    theta = []

    # we now translate the equality constraints into Pi-respecting formulas
    eq_constraint_mat_ker_x = context.select_rows_corresp_x(eq_constraint_mat_ker)
    eq_constraint_mat_ker_y = context.select_rows_corresp_y(eq_constraint_mat_ker)

    eq_constraint_lindep_x = compute_kernel(np.transpose(eq_constraint_mat_ker_x))
    eq_constraint_lindep_y = compute_kernel(np.transpose(eq_constraint_mat_ker_y))

    print("Linear dependencies of Gamma^= with respect to x: %s" % eq_constraint_lindep_x)
    print("Linear dependencies of Gamma^= with respect to y: %s" % eq_constraint_lindep_y)

    for lin_dependency_witness in np.transpose(eq_constraint_lindep_x):
        theta.append(
            context.x_or_y_linear_comb_to_z3_expr(lin_dependency_witness, True)
            == np.dot(model_vec_x, lin_dependency_witness))

    for lin_dependency_witness in np.transpose(eq_constraint_lindep_y):
        theta.append(
            context.x_or_y_linear_comb_to_z3_expr(lin_dependency_witness, False)
            == np.dot(model_vec_y, lin_dependency_witness))

    print("Theta: %s" % theta)

    assert is_valid(z3.Implies(z3.And(*gamma), z3.And(*theta)))

    disjunct_solver = z3.Solver()
    disjunct_solver.add(z3.And(*theta))

    print(disjunct_solver.check())

    print(disjunct_solver.model())
