import logging
from fractions import Fraction

import numpy as np
import z3

from context import VarDecContext
from gauss import compute_kernel
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


def cover(context: VarDecContext, phi_context: FormulaContext, gamma: list, /):
    pass


def vardec(phi, x: list, y: list):

    context = VarDecContext(x, y)
    phi_context = FormulaContext(phi, context)

    global_solver = z3.Solver()
    global_solver.add(phi)

    while global_solver.check() == z3.sat:
        phi_model = global_solver.model()
        phi_model_vec = context.model_to_vec(phi_model)
        phi_model_vec_x = context.select_entries_corresp_x(phi_model_vec)
        phi_model_vec_y = context.select_entries_corresp_y(phi_model_vec)

        _logger.info("Model: %s Projected onto: x: %s y: %s", phi_model_vec, phi_model_vec_x, phi_model_vec_y)

        gamma = [
            constraint.get_version_satisfying_model(context, phi_model_vec) for constraint in phi_context.constraints
        ]

        gamma_eq_constraint_indices = [
            i for i, constraint in enumerate(phi_context.constraints)
            if constraint.model_satisfies_equality_version(phi_model_vec)
        ]

        assert is_sat(z3.And(*gamma))
        assert is_valid(z3.Implies(z3.And(*(v == phi_model[v] for v in phi_model)), z3.And(*gamma)))

        _logger.debug("Gamma: %s", gamma)
        _logger.debug("Equality constraint id's: %s", gamma_eq_constraint_indices)

        # compute the equality constraints matrix

        gamma_eq_constraint_mat = np.array([
            phi_context.constraints[i].get_lin_combination_copy() for i in gamma_eq_constraint_indices
        ], dtype=Fraction)

        _logger.debug("Gamma equality constraint matrix: %s", gamma_eq_constraint_mat)

        # compute the kernel of the equality constraints matrix

        gamma_eq_constraint_mat_ker = compute_kernel(gamma_eq_constraint_mat)

        _logger.debug("Gamma equality constraint matrix kernel: %s", gamma_eq_constraint_mat_ker)

        theta = []

        # we now translate the equality constraints into Pi-respecting formulas
        gamma_eq_constraint_mat_ker_x = context.select_rows_corresp_x(gamma_eq_constraint_mat_ker)
        gamma_eq_constraint_mat_ker_y = context.select_rows_corresp_y(gamma_eq_constraint_mat_ker)

        gamma_eq_constraint_lindep_x = compute_kernel(np.transpose(gamma_eq_constraint_mat_ker_x))
        gamma_eq_constraint_lindep_y = compute_kernel(np.transpose(gamma_eq_constraint_mat_ker_y))

        _logger.debug("Linear dependencies of Gamma^= with respect to x: %s", gamma_eq_constraint_lindep_x)
        _logger.debug("Linear dependencies of Gamma^= with respect to y: %s", gamma_eq_constraint_lindep_y)

        for lin_dependency_witness in np.transpose(gamma_eq_constraint_lindep_x):
            theta.append(
                context.x_or_y_linear_comb_to_z3_expr(lin_dependency_witness, True)
                == np.dot(phi_model_vec_x, lin_dependency_witness))

        for lin_dependency_witness in np.transpose(gamma_eq_constraint_lindep_y):
            theta.append(
                context.x_or_y_linear_comb_to_z3_expr(lin_dependency_witness, False)
                == np.dot(phi_model_vec_y, lin_dependency_witness))

        print("Theta: %s" % theta)

        assert is_valid(z3.Implies(z3.And(*gamma), z3.And(*theta)))

        disjunct_solver = z3.Solver()
        disjunct_solver.add(z3.And(*theta))

        while disjunct_solver.check() == z3.sat:
            omega_model = disjunct_solver.model()
            omega_model_vec = context.model_to_vec(omega_model)
            omega_model_vec_x = context.select_entries_corresp_x(omega_model_vec)
            omega_model_vec_y = context.select_entries_corresp_y(omega_model_vec)

            _logger.debug("Omega model: %s", omega_model_vec)

            # determine the equality constraints the disjunct containing the model satisfies

            omega_eq_constraint_indices = [
                i for i, constraint in enumerate(phi_context.constraints)
                if constraint.model_satisfies_equality_version(omega_model_vec)
            ]

            _logger.debug("Omega equality constraint id's: %s", omega_eq_constraint_indices)

            # compute the equality constraints matrix

            omega_eq_constraint_mat = np.array([
                phi_context.constraints[i].get_lin_combination_copy() for i in omega_eq_constraint_indices
            ], dtype=Fraction)

            omega_eq_constraint_mat_ker = compute_kernel(omega_eq_constraint_mat)

            omega_eq_constraint_mat_ker_x = context.select_rows_corresp_x(omega_eq_constraint_mat_ker)
            omega_eq_constraint_mat_ker_y = context.select_rows_corresp_y(omega_eq_constraint_mat_ker)

            # TODO: remove
            return

        # TODO: remove
        return
