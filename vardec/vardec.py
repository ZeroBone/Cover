import logging
from fractions import Fraction

import numpy as np
import z3

from context import VarDecContext
from gauss import compute_kernel, compute_gen_set_of_intersection_of_mat_images
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

        gamma_eq_constraint_mat_lindep_x = compute_kernel(np.transpose(gamma_eq_constraint_mat_ker_x))
        gamma_eq_constraint_mat_lindep_y = compute_kernel(np.transpose(gamma_eq_constraint_mat_ker_y))

        _logger.debug("Linear dependencies of Gamma^= with respect to x: %s", gamma_eq_constraint_mat_lindep_x)
        _logger.debug("Linear dependencies of Gamma^= with respect to y: %s", gamma_eq_constraint_mat_lindep_y)

        for lin_dependency_witness in np.transpose(gamma_eq_constraint_mat_lindep_x):
            theta.append(
                context.x_or_y_linear_comb_to_z3_expr(lin_dependency_witness, True)
                == np.dot(phi_model_vec_x, lin_dependency_witness))

        for lin_dependency_witness in np.transpose(gamma_eq_constraint_mat_lindep_y):
            theta.append(
                context.x_or_y_linear_comb_to_z3_expr(lin_dependency_witness, False)
                == np.dot(phi_model_vec_y, lin_dependency_witness))

        print("Theta: %s" % theta)

        assert is_valid(z3.Implies(z3.And(*gamma), z3.And(*theta)))

        upsilon = []

        disjunct_solver = z3.Solver()
        disjunct_solver.add(z3.And(*theta))

        while disjunct_solver.check() == z3.sat:
            omega_model = disjunct_solver.model()
            omega_model_vec = context.model_to_vec(omega_model)
            omega_model_vec_x = context.select_entries_corresp_x(omega_model_vec)
            omega_model_vec_y = context.select_entries_corresp_y(omega_model_vec)

            _logger.info("Found new disjunct Omega corresponding to model %s", omega_model_vec)

            # determine the equality constraints the disjunct containing the model satisfies

            omega_eq_constraint_indices = []
            not_omega_eq_constraint_indices = []

            for i, constraint in enumerate(phi_context.constraints):
                if constraint.model_satisfies_equality_version(omega_model_vec):
                    omega_eq_constraint_indices.append(i)
                else:
                    not_omega_eq_constraint_indices.append(i)

            _logger.debug(
                "Omega equality constraint id's: %s Id's of absent constraints: %s",
                omega_eq_constraint_indices,
                not_omega_eq_constraint_indices
            )

            # translate the computed indices of constraints into actuall constraints

            omega_eq_constraints = [
                phi_context.constraints[i].get_equality_expr(context) for i in omega_eq_constraint_indices
            ]

            _logger.debug("Omega equality constraints: %s" % omega_eq_constraints)

            assert len(omega_eq_constraints) == len(omega_eq_constraint_indices)

            # compute the equality constraints matrix

            omega_eq_constraint_mat = np.array([
                phi_context.constraints[i].get_lin_combination_copy() for i in omega_eq_constraint_indices
            ], dtype=Fraction)

            omega_eq_constraint_mat_ker = compute_kernel(omega_eq_constraint_mat)

            omega_eq_constraint_mat_ker_x = context.select_rows_corresp_x(omega_eq_constraint_mat_ker)
            omega_eq_constraint_mat_ker_y = context.select_rows_corresp_y(omega_eq_constraint_mat_ker)

            omega_eq_constraint_mat_lindep_x = compute_kernel(np.transpose(omega_eq_constraint_mat_ker_x))
            omega_eq_constraint_mat_lindep_y = compute_kernel(np.transpose(omega_eq_constraint_mat_ker_y))

            _logger.info("Linear dependencies of Omega^= with respect to x:\n%s", omega_eq_constraint_mat_lindep_x)
            _logger.info("Linear dependencies of Omega^= with respect to y:\n%s", omega_eq_constraint_mat_lindep_y)

            x_lindep_in_omega_but_not_in_gamma = compute_gen_set_of_intersection_of_mat_images(
                omega_eq_constraint_mat_lindep_x, gamma_eq_constraint_mat_ker_x
            )
            y_lindep_in_omega_but_not_in_gamma = compute_gen_set_of_intersection_of_mat_images(
                omega_eq_constraint_mat_lindep_y, gamma_eq_constraint_mat_ker_y
            )

            _logger.info("Linear dependencies present in Omega but absent in Gamma, with respect to x:\n%s",
                         x_lindep_in_omega_but_not_in_gamma)
            _logger.info("Linear dependencies present in Omega but absent in Gamma, with respect to y:\n%s",
                         y_lindep_in_omega_but_not_in_gamma)

            w_predicate = None

            for wrt_x, lindep_diff, omega_model_vec_part in [
                (True, x_lindep_in_omega_but_not_in_gamma, omega_model_vec_x),
                (False, y_lindep_in_omega_but_not_in_gamma, omega_model_vec_y)
            ]:
                for w in np.transpose(lindep_diff):
                    if not np.any(w):
                        # this column is a zero-column
                        continue
                    _logger.info("Found witness that Omega has more linear dependencies compared to Gamma.")
                    _logger.info("The witness is: w = %s", w)
                    w_predicate = context.x_or_y_linear_comb_to_z3_expr(w, wrt_x), np.dot(omega_model_vec_part, w)
                    break

                if w_predicate is not None:
                    break

            if w_predicate is not None:
                w_lhs, w_rhs = w_predicate
                w_predicate_eq = w_lhs == w_rhs

                _logger.info("w predicate: %s", w_predicate_eq)

                assert is_valid(z3.Implies(z3.And(*omega_eq_constraints), w_predicate_eq))

                w_predicate_lt = w_lhs < w_rhs
                w_predicate_gt = w_lhs > w_rhs

                if is_valid(z3.Implies(z3.And(*gamma), w_predicate_lt)):
                    _logger.info("Excellent! Gamma entails the < version of the w predicate.")
                    disjunct_solver.add(w_predicate_lt)
                    upsilon.append(w_predicate_lt)
                elif is_valid(z3.Implies(z3.And(*gamma), w_predicate_gt)):
                    _logger.info("Excellent! Gamma entails the > version of the w predicate.")
                    disjunct_solver.add(w_predicate_gt)
                    upsilon.append(w_predicate_gt)
                else:
                    _logger.info("Gamma unfortunately entails neither the < nor the > version of the w predicate.")
                    disjunct_solver.add(w_lhs != w_rhs)
                    upsilon.append(w_lhs != w_rhs)

            else:
                _logger.info(
                    "The current disjunct Omega cannot be distinguished from Gamma in the language of Pi-decompositions"
                )

                disjunct_solver.add(z3.Or(
                    # new disjuncts must either disagree with Omega on some of its equality predicates
                    *(z3.Not(eq) for eq in omega_eq_constraints),
                    # or they must contain strictly more equality predicates compared to Omega
                    *(phi_context.constraints[i].get_equality_expr(context) for i in not_omega_eq_constraint_indices)
                ))

        print(upsilon)

        # TODO: remove
        return
