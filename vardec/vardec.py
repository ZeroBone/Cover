import itertools
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


def _rational_to_z3_ratval(frac: Fraction, /) -> z3.RatVal:
    return z3.RatVal(frac.numerator, frac.denominator)


def cover(context: VarDecContext, phi_context: FormulaContext, gamma_model, gamma_additional_constraints: list = None):

    _logger.info("=== [Covering algorithm] ===")

    if gamma_additional_constraints is None:
        gamma_additional_constraints = []

    _logger.info(
        "Additional constraints: %s",
        [constraint.get_equality_expr(context) for constraint in gamma_additional_constraints]
    )

    gamma_model_vec = context.model_to_vec(gamma_model)
    gamma_model_vec_x = context.select_entries_corresp_x(gamma_model_vec)
    gamma_model_vec_y = context.select_entries_corresp_y(gamma_model_vec)

    _logger.info("Model: %s Projected onto: x: %s y: %s", gamma_model_vec, gamma_model_vec_x, gamma_model_vec_y)

    gamma = [
        *(constraint.get_version_satisfying_model(context, gamma_model_vec) for constraint in phi_context.constraints),
        *(constraint.get_equality_expr(context) for constraint in gamma_additional_constraints)
    ]

    gamma_eq_constraint_indices = [
        i for i, constraint in enumerate(phi_context.constraints)
        if constraint.model_satisfies_equality_version(gamma_model_vec)
    ]

    assert is_sat(z3.And(*gamma))
    assert is_valid(z3.Implies(z3.And(*(v == gamma_model[v] for v in gamma_model)), z3.And(*gamma)))

    _logger.debug("Gamma: %s", gamma)
    _logger.debug("Equality constraint id's: %s", gamma_eq_constraint_indices)

    # compute the equality constraints matrix

    gamma_eq_constraint_mat = np.array([
        *(phi_context.constraints[i].get_lin_combination_copy() for i in gamma_eq_constraint_indices),
        *(constraint.get_lin_combination_copy() for constraint in gamma_additional_constraints)
    ], dtype=Fraction)

    if gamma_eq_constraint_mat.shape[0] == 0:
        # the matrix is empty, that is, there are no equality constraints
        gamma_eq_constraint_mat = np.zeros((1, context.variable_count()), dtype=Fraction)

    _logger.debug("Gamma equality constraint matrix: %s", gamma_eq_constraint_mat)

    # compute the kernel of the equality constraints matrix

    gamma_eq_constraint_mat_ker = compute_kernel(gamma_eq_constraint_mat)

    _logger.debug("Gamma equality constraint matrix kernel: %s", gamma_eq_constraint_mat_ker)

    theta = []

    # we now translate the equality constraints into Pi-respecting formulas
    gamma_eq_constraint_mat_ker_x = context.select_rows_corresp_x(gamma_eq_constraint_mat_ker)
    gamma_eq_constraint_mat_ker_y = context.select_rows_corresp_y(gamma_eq_constraint_mat_ker)

    # analyze the matrix with the goal of determining whether the predicate set is Pi-simple or not
    for var_name, gamma_eq_constraint_mat_ker_var, context_var, gamma_model_vec_var in [
        ("X", gamma_eq_constraint_mat_ker_x, context.x, gamma_model_vec_x),
        ("Y", gamma_eq_constraint_mat_ker_y, context.y, gamma_model_vec_y)
    ]:
        if not np.any(gamma_eq_constraint_mat_ker_var):
            # the segment of the matrix corresponding to the var variable is zero
            # that is: var (either X or Y) is fixed
            _logger.info("Gamma is Pi-simple (%s is fixed), hence covering is trivial.", var_name)

            sigma = [
                (var, _rational_to_z3_ratval(gamma_model_vec_var[var_index]))
                for var_index, var in enumerate(context_var)
            ]

            decomposition = z3.And(
                *(z3.substitute(p, *sigma) for p in gamma),
                *(var == val for var, val in sigma)
            )

            assert is_valid(decomposition == z3.And(*gamma))
            return decomposition

    gamma_eq_constraint_mat_lindep_x = compute_kernel(np.transpose(gamma_eq_constraint_mat_ker_x))
    gamma_eq_constraint_mat_lindep_y = compute_kernel(np.transpose(gamma_eq_constraint_mat_ker_y))

    _logger.debug("Linear dependencies of Gamma^= with respect to x: %s", gamma_eq_constraint_mat_lindep_x)
    _logger.debug("Linear dependencies of Gamma^= with respect to y: %s", gamma_eq_constraint_mat_lindep_y)

    for lin_dependency_witness in np.transpose(gamma_eq_constraint_mat_lindep_x):
        theta.append(
            context.x_or_y_linear_comb_to_z3_expr(lin_dependency_witness, True)
            == np.dot(gamma_model_vec_x, lin_dependency_witness))

    for lin_dependency_witness in np.transpose(gamma_eq_constraint_mat_lindep_y):
        theta.append(
            context.x_or_y_linear_comb_to_z3_expr(lin_dependency_witness, False)
            == np.dot(gamma_model_vec_y, lin_dependency_witness))

    _logger.info("Theta: %s", theta)

    assert is_valid(z3.Implies(z3.And(*gamma), z3.And(*theta)))

    delta = []
    upsilon_lt_gt = []
    upsilon_neq = []

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

        if len(omega_eq_constraint_indices) > 0:
            omega_eq_constraint_mat = np.array([
                phi_context.constraints[i].get_lin_combination_copy() for i in omega_eq_constraint_indices
            ], dtype=Fraction)
        else:
            omega_eq_constraint_mat = np.zeros((1, context.variable_count()), dtype=Fraction)

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

            _logger.info("w predicate: %s", w_lhs == w_rhs)

            assert is_valid(z3.Implies(z3.And(*omega_eq_constraints), w_lhs == w_rhs))

            w_predicate_lt = w_lhs < w_rhs
            w_predicate_gt = w_lhs > w_rhs

            if is_valid(z3.Implies(z3.And(*gamma), w_predicate_lt)):
                _logger.info("Excellent! Gamma entails the < version of the w predicate.")
                disjunct_solver.add(w_predicate_lt)
                upsilon_lt_gt.append(w_predicate_lt)
            elif is_valid(z3.Implies(z3.And(*gamma), w_predicate_gt)):
                _logger.info("Excellent! Gamma entails the > version of the w predicate.")
                disjunct_solver.add(w_predicate_gt)
                upsilon_lt_gt.append(w_predicate_gt)
            else:
                _logger.info("Gamma unfortunately entails neither the < nor the > version of the w predicate.")
                disjunct_solver.add(w_lhs != w_rhs)
                upsilon_neq.append((w_lhs, w_rhs))

            rec_solver = z3.Solver()
            rec_solver.add(z3.And(*gamma))
            rec_solver.add(w_lhs == w_rhs)

            if rec_solver.check() == z3.sat:
                rec_model = rec_solver.model()
                _logger.info("Gamma \\cup {w} is satisfiable. Model: %s. This means we recurse.", rec_model)

                # make sure that Gamma doesn't lose any models
                rec_cover = cover(
                    context,
                    phi_context,
                    rec_model,
                    gamma_additional_constraints + [predicate_to_linear_constraint(context, w_lhs == w_rhs)]
                )

                delta.append(rec_cover)

            else:
                _logger.info("Gamma \\cup {w} is unsatisfiable, no recursion happens.")

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

    _logger.debug("Upsilon^{<>} = %s", upsilon_lt_gt)
    _logger.debug("Upsilon^{\\neq} = %s", upsilon_neq)

    decomposition_disjuncts = []

    for subset_size in range(len(upsilon_neq) + 1):
        for subset in itertools.combinations((i for i in range(len(upsilon_neq))), subset_size):
            subset_set = set(subset)
            current_decomposition_disjunct = []
            for i, (d_lhs, d_rhs) in enumerate(upsilon_neq):
                if i in subset_set:
                    # consider the < version
                    current_decomposition_disjunct.append(d_lhs < d_rhs)
                else:
                    # consider the > version
                    current_decomposition_disjunct.append(d_lhs > d_rhs)

            if is_sat(z3.And(*gamma, *current_decomposition_disjunct)):
                decomposition_disjuncts.append(z3.And(*current_decomposition_disjunct))
            else:
                _logger.info("Ignoring disjunct corresponding to %s", subset_set)

    decomposition = z3.Or(
        z3.And(*theta, *upsilon_lt_gt, z3.Or(*decomposition_disjuncts)),
        *delta
    )

    _logger.info("Covering call returns decomposition:\n%s", decomposition)

    return decomposition


def vardec(phi, x: list, y: list, debug_mode=True):

    context = VarDecContext(x, y)
    phi_context = FormulaContext(phi, context)

    phi_dec = []

    entailment_solver = z3.Solver()
    entailment_solver.add(z3.Not(phi))

    global_solver = z3.Solver()
    global_solver.add(phi)

    while global_solver.check() == z3.sat:
        gamma_model = global_solver.model()
        psi = cover(context, phi_context, gamma_model)

        _logger.info("Covering algorithm produced psi:\n%s", psi)

        if debug_mode:
            assert is_valid(z3.Implies(
                z3.And(*(
                    constraint.get_version_satisfying_model(
                        context,
                        context.model_to_vec(gamma_model)
                    )
                    for constraint in phi_context.constraints
                )),
                psi
            ))

        entailment_solver.push()
        entailment_solver.add(psi)

        if entailment_solver.check() == z3.sat:
            # phi is not Pi-decomposable
            return None

        entailment_solver.pop()

        phi_dec.append(psi)

        global_solver.add(z3.Not(psi))

    return z3.Or(*phi_dec)
