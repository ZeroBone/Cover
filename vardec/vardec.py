import itertools
import logging
from fractions import Fraction
from typing import List

import numpy as np
import z3

from context import VarDecContext, block_str
from gauss import compute_kernel, compute_gen_set_of_intersection_of_mat_images
from linconstraint import predicate_to_linear_constraint, LinearConstraint
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

        # create solver for efficiently checking entailment queries

        self._entailment_solver = z3.Solver()
        self._entailment_solver.add(z3.Not(phi))

    def query_whether_formula_entails_phi(self, query_formula, /) -> bool:
        self._entailment_solver.push()
        self._entailment_solver.add(query_formula)
        result = self._entailment_solver.check()
        self._entailment_solver.pop()
        assert result == z3.sat or result == z3.unsat
        return result == z3.unsat

    def get_constraint_count(self) -> int:
        return len(self.constraints)


def _rational_to_z3_ratval(frac: Fraction, /) -> z3.RatVal:
    return z3.RatVal(frac.numerator, frac.denominator)


def _matrix_add_zero_row_if_empty(mat: np.ndarray, mat_cols: int, /):
    if mat.shape[0] == 0:
        return np.zeros((1, mat_cols), dtype=Fraction)
    return mat


def cover(
    context: VarDecContext,
    phi_context: FormulaContext,
    gamma_model,
    gamma_additional_constraints: List[LinearConstraint] = None, /, *,
    debug_mode: bool = True,
    use_heuristics: bool = True
):

    _logger.info("=== [Covering algorithm] ===")

    if gamma_additional_constraints is None:
        gamma_additional_constraints = []

    if debug_mode:
        _logger.debug(
            "Additional constraints: %s",
            [constraint.get_equality_expr(context) for constraint in gamma_additional_constraints]
        )

    gamma_model_vec = context.model_to_vec(gamma_model)
    gamma_model_vec_proj = tuple(
        context.project_vector_onto_block(gamma_model_vec, b)
        for b in (VarDecContext.X, VarDecContext.Y)
    )

    if debug_mode:
        _logger.debug("Model: %s", gamma_model_vec)

    gamma = [
        *(ct.get_version_satisfying_model(context, gamma_model_vec) for ct in phi_context.constraints),
        *(ct.get_equality_expr(context) for ct in gamma_additional_constraints)
    ]

    # choose predicates respecting the partition
    theta = [p for p in gamma if context.predicate_respects_pi(p)]

    _logger.info("Initialized Theta to be the set of Pi-respecting predicates in Gamma, that is:\nTheta = %s", theta)

    gamma_eq_constraint_indices = [
        i for i, constraint in enumerate(phi_context.constraints)
        if constraint.model_satisfies_equality_version(gamma_model_vec)
    ]

    if debug_mode:
        assert is_sat(z3.And(*gamma))
        assert is_valid(z3.Implies(z3.And(*(v == gamma_model[v] for v in gamma_model)), z3.And(*gamma)))
        _logger.debug("Gamma: %s", gamma)
        _logger.debug("Equality constraint id's: %s", gamma_eq_constraint_indices)

    # compute the equality constraints matrix
    # TODO: implement caching for this

    gamma_eq_constraint_mat = _matrix_add_zero_row_if_empty(
        np.array([
            *(phi_context.constraints[i].get_lin_combination_copy() for i in gamma_eq_constraint_indices),
            *(constraint.get_lin_combination_copy() for constraint in gamma_additional_constraints)
        ], dtype=Fraction),
        context.variable_count()
    )

    if debug_mode:
        _logger.debug("Gamma equality constraint matrix:\n%s", gamma_eq_constraint_mat)

    # compute the kernel of the equality constraints matrix

    gamma_eq_constraint_mat_ker = compute_kernel(gamma_eq_constraint_mat)

    if debug_mode:
        _logger.debug("Gamma equality constraint matrix kernel:\n%s", gamma_eq_constraint_mat_ker)

    # we now translate the equality constraints into Pi-respecting formulas
    gamma_eq_constraint_mat_ker_proj = tuple(
        context.project_matrix_onto_block(gamma_eq_constraint_mat_ker, b)
        for b in (VarDecContext.X, VarDecContext.Y)
    )

    # analyze the matrix with the goal of determining whether the predicate set is Pi-simple or not
    for b in VarDecContext.X, VarDecContext.Y:
        if not np.any(gamma_eq_constraint_mat_ker_proj[b]):
            # the projected matrix is zero, that is, block is fixed
            _logger.info("Gamma is Pi-simple (%s is fixed), hence covering is trivial.", block_str(b))

            sigma = [
                (var, _rational_to_z3_ratval(gamma_model_vec_proj[b][i]))
                for i, var in enumerate(context.block_variables_iter(b))
            ]

            decomposition = z3.And(
                *(z3.substitute(p, *sigma) for p in gamma),
                *(var == val for var, val in sigma)
            )

            if debug_mode:
                assert is_valid(decomposition == z3.And(*gamma))
            return decomposition

    gamma_eq_constraint_mat_lindep = tuple(
        compute_kernel(np.transpose(gamma_eq_constraint_mat_ker_proj[b]))
        for b in (VarDecContext.X, VarDecContext.Y)
    )

    if debug_mode:
        for b in VarDecContext.X, VarDecContext.Y:
            _logger.debug(
                "Linear dependencies of Gamma^= with respect to %s:\n%s",
                block_str(b),
                gamma_eq_constraint_mat_lindep[b]
            )

    for b in VarDecContext.X, VarDecContext.Y:
        # iterate over the columns of the matrix
        for lin_dependency_witness in np.transpose(gamma_eq_constraint_mat_lindep[b]):
            theta.append(
                context.block_linear_comb_to_expr(lin_dependency_witness, b)
                == np.dot(gamma_model_vec_proj[b], lin_dependency_witness))

    _logger.info("Theta (with Gamma's linear dependencies enforced):\n%s", theta)

    if debug_mode:
        assert is_valid(z3.Implies(z3.And(*gamma), z3.And(*theta)))

    if use_heuristics:
        # now we try to cover Gamma in a very agressive manner, for which model flooding doesn't hold
        # this is an optional heuristic designed to improve the performance

        if phi_context.query_whether_formula_entails_phi(z3.And(*theta)):
            _logger.info("Heuristic success: Theta entails phi, so we can cover entire Theta")
            # try to expand even further
            theta_expanded = []
            while len(theta) > 0:
                p = theta.pop()
                cur_expanded_covering = z3.And(*theta_expanded, *theta)
                if phi_context.query_whether_formula_entails_phi(cur_expanded_covering):
                    # predicate p can be removed from theta
                    continue
                # the expanding formula doesn't entail phi
                # hence, it is crucial to keep p
                theta_expanded.append(p)

            _logger.info("Theta expanded: %s", theta_expanded)
            return z3.And(*theta_expanded)

        _logger.info("Heuristic fail: Theta does not entail phi, computing the sound and complete covering...")
        # the heuristic failed, so we compute the sound and complete covering

    delta = []
    upsilon_lt_gt = []
    upsilon_neq = []

    disjunct_solver = z3.Solver()
    disjunct_solver.add(z3.And(*theta))

    while disjunct_solver.check() == z3.sat:
        omega_model = disjunct_solver.model()
        omega_model_vec = context.model_to_vec(omega_model)
        omega_model_vec_proj = context.project_vector_onto_block(omega_model_vec, VarDecContext.X), \
            context.project_vector_onto_block(omega_model_vec, VarDecContext.Y)

        _logger.info("Found new disjunct Omega corresponding to model %s", omega_model_vec)

        # determine the equality constraints the disjunct containing the model satisfies

        omega_eq_constraint_indices = []
        not_omega_eq_constraint_indices = []

        for i, constraint in enumerate(phi_context.constraints):
            if constraint.model_satisfies_equality_version(omega_model_vec):
                omega_eq_constraint_indices.append(i)
            else:
                not_omega_eq_constraint_indices.append(i)

        if debug_mode:
            _logger.debug(
                "Omega equality constraint id's: %s Id's of absent constraints: %s",
                omega_eq_constraint_indices,
                not_omega_eq_constraint_indices
            )

        # translate the computed indices of constraints into actual constraints
        omega_eq_constraints = [
            phi_context.constraints[i].get_equality_expr(context) for i in omega_eq_constraint_indices
        ]

        _logger.debug("Omega equality constraints: %s", omega_eq_constraints)

        # compute the equality constraints matrix

        omega_eq_constraint_mat = _matrix_add_zero_row_if_empty(
            np.array([
                phi_context.constraints[i].get_lin_combination_copy() for i in omega_eq_constraint_indices
            ], dtype=Fraction),
            context.variable_count()
        )

        omega_eq_constraint_mat_ker = compute_kernel(omega_eq_constraint_mat)

        omega_eq_constraint_mat_ker_proj = \
            context.project_matrix_onto_block(omega_eq_constraint_mat_ker, VarDecContext.X),\
            context.project_matrix_onto_block(omega_eq_constraint_mat_ker, VarDecContext.Y),

        omega_eq_constraint_mat_lindep = tuple(
            compute_kernel(np.transpose(omega_eq_constraint_mat_ker_proj[b]))
            for b in (VarDecContext.X, VarDecContext.Y)
        )

        if debug_mode:
            for b in VarDecContext.X, VarDecContext.Y:
                _logger.debug(
                    "Linear dependencies of Omega^= with respect to %s:\n%s",
                    block_str(b),
                    omega_eq_constraint_mat_lindep[b]
                )

        lindep_diff = tuple(
            compute_gen_set_of_intersection_of_mat_images(
                omega_eq_constraint_mat_lindep[b], gamma_eq_constraint_mat_ker_proj[b]
            )
            for b in (VarDecContext.X, VarDecContext.Y)
        )

        if debug_mode:
            for b in VarDecContext.X, VarDecContext.Y:
                _logger.debug(
                    "Linear dependencies present in Omega but absent in Gamma, with respect to %s:\n%s",
                    block_str(b),
                    lindep_diff[b]
                )

        w_pred_constraint = None

        for b in VarDecContext.X, VarDecContext.Y:
            for w in np.transpose(lindep_diff[b]):

                if not np.any(w):
                    # this column is a zero-column
                    continue

                _logger.info(
                    "Found witness that Omega has more linear dependencies compared to Gamma with respect to %s: %s",
                    block_str(b),
                    w
                )

                w_pred_constraint = LinearConstraint(
                    context.project_vector_back_from_block(w, b),
                    np.dot(omega_model_vec_proj[b], w)
                )

                break

            if w_pred_constraint is not None:
                break

        if w_pred_constraint is not None:
            w_lhs = w_pred_constraint.get_lhs_linear_combination_expr(context)
            w_rhs = w_pred_constraint.get_rhs_constrant()

            _logger.info("Witness predicate: %s", w_lhs == w_rhs)

            if debug_mode:
                assert is_valid(z3.Implies(z3.And(*omega_eq_constraints), w_lhs == w_rhs))

            w_predicate_lt = w_lhs < w_rhs
            w_predicate_gt = w_lhs > w_rhs

            if is_valid(z3.Implies(z3.And(*gamma), w_predicate_lt)):
                _logger.info("Excellent! Gamma entails the < version of the witness predicate.")
                disjunct_solver.add(w_predicate_lt)
                upsilon_lt_gt.append(w_predicate_lt)
            elif is_valid(z3.Implies(z3.And(*gamma), w_predicate_gt)):
                _logger.info("Excellent! Gamma entails the > version of the witness predicate.")
                disjunct_solver.add(w_predicate_gt)
                upsilon_lt_gt.append(w_predicate_gt)
            else:
                _logger.info("Gamma unfortunately entails neither the < nor the > version of the witness predicate.")
                disjunct_solver.add(w_lhs != w_rhs)
                upsilon_neq.append((w_lhs, w_rhs))

            rec_solver = z3.Solver()
            rec_solver.add(z3.And(*gamma))
            rec_solver.add(w_lhs == w_rhs)

            if rec_solver.check() == z3.sat:
                rec_model = rec_solver.model()
                _logger.debug(
                    "Gamma together with the witness predicate is satisfiable, so we cover recursively.\nModel: %s",
                    context.model_to_vec(rec_model)
                )

                # make sure that Gamma doesn't lose any models
                rec_cover = cover(
                    context,
                    phi_context,
                    rec_model,
                    gamma_additional_constraints + [w_pred_constraint],
                    debug_mode=debug_mode,
                    use_heuristics=use_heuristics
                )

                delta.append(rec_cover)

            else:
                _logger.info("Gamma together with the witness predicate is unsatisfiable, no recursion happens.")

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
            elif debug_mode:
                _logger.debug("Ignoring disjunct corresponding to %s", subset_set)

    decomposition = z3.Or(
        z3.And(*theta, *upsilon_lt_gt, z3.Or(*decomposition_disjuncts)),
        *delta
    )

    _logger.info("Covering call returns decomposition:\n%s", decomposition)

    return decomposition


def vardec(phi, x: list, y: list, debug_mode=True, use_heuristics=True):

    context = VarDecContext(x, y)
    phi_context = FormulaContext(phi, context)

    phi_dec = []

    global_solver = z3.Solver()
    global_solver.add(phi)

    while global_solver.check() == z3.sat:
        gamma_model = global_solver.model()
        psi = cover(context, phi_context, gamma_model, debug_mode=debug_mode, use_heuristics=use_heuristics)

        _logger.info("Covering algorithm produced psi:\n%s", psi)

        if debug_mode:
            gamma = z3.And(*(
                constraint.get_version_satisfying_model(
                    context,
                    context.model_to_vec(gamma_model)
                )
                for constraint in phi_context.constraints
            ))

            assert is_valid(z3.Implies(gamma, psi))

        if not phi_context.query_whether_formula_entails_phi(psi):
            # psi doesn't entail phi, so we conclude thatphi is not Pi-decomposable
            return None

        phi_dec.append(psi)
        global_solver.add(z3.Not(psi))

    return z3.Or(*phi_dec)
