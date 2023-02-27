import itertools
import logging
from fractions import Fraction
from typing import List

import numpy as np
# noinspection PyPackageRequirements
import z3

from visualizer import DummyCoverVisualizer, CoverVisualizer, Visualizer, ActualCoverVisualizer
from vardec_context import VarDecContext, block_str
from formula_context import FormulaContext
from gauss import compute_kernel, compute_gen_set_of_intersection_of_mat_images
from linear_constraint import LinearConstraint
from z3_utils import is_sat, is_valid, replace_strict_inequality_by_nonstrict

_logger = logging.getLogger("vardec")


def _rational_to_z3_ratval(frac: Fraction, /) -> z3.RatVal:
    return z3.RatVal(frac.numerator, frac.denominator)


def _matrix_add_zero_row_if_empty(mat: np.ndarray, mat_cols: int, /):
    if mat.shape[0] == 0:
        return np.zeros((1, mat_cols), dtype=Fraction)
    return mat


def _cover(
    context: VarDecContext,
    phi_context: FormulaContext,
    gamma_model,
    gamma_additional_eq_constraints: List[LinearConstraint] = None, /, *,
    visualizer: CoverVisualizer = DummyCoverVisualizer()
):

    _logger.info("=== [Covering algorithm] ===")
    context.stat_on_cover_call()

    if gamma_additional_eq_constraints is None:
        gamma_additional_eq_constraints = []

    if context.debug_mode:
        _logger.debug(
            "Additional constraints: %s",
            [constraint.get_equality_expr(context) for constraint in gamma_additional_eq_constraints]
        )

    gamma_model_vec = context.model_to_vec(gamma_model)
    gamma_model_vec_proj = tuple(
        context.project_vector_onto_block(gamma_model_vec, b)
        for b in (VarDecContext.X, VarDecContext.Y)
    )

    if context.debug_mode:
        _logger.debug("Model: %s", gamma_model_vec)

    _logger.info("Gamma tag: %s", phi_context.model_vec_to_tag(gamma_model_vec))

    gamma = [
        *(ct.get_version_satisfying_model(context, gamma_model_vec) for ct in phi_context.constraints),
        *(ct.get_equality_expr(context) for ct in gamma_additional_eq_constraints)
    ]

    if context.debug_mode:
        for ct in gamma_additional_eq_constraints:
            assert ct.respects_pi(context)

    # choose equality predicates respecting the partition
    theta_equality_linear_constraints = [
        *(ct for ct in phi_context.constraints if
            ct.model_satisfies_equality_version(gamma_model_vec) and ct.respects_pi(context)),
        *gamma_additional_eq_constraints
    ]

    # choose predicates respecting the partition
    theta = [p for p in gamma if context.predicate_respects_pi(p)]

    _logger.info("Initialized Theta to be the set of Pi-respecting predicates in Gamma, that is:\nTheta = %s", theta)

    # ids' of equality constraints in Gamma, which originate in phi
    gamma_eq_constraint_from_phi_indices = [
        i for i, constraint in enumerate(phi_context.constraints)
        if constraint.model_satisfies_equality_version(gamma_model_vec)
    ]

    if context.debug_mode:
        assert is_sat(z3.And(*gamma))
        assert is_valid(z3.Implies(
            context.vector_to_enforcing_expr(gamma_model_vec),
            z3.And(*gamma)
        ))
        _logger.debug("Gamma: %s", gamma)
        _logger.debug(
            "Id's of equality constraints in Gamma which originate in phi: %s",
            gamma_eq_constraint_from_phi_indices
        )

    # compute the equality constraints matrix
    gamma_eq_constraint_mat = _matrix_add_zero_row_if_empty(
        np.array([
            *(phi_context.constraints[i].get_lhs_linear_combination_vector()
              for i in gamma_eq_constraint_from_phi_indices),
            *(constraint.get_lhs_linear_combination_vector() for constraint in gamma_additional_eq_constraints)
        ], dtype=Fraction),
        context.variable_count()
    )

    _logger.debug("Gamma equality constraint matrix:\n%s", gamma_eq_constraint_mat)

    # compute the kernel of the equality constraints matrix

    gamma_eq_constraint_mat_ker = compute_kernel(gamma_eq_constraint_mat)

    if context.debug_mode:
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

            visualizer.on_cover_init_and_ret_pi_simple()

            if context.debug_mode:
                assert is_valid(decomposition == z3.And(*gamma))
            return decomposition

    gamma_eq_constraint_mat_lindep = tuple(
        compute_kernel(np.transpose(gamma_eq_constraint_mat_ker_proj[b]))
        for b in (VarDecContext.X, VarDecContext.Y)
    )

    if context.debug_mode:
        for b in VarDecContext.X, VarDecContext.Y:
            _logger.debug(
                "Linear dependencies of Gamma^= with respect to %s:\n%s",
                block_str(b),
                gamma_eq_constraint_mat_lindep[b]
            )

    for b in VarDecContext.X, VarDecContext.Y:
        # iterate over the columns of the matrix
        for lin_dependency_witness in np.transpose(gamma_eq_constraint_mat_lindep[b]):

            # noinspection PyTypeChecker
            lin_dependency_constraint = LinearConstraint(
                context.project_vector_back_from_block(lin_dependency_witness, b),
                np.dot(gamma_model_vec_proj[b], lin_dependency_witness)
            )

            theta_equality_linear_constraints.append(lin_dependency_constraint)

            theta.append(lin_dependency_constraint.get_equality_expr(context))

    _logger.info("Theta (with Gamma's linear dependencies enforced):\n%s", theta)

    if context.debug_mode:
        assert is_valid(z3.Implies(z3.And(*gamma), z3.And(*theta)))
        assert is_valid(z3.Implies(
            z3.And(*gamma),
            z3.And(*(ct.get_equality_expr(context) for ct in theta_equality_linear_constraints))
        ))

    visualizer.on_cover_init_pi_complex(gamma_additional_eq_constraints, z3.And(*theta))

    if context.use_heuristics:
        # now we try to cover Gamma in a very agressive manner, for which model flooding doesn't hold
        # this is an optional heuristic designed to improve the performance

        if phi_context.query_whether_formula_entails_phi(z3.And(*theta)):
            _logger.info("Heuristic success: Theta entails phi, so we can cover entire Theta")
            context.stat_on_heuristic_success()
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

            if context.debug_mode:
                assert phi_context.query_whether_formula_entails_phi(z3.And(*theta_expanded))

            # another heuristic: try replacing strict inequalities by non-strict ones
            theta_expanded_nonstrict = z3.And(*(replace_strict_inequality_by_nonstrict(p) for p in theta_expanded))

            if phi_context.query_whether_formula_entails_phi(theta_expanded_nonstrict):
                return theta_expanded_nonstrict

            return z3.And(*theta_expanded)

        _logger.info("Heuristic fail: Theta does not entail phi, computing the sound and complete covering...")
        context.stat_on_heuristic_fail()
        # the heuristic failed, so we compute the sound and complete covering

    delta = []
    upsilon_lt_gt = []
    upsilon_neq = []

    restrict_to_not_phi = z3.Bool("_rtnp")

    disjunct_solver = z3.Solver()
    disjunct_solver.add(z3.And(*theta))

    # idea of the blast heuristic: try to find first those disjuncts, where phi is false
    # once such false disjunct is found, compute a witnessing predicate separating it from Gamma
    # and then check whether Delta OR (Theta AND [separating predicate]) entails phi
    # if yes, then this is the covering

    disjunct_solver.add(z3.Implies(restrict_to_not_phi, z3.Not(phi_context.phi)))

    if not context.use_blast_heuristic:
        disjunct_solver.add(z3.Not(restrict_to_not_phi))

    # whether we have already considered all those Omega, which agree with (not phi) on at least one model
    false_disjuncts_exhausted = False

    _logger.info("Starting to enumerate disjuncts of Theta.")

    while True:

        false_disjuncts_exhausted_in_this_iter = False

        if false_disjuncts_exhausted or not context.use_blast_heuristic:
            if disjunct_solver.check() == z3.unsat:
                break
        else:
            # we are using the neg phi cover heuristic
            # try to first find a disjunct where the formula is false

            if disjunct_solver.check(restrict_to_not_phi) == z3.sat:
                # found disjunct agreein with (not phi) on some model
                pass
            elif disjunct_solver.check() == z3.sat:
                false_disjuncts_exhausted = True
                false_disjuncts_exhausted_in_this_iter = True
                _logger.info("Blast heuristic: false disjuncts exhausted.")
            else:
                break

        omega_model = disjunct_solver.model()
        omega_model_vec = context.model_to_vec(omega_model)
        omega_model_vec_proj = tuple(
            context.project_vector_onto_block(omega_model_vec, b)
            for b in (VarDecContext.X, VarDecContext.Y)
        )

        _logger.info("Found new disjunct Omega corresponding to model %s", omega_model_vec)
        _logger.info("Omega tag (in the original formula): %s", phi_context.model_vec_to_tag(omega_model_vec))

        # determine the equality constraints the disjunct containing the model satisfies
        omega_eq_constraint_indices = []
        # id's of constraints, such that their <, or > version is true under the model, but not the = version
        not_omega_eq_constraint_indices = []

        for i, constraint in enumerate(phi_context.constraints):
            if constraint.model_satisfies_equality_version(omega_model_vec):
                omega_eq_constraint_indices.append(i)
            else:
                not_omega_eq_constraint_indices.append(i)

        if context.debug_mode:
            _logger.debug(
                "Omega equality constraint id's: %s Id's of absent constraints: %s",
                omega_eq_constraint_indices,
                not_omega_eq_constraint_indices
            )

        # compute the omega equality constraints matrix
        omega_eq_constraint_mat = _matrix_add_zero_row_if_empty(
            np.array([
                *(phi_context.constraints[i].get_lhs_linear_combination_vector() for i in omega_eq_constraint_indices),
                # every omega contains all the equality predicates from theta
                *(constraint.get_lhs_linear_combination_vector() for constraint in theta_equality_linear_constraints)
            ], dtype=Fraction),
            context.variable_count()
        )

        _logger.debug("Omega equality constraints matrix:\n%s", omega_eq_constraint_mat)

        omega_eq_constraint_mat_ker = compute_kernel(omega_eq_constraint_mat)

        omega_eq_constraint_mat_ker_proj = tuple(
            context.project_matrix_onto_block(omega_eq_constraint_mat_ker, b)
            for b in (VarDecContext.X, VarDecContext.Y)
        )

        omega_eq_constraint_mat_lindep = tuple(
            compute_kernel(np.transpose(omega_eq_constraint_mat_ker_proj[b]))
            for b in (VarDecContext.X, VarDecContext.Y)
        )

        if context.debug_mode:
            for b in VarDecContext.X, VarDecContext.Y:
                _logger.debug(
                    "Linear dependencies of Omega^= with respect to %s:\n%s",
                    block_str(b),
                    omega_eq_constraint_mat_lindep[b]
                )

        lindep_diff = tuple(
            compute_gen_set_of_intersection_of_mat_images(
                omega_eq_constraint_mat_lindep[b], gamma_eq_constraint_mat_ker_proj[b],
                debug_mode=context.debug_mode
            )
            for b in (VarDecContext.X, VarDecContext.Y)
        )

        if context.debug_mode:
            for b in VarDecContext.X, VarDecContext.Y:
                _logger.debug(
                    "Linear dependencies present in Omega but absent in Gamma, with respect to %s:\n%s",
                    block_str(b),
                    lindep_diff[b]
                )

        w_pred_constraint = None

        # this variable will be set to True iff we will find a predicate in the loop below,
        # such that Gamma entails either the < or the > version of it
        nice_predicate_found = False

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

                # noinspection PyTypeChecker
                w_pred_constraint = LinearConstraint(
                    context.project_vector_back_from_block(w, b),
                    np.dot(omega_model_vec_proj[b], w)
                )

                w_lhs = w_pred_constraint.get_lhs_linear_combination_expr(context)
                w_rhs = w_pred_constraint.get_rhs_constrant()

                if is_valid(z3.Implies(z3.And(*gamma), w_lhs < w_rhs)):
                    _logger.info("Excellent! Gamma entails the < version of the witness predicate.")
                    disjunct_solver.add(w_lhs < w_rhs)
                    upsilon_lt_gt.append(w_lhs < w_rhs)
                    nice_predicate_found = True
                    break
                elif is_valid(z3.Implies(z3.And(*gamma), w_lhs > w_rhs)):
                    _logger.info("Excellent! Gamma entails the > version of the witness predicate.")
                    disjunct_solver.add(w_lhs > w_rhs)
                    upsilon_lt_gt.append(w_lhs > w_rhs)
                    nice_predicate_found = True
                    break

            if nice_predicate_found:
                break

        if w_pred_constraint is not None:

            context.stat_on_distinguishable_disjunct()

            w_lhs = w_pred_constraint.get_lhs_linear_combination_expr(context)
            w_rhs = w_pred_constraint.get_rhs_constrant()

            _logger.info("Witness predicate: %s", w_lhs == w_rhs)

            if context.debug_mode:
                _omega_eq = z3.And(
                    # equality predicates of Omega, from the original formula phi
                    *(phi_context.constraints[i].get_equality_expr(context) for i in omega_eq_constraint_indices),
                    # equality predicates of Omega which originate from Theta
                    *(constraint.get_equality_expr(context) for constraint in theta_equality_linear_constraints)
                )
                assert is_sat(_omega_eq)
                assert is_valid(z3.Implies(_omega_eq, w_lhs == w_rhs))

            if not nice_predicate_found:
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
                rec_cover = _cover(
                    context,
                    phi_context,
                    rec_model,
                    gamma_additional_eq_constraints + [w_pred_constraint],
                    visualizer=visualizer.create_visualizer_for_recursive_call()
                )

                delta.append(rec_cover)

            else:
                _logger.info("Gamma together with the witness predicate is unsatisfiable, no recursion happens.")

        else:
            context.stat_on_indistinguishable_disjunct()

            _logger.info(
                "The current disjunct Omega cannot be distinguished from Gamma in the language of Pi-decompositions"
            )

            disjunct_solver.add(z3.Or(
                # new disjuncts must either disagree with Omega on some of its equality predicates
                *(z3.Not(phi_context.constraints[i].get_equality_expr(context)) for i in omega_eq_constraint_indices),
                # or they must contain strictly more equality predicates compared to Omega
                *(phi_context.constraints[i].get_equality_expr(context) for i in not_omega_eq_constraint_indices)
            ))

        if context.use_blast_heuristic and \
                (false_disjuncts_exhausted_in_this_iter or not false_disjuncts_exhausted):

            heuristic_covering = z3.Or(
                *delta,
                z3.And(
                    *theta,
                    *upsilon_lt_gt,
                    *(lhs != rhs for lhs, rhs in upsilon_neq)
                )
            )

            if phi_context.query_whether_formula_entails_phi(heuristic_covering):
                _logger.info("Blast heuristic success!")
                context.stat_on_blast_heuristic_success()
                return heuristic_covering
            else:
                _logger.info("Blast heuristic fail!")
                context.stat_on_blast_heuristic_fail()

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
            elif context.debug_mode:
                _logger.debug("Ignoring disjunct corresponding to %s", subset_set)

    decomposition = z3.Or(
        *delta,
        z3.And(*theta, *upsilon_lt_gt, z3.Or(*decomposition_disjuncts))
    )

    _logger.info("Covering call returns decomposition:\n%s", decomposition)

    return decomposition


def vardec_binary(phi, context: VarDecContext, /, *, visualizer: Visualizer = None):

    phi_context = FormulaContext(phi, context)

    if visualizer is not None:
        visualizer.set_contexts(context, phi_context)

    cover_visualizer = DummyCoverVisualizer()

    phi_dec = []

    global_solver = z3.Solver()
    global_solver.add(phi)

    while global_solver.check() == z3.sat:
        gamma_model = global_solver.model()

        if visualizer is not None:
            cover_visualizer = visualizer.get_cover_visualizer_for_next_gamma(gamma_model)
            assert isinstance(cover_visualizer, ActualCoverVisualizer)

        psi = _cover(
            context,
            phi_context,
            gamma_model,
            visualizer=cover_visualizer
        )

        _logger.info("Covering algorithm produced psi:\n%s", psi)

        if context.debug_mode:
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
