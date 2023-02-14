import itertools
from fractions import Fraction
from typing import List

import numpy as np
import z3

from formula_context import FormulaContext
from linconstraint import LinearConstraint
from gauss import compute_kernel, check_image_space_inclusion
from vardec_context import VarDecContext
from z3_utils import is_sat


class _DisjunctLinearDependencies:

    def __init__(self, ker_proj: tuple, /):
        self._ker_proj = ker_proj

    def _projection_is_subspace_of(self, other, wrt_block: int, /) -> bool:
        assert isinstance(other, _DisjunctLinearDependencies)
        assert wrt_block == VarDecContext.X or wrt_block == VarDecContext.Y
        return check_image_space_inclusion(
            self._ker_proj[wrt_block],
            other._ker_proj[wrt_block]
        )

    def _projection_space_is_equal(self, other, wrt_block: int, /) -> bool:
        assert isinstance(other, _DisjunctLinearDependencies)
        assert wrt_block == VarDecContext.X or wrt_block == VarDecContext.Y
        return self._projection_is_subspace_of(other, wrt_block) and other._projection_is_subspace_of(self, wrt_block)

    def is_equivalent_to(self, other) -> bool:
        assert isinstance(other, _DisjunctLinearDependencies)
        return self._projection_space_is_equal(other, VarDecContext.X) and \
            self._projection_space_is_equal(other, VarDecContext.Y)


class _DisjunctGraphBuilder:

    def __init__(self, context: VarDecContext, all_constraints: List[LinearConstraint], /):
        self._context = context
        self._all_constraints = all_constraints

        # keys: frozensets of indices to all_constraints array
        # values: _DisjunctLinearDependencies instances
        self._lindep_table = {}

        # array of arrays of frozensets
        self._disjunct_groups = []

        # keys: frozensets of indices to all_constraints array
        # value: id of the group of disjuncts the disjunct belongs to
        self._group_table = {}

    def model_vec_to_disjunct(self, model_vec: np.ndarray) -> list:
        return [
            ct.get_version_satisfying_model(self._context, model_vec) for ct in self._all_constraints
        ]

    def model_vec_to_equality_constraint_ids(self, model_vec: np.ndarray) -> frozenset:
        return frozenset(ct_id for ct_id, _ in itertools.filterfalse(
            lambda el: not el[1].model_satisfies_equality_version(model_vec),
            enumerate(self._all_constraints)
        ))

    def _get_disjunct_linear_dependencies(self, disjunct_eq_constraints: frozenset) -> _DisjunctLinearDependencies:

        if disjunct_eq_constraints in self._lindep_table:
            return self._lindep_table[disjunct_eq_constraints]

        if len(disjunct_eq_constraints) > 0:
            constraint_matrix = np.array([
                self._all_constraints[i].get_lhs_linear_combination_vector() for i in disjunct_eq_constraints
            ], dtype=Fraction)
        else:
            constraint_matrix = np.zeros((1, self._context.variable_count()), dtype=Fraction)

        constraint_matrix_ker = compute_kernel(constraint_matrix)

        constraint_matrix_ker_proj = tuple(
            self._context.project_matrix_onto_block(constraint_matrix_ker, b)
            for b in (VarDecContext.X, VarDecContext.Y)
        )

        disj_lindep = _DisjunctLinearDependencies(constraint_matrix_ker_proj)

        self._lindep_table[disjunct_eq_constraints] = disj_lindep

        return disj_lindep

    def register_disjunct(self, disjunct_eq_constraints: frozenset):

        if disjunct_eq_constraints in self._group_table:
            # current disjunct is already registered, nothing to do
            return

        disjunct_lindep = self._get_disjunct_linear_dependencies(disjunct_eq_constraints)

        # determine in what group the disjunct belongs to
        for group_id, group in enumerate(self._disjunct_groups):
            # group representative
            group_repr = group[0]
            group_repr_lindep = self._lindep_table[group_repr]

            if disjunct_lindep.is_equivalent_to(group_repr_lindep):
                # the current disjunct falls into this group
                group.append(disjunct_eq_constraints)
                self._group_table[disjunct_eq_constraints] = group_id
                return

        # the current disjunct doesn't belong to any existing group
        # hence, we need to create a new one
        new_group_id = len(self._disjunct_groups)
        self._disjunct_groups.append([disjunct_eq_constraints])
        self._group_table[disjunct_eq_constraints] = new_group_id

    def debug_print(self):
        print("Disjunct groups:", self._disjunct_groups)


def compute_all_disjuncts(
    context: VarDecContext,
    phi_context: FormulaContext,
    domain_lin_constraints: List[LinearConstraint],
    domain_formula
):

    disjunct_solver = z3.Solver()
    disjunct_solver.add(domain_formula)

    disj_graph_builder = _DisjunctGraphBuilder(context, phi_context.constraints + domain_lin_constraints)

    disj_count = 0

    while disjunct_solver.check() == z3.sat:
        disj_model_vec = context.model_to_vec(disjunct_solver.model())
        disjunct = disj_graph_builder.model_vec_to_disjunct(disj_model_vec)
        assert is_sat(z3.And(*disjunct))

        print("Disjunct:", disjunct)

        eq_constraint_ids = disj_graph_builder.model_vec_to_equality_constraint_ids(disj_model_vec)

        disj_graph_builder.register_disjunct(eq_constraint_ids)

        print(eq_constraint_ids)

        disj_count += 1

        disjunct_solver.add(z3.Or(
            *(z3.Not(d) for d in disjunct)
        ))

    print("Disjunct count: %d" % disj_count)
    disj_graph_builder.debug_print()
