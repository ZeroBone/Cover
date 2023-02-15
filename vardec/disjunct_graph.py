import itertools
import math
from fractions import Fraction
from typing import List

import graphviz
import networkx as nx
import numpy as np
import z3

from formula_context import FormulaContext
from linconstraint import LinearConstraint
from gauss import compute_kernel, check_image_space_inclusion
from vardec_context import VarDecContext
from z3_utils import is_sat, is_valid


class _DisjunctKernelProjectedSolutionSpaces:

    def __init__(self, ker_proj: tuple, /):
        self._ker_proj = ker_proj

    def projection_is_subspace_of(self, other, wrt_block: int, /) -> bool:
        assert isinstance(other, _DisjunctKernelProjectedSolutionSpaces)
        assert wrt_block == VarDecContext.X or wrt_block == VarDecContext.Y
        return check_image_space_inclusion(
            self._ker_proj[wrt_block],
            other._ker_proj[wrt_block]
        )

    def _projection_space_is_equal(self, other, wrt_block: int, /) -> bool:
        assert isinstance(other, _DisjunctKernelProjectedSolutionSpaces)
        assert wrt_block == VarDecContext.X or wrt_block == VarDecContext.Y
        return self.projection_is_subspace_of(other, wrt_block) and other.projection_is_subspace_of(self, wrt_block)

    def is_equivalent_to(self, other) -> bool:
        assert isinstance(other, _DisjunctKernelProjectedSolutionSpaces)
        return self._projection_space_is_equal(other, VarDecContext.X) and \
            self._projection_space_is_equal(other, VarDecContext.Y)


class _DisjunctGroup:

    def __init__(self, group_id: int, repr_eq_ct_ids: frozenset):
        self.group_id = group_id
        self.eq_ct = [repr_eq_ct_ids]
        self.disjunct_labels = []

    def get_repr_eq_ct_ids(self) -> frozenset:
        return self.eq_ct[0]

    def add_disjunct_label(self, label: str, color: str, /):
        self.disjunct_labels.append((label, color))

    def get_graphviz_group_node_label(self) -> str:

        self.disjunct_labels = sorted(self.disjunct_labels)

        def _escape_char(s: str) -> str:
            return s.replace("<", "&#60;").replace(">", "&#62;")

        def _compute_disjunct_html(label: str, color: str) -> str:
            return "<font color='%s'>%s</font>" % (color, _escape_char(label))

        if len(self.disjunct_labels) <= 5:
            return "<%s>" % "|".join(
                _compute_disjunct_html(label, color) for label, color in self.disjunct_labels
            )

        columns = int(math.ceil(.6 * math.sqrt(len(self.disjunct_labels))))

        cur_label_index = 0

        rows = []

        while cur_label_index < len(self.disjunct_labels):
            rows.append("{%s}" % "|".join(
                _compute_disjunct_html(label, color)
                for label, color in self.disjunct_labels[cur_label_index:cur_label_index + columns]
            ))
            cur_label_index += columns

        return "<%s>" % "|".join(rows)


class _DisjunctGraphBuilder:

    def __init__(self, context: VarDecContext, all_constraints: List[LinearConstraint], /):
        self._context = context
        self._all_constraints = all_constraints

        # keys: frozensets of indices to all_constraints array
        # values: _DisjunctKernelProjectedSolutionSpaces instances
        self._lindep_table = {}

        # array of _DisjunctGroup instances
        self._disjunct_groups: List[_DisjunctGroup] = []

        # keys: frozensets of indices to all_constraints array
        # value: id of the group of disjuncts the disjunct belongs to
        self._group_table = {}

    def model_vec_to_disjunct(self, model_vec: np.ndarray, /) -> list:
        return [
            ct.get_version_satisfying_model(self._context, model_vec) for ct in self._all_constraints
        ]

    def _model_vec_to_equality_constraint_ids(self, model_vec: np.ndarray, /) -> frozenset:
        return frozenset(ct_id for ct_id, _ in itertools.filterfalse(
            lambda el: not el[1].model_satisfies_equality_version(model_vec),
            enumerate(self._all_constraints)
        ))

    def _model_vec_to_label(self, model_vec: np.ndarray) -> str:
        return "".join(
            ct.get_predicate_symbol_satisfying_model(model_vec)
            for ct in self._all_constraints
        )

    def _get_disjunct_linear_dependencies(self, disjunct_eq_constraints: frozenset, /)\
            -> _DisjunctKernelProjectedSolutionSpaces:

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

        disj_lindep = _DisjunctKernelProjectedSolutionSpaces(constraint_matrix_ker_proj)

        self._lindep_table[disjunct_eq_constraints] = disj_lindep

        return disj_lindep

    def _register_disjunct_eq_constraints(self, eq_constraint_ids: frozenset, /):

        if eq_constraint_ids in self._group_table:
            # current combination of equality constraints is already known, nothing to do
            return

        disjunct_lindep = self._get_disjunct_linear_dependencies(eq_constraint_ids)

        # determine in what group the disjunct belongs to
        for group_id, group in enumerate(self._disjunct_groups):
            # group representative
            group_repr = group.get_repr_eq_ct_ids()
            group_repr_lindep = self._lindep_table[group_repr]

            if disjunct_lindep.is_equivalent_to(group_repr_lindep):
                # the current disjunct falls into this group
                self._group_table[eq_constraint_ids] = group_id
                return

        # the current disjunct doesn't belong to any existing group
        # hence, we need to create a new one
        new_group_id = len(self._disjunct_groups)
        self._disjunct_groups.append(_DisjunctGroup(new_group_id, eq_constraint_ids))
        self._group_table[eq_constraint_ids] = new_group_id

    def add_disjunct(self, model_vec: np.ndarray, phi_context: FormulaContext, /):

        eq_constraint_ids = self._model_vec_to_equality_constraint_ids(model_vec)

        self._register_disjunct_eq_constraints(eq_constraint_ids)

        assert eq_constraint_ids in self._lindep_table
        assert eq_constraint_ids in self._group_table

        group = self._disjunct_groups[self._group_table[eq_constraint_ids]]

        disjunct_entails_phi = phi_context.model_check(model_vec, self._context)

        if disjunct_entails_phi:
            assert is_valid(z3.Implies(z3.And(*self.model_vec_to_disjunct(model_vec)), phi_context.phi))
        else:
            assert is_valid(z3.Implies(z3.And(*self.model_vec_to_disjunct(model_vec)), z3.Not(phi_context.phi)))

        group.add_disjunct_label(
            self._model_vec_to_label(model_vec),
            "green3" if disjunct_entails_phi else "red3"
        )

    def create_group_graph(self) -> graphviz.Digraph:

        x_edges = []
        y_edges = []

        for first_group in self._disjunct_groups:
            first_group_id = first_group.group_id

            first_group_repr = first_group.get_repr_eq_ct_ids()
            first_group_repr_ker = self._lindep_table[first_group_repr]

            for second_group_id in range(first_group.group_id + 1, len(self._disjunct_groups)):
                second_group = self._disjunct_groups[second_group_id]

                second_group_repr = second_group.get_repr_eq_ct_ids()
                second_group_repr_ker = self._lindep_table[second_group_repr]

                assert not first_group_repr_ker.is_equivalent_to(second_group_repr_ker)

                first_subspace_second_x = first_group_repr_ker.projection_is_subspace_of(
                    second_group_repr_ker,
                    VarDecContext.X
                )
                first_subspace_second_y = first_group_repr_ker.projection_is_subspace_of(
                    second_group_repr_ker,
                    VarDecContext.Y
                )

                second_subspace_first_x = second_group_repr_ker.projection_is_subspace_of(
                    first_group_repr_ker,
                    VarDecContext.X
                )
                second_subspace_first_y = second_group_repr_ker.projection_is_subspace_of(
                    first_group_repr_ker,
                    VarDecContext.Y
                )

                # it cannot be the case that both projected spaces are the same
                assert not (
                    first_subspace_second_x and second_subspace_first_x
                    and first_subspace_second_y and second_subspace_first_y
                )

                # true iff the first group's disjuncts have strictly more linear dependencies
                # compared to the second group's disjuncts
                first_strict_subspace_of_second_wrt_x = first_subspace_second_x and not second_subspace_first_x
                first_strict_subspace_of_second_wrt_y = first_subspace_second_y and not second_subspace_first_y

                # true iff the second group's disjuncts have strictly more linear dependencies
                # compared to the first group's disjuncts
                second_strict_subspace_of_first_wrt_x = second_subspace_first_x and not first_subspace_second_x
                second_strict_subspace_of_first_wrt_y = second_subspace_first_y and not first_subspace_second_y

                # we draw edges always from disjunct with less linear dependencies
                # to disjuncts with more linear dependencies

                if first_strict_subspace_of_second_wrt_x:
                    # first has more linear dependencies compared to second wrt x
                    x_edges.append((second_group_id, first_group_id))

                if first_strict_subspace_of_second_wrt_y:
                    # first has more linear dependencies compared to second wrt y
                    y_edges.append((second_group_id, first_group_id))

                if second_strict_subspace_of_first_wrt_x:
                    # second has more linear dependencies compared to first wrt x
                    x_edges.append((first_group_id, second_group_id))

                if second_strict_subspace_of_first_wrt_y:
                    # second has more linear dependencies compared to first wrt y
                    y_edges.append((first_group_id, second_group_id))

        ng_x = nx.DiGraph()
        ng_x.add_edges_from(x_edges)
        ng_x = nx.transitive_reduction(ng_x)

        ng_y = nx.DiGraph()
        ng_y.add_edges_from(y_edges)
        ng_y = nx.transitive_reduction(ng_y)

        g = graphviz.Digraph(
            "G",
            filename="generated_figures/group.gv",
            graph_attr={"rankdir": "LR"},
            node_attr={
                "shape": "record",
                "fontname": "Courier New",
                "fontsize": "7pt",
                "margin": "0.01",
                "width": "0.1",
                "height": "0.1"
            }
        )

        group_label_pattern = "group_%03d"

        for group in self._disjunct_groups:
            g.node(group_label_pattern % group.group_id, group.get_graphviz_group_node_label())

        x_edges_set = set(ng_x.edges)
        y_edges_set = set(ng_y.edges)

        common_edges = x_edges_set.intersection(y_edges_set)

        for u, v in common_edges:
            g.edge(group_label_pattern % u, group_label_pattern % v, label="X,Y")

        for u, v in x_edges_set:
            if (u, v) in common_edges:
                continue
            g.edge(group_label_pattern % u, group_label_pattern % v, label="X")

        for u, v in y_edges_set:
            if (u, v) in common_edges:
                continue
            g.edge(group_label_pattern % u, group_label_pattern % v, label="Y")

        return g


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

        disj_graph_builder.add_disjunct(disj_model_vec, phi_context)

        disj_count += 1

        disjunct_solver.add(z3.Or(
            *(z3.Not(d) for d in disjunct)
        ))

    print("Disjunct count: %d" % disj_count)
    g = disj_graph_builder.create_group_graph()
    g.render("group.gv", "generated_figures", engine="dot", format="svg")
