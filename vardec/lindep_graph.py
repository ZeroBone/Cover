import logging
from fractions import Fraction

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from context import VarDecContext
from gauss import compute_kernel, check_image_space_inclusion
from vardec import FormulaContext


_logger = logging.getLogger("vardec")


def _subsets_of_size(subset_size: int, total_size: int, /, determined_set: list = None, start_index: int = 0):

    det_set = [] if determined_set is None else determined_set

    if subset_size == 0:
        yield det_set
        return

    for i in range(start_index, total_size):
        # choose i as the element of the set
        yield from _subsets_of_size(subset_size - 1, total_size, det_set + [i], i + 1)


def _frozen_subset_to_label(ss: frozenset) -> str:
    return ",".join(str(n) for n in sorted(ss))


def build_lindep_graph(phi, x: list, y: list):

    context = VarDecContext(x, y)
    phi_context = FormulaContext(phi, context)

    # TODO: handle empty set separately

    kernel_table = {}

    for subset_size in range(1, phi_context.get_constraint_count()):

        _logger.debug("Subset size: %d", subset_size)

        for i_subset in _subsets_of_size(subset_size, phi_context.get_constraint_count()):
            _logger.debug("Subset: %s", i_subset)

            # we now build the constraint matrix
            constraint_matrix = np.array([
                phi_context.constraints[i].get_lin_combination_copy() for i in i_subset
            ], dtype=Fraction)

            _logger.debug("Constraint matrix: %s", constraint_matrix)

            constraint_matrix_kernel = compute_kernel(constraint_matrix)

            _logger.debug("Constraint matrix kernel: %s", constraint_matrix_kernel)

            kernel_table[frozenset(i_subset)] = constraint_matrix_kernel

    subsets = list(kernel_table.keys())
    subset_count = len(subsets)

    _logger.info("Subset count: %d", subset_count)

    g = nx.Graph()

    for subset in subsets:
        subset_label = _frozen_subset_to_label(subset)
        g.add_node(subset_label)

    for i, i_subset in enumerate(subsets):
        for j in range(i + 1, subset_count):
            j_subset = subsets[j]

            i_kernel = kernel_table[i_subset].copy()
            j_kernel = kernel_table[j_subset].copy()

            # print(i_kernel, j_kernel)

            i_kernel_x = context.select_rows_corresp_x(i_kernel)
            i_kernel_y = context.select_rows_corresp_y(i_kernel)

            j_kernel_x = context.select_rows_corresp_x(j_kernel)
            j_kernel_y = context.select_rows_corresp_y(j_kernel)

            # print(i_kernel_x, j_kernel_x)

            i_x_subsetof_j_x = check_image_space_inclusion(i_kernel_x, j_kernel_x)
            j_x_subsetof_i_x = check_image_space_inclusion(j_kernel_x, i_kernel_x)
            i_x_equal_j_x = i_x_subsetof_j_x and j_x_subsetof_i_x

            i_y_subsetof_j_y = check_image_space_inclusion(i_kernel_y, j_kernel_y)
            j_y_subsetof_i_y = check_image_space_inclusion(j_kernel_y, i_kernel_y)
            i_y_equal_j_y = i_y_subsetof_j_y and j_y_subsetof_i_y

            i_subset_label = _frozen_subset_to_label(i_subset)
            j_subset_label = _frozen_subset_to_label(j_subset)

            """
            if i_x_subsetof_j_x and not i_x_equal_j_x:
                if not nx.has_path(g, i_subset_label, j_subset_label):
                    pass
                g.add_edge(i_subset_label, j_subset_label)
            if j_x_subsetof_i_x and not i_x_equal_j_x:
                if not nx.has_path(g, j_subset_label, i_subset_label):
                    pass
                g.add_edge(j_subset_label, i_subset_label)

            if i_y_subsetof_j_y and not i_y_equal_j_y:
                g.add_edge(i_subset_label, j_subset_label)
            if j_y_subsetof_i_y and not i_y_equal_j_y:
                g.add_edge(j_subset_label, i_subset_label)
            """

            if nx.has_path(g, i_subset_label, j_subset_label):
                assert i_x_equal_j_x and i_y_equal_j_y

            if i_x_equal_j_x and i_y_equal_j_y:
                if not nx.has_path(g, i_subset_label, j_subset_label):
                    g.add_edge(i_subset_label, j_subset_label)

            # print("X:", i_x_subsetof_j_x, j_x_subsetof_i_x, i_x_equal_j_x)
            # print("Y:", i_y_subsetof_j_y, j_y_subsetof_i_y, i_y_equal_j_y)

    g.remove_nodes_from(list(nx.isolates(g)))

    # pos = nx.spring_layout(g, k=100)
    # pos = nx.spiral_layout(g, equidistant=True)
    # pos = nx.shell_layout(g)
    pos = nx.planar_layout(g)

    nx.draw_networkx_nodes(g, pos, cmap=plt.get_cmap('jet'), node_size=250)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edges(g, pos, arrows=True, edge_color="black")
    # plt.show()
    plt.savefig("graph.png")
