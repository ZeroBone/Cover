import logging
from fractions import Fraction

import numpy as np

from context import VarDecContext
from linconstraint import predicate_to_linear_constraint
from gauss import compute_kernel, check_image_space_inclusion
from z3_utils import get_formula_predicates


_logger = logging.getLogger("vardec")


def _subsets_of_size(subset_size: int, total_size: int, /, determined_set: list = None, start_index: int = 0):

    det_set = [] if determined_set is None else determined_set

    if subset_size == 0:
        yield det_set
        return

    for i in range(start_index, total_size):
        # choose i as the element of the set
        yield from _subsets_of_size(subset_size - 1, total_size, det_set + [i], i + 1)


def vardec(phi, x: list, y: list):

    context = VarDecContext(x, y)

    _logger.info("Formula:\n%s", phi)

    constraints = []

    for predicate in get_formula_predicates(phi):
        _logger.info("Predicate: %s", predicate)

        constraint = predicate_to_linear_constraint(context, predicate)

        if constraint not in constraints:
            constraints.append(constraint)

    _logger.info("Constraints:\n%s", ",\n".join((str(c) for c in constraints)))

    constraint_count = len(constraints)

    # TODO: handle empty set separately

    kernel_table = {}

    for subset_size in range(1, constraint_count):

        _logger.debug("Subset size: %d", subset_size)

        for i_subset in _subsets_of_size(subset_size, constraint_count):
            _logger.debug("Subset: %s", i_subset)

            # we now build the constraint matrix
            constraint_matrix = np.array([
                constraints[i].get_lin_combination_copy() for i in i_subset
            ], dtype=Fraction)

            _logger.debug("Constraint matrix: %s", constraint_matrix)

            constraint_matrix_kernel = compute_kernel(constraint_matrix)

            _logger.debug("Constraint matrix kernel: %s", constraint_matrix_kernel)

            kernel_table[frozenset(i_subset)] = constraint_matrix_kernel

    subsets = list(kernel_table.keys())
    subset_count = len(subsets)

    _logger.info("Subset count: %d", subset_count)

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

            print("X:", i_x_subsetof_j_x, j_x_subsetof_i_x, i_x_equal_j_x)
            print("Y:", i_y_subsetof_j_y, j_y_subsetof_i_y, i_y_equal_j_y)

