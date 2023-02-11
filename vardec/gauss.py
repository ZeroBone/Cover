from fractions import Fraction
import logging
import numpy as np


_logger = logging.getLogger("vardec")


class AffineVectorSpace:

    def __init__(self, generator: np.ndarray, span: np.ndarray, /):
        self.generator = generator
        self.span = span

    def __str__(self):
        return "%s + %s" % (self.generator, self.span)


def rref(a: np.ndarray) -> set:

    x = 0
    m, n = a.shape

    not_leading_columns = set(range(n))

    for y in range(m):

        if x >= n:
            return not_leading_columns

        y_cur = y

        # we search for some non-zero element in the x-th column to eliminate
        while a[y_cur][x] == 0:

            y_cur += 1

            if y_cur == m:
                # we reached the bottom of the matrix, reset y_cur
                # and go to the next column, if it exists
                y_cur = y
                x += 1

                if n == x:
                    return not_leading_columns

        # swap rows to ensure that the non-zero pivot element is on the diagonal
        a[[y_cur, y]] = a[[y, y_cur]]

        assert a[y][x] != 0
        not_leading_columns.discard(x)

        pivot_value = a[y][x]

        # make sure that the leading term is a one
        a[y] *= Fraction(1, pivot_value)

        # make sure that we have zeros in the entire column, except the current row y
        for y_elim in range(m):

            if y_elim == y:
                continue

            value_to_eliminate = a[y_elim][x]

            a[y_elim] -= value_to_eliminate * a[y]

        x += 1

    return not_leading_columns


def compute_kernel(a: np.ndarray) -> np.ndarray:

    n = a.shape[1]

    not_leading_columns = rref(a)

    span = np.empty((n, len(not_leading_columns)), dtype=Fraction)

    span_column = 0

    for x in not_leading_columns:
        column = a[:, x]
        column_i = 0

        for i in range(n):

            if i in not_leading_columns:
                span[i][span_column] = 1 if i == x else 0
            else:
                assert column_i < a.shape[0]
                span[i][span_column] = -column[column_i]
                column_i += 1

        span_column += 1

    return span


def compute_affine_space(a: np.ndarray, b: np.ndarray) -> AffineVectorSpace | None:

    b = -np.array([b]).T

    a = np.append(a, b, axis=1)

    span = compute_kernel(a)

    # _logger.debug("Span:\n%s", span)

    if span[:, -1][a.shape[1] - 1] != 1:
        # no solution
        return None

    span = np.delete(span, -1, axis=0)

    generator = span[:, -1]

    span = np.delete(span, -1, axis=1)

    return AffineVectorSpace(generator, span)


def check_image_space_inclusion(a_mat: np.ndarray, b_mat: np.ndarray):
    """ Tests whether the image of a_mat is a subspace of the image of b_mat """

    # we do this test by checking that every column of a_mat is in the image of b_mat

    for x in range(a_mat.shape[1]):
        column = a_mat[:, x]

        if compute_affine_space(b_mat.copy(), column.copy()) is None:
            # no solution, that is, the column vector is not in the image of b_mat
            return False

    return True


def check_image_space_equality(a_mat: np.ndarray, b_mat: np.ndarray):
    """ Tests whether the image of a_mat is equal to the image of b_mat """
    return check_image_space_inclusion(a_mat, b_mat) and check_image_space_inclusion(b_mat, a_mat)


def compute_gen_set_of_intersection_of_mat_images(a_mat: np.ndarray, b_mat: np.ndarray, /, *, debug_mode: bool = True):
    """ Computes a set of vectors generating im(a_mat) intersected with im(b_mat) """
    assert a_mat.shape[0] == b_mat.shape[0]

    a_mat_columns = a_mat.shape[1]

    u_mat = np.concatenate((a_mat, -b_mat), axis=1)

    u_mat_ker = compute_kernel(u_mat)

    # this matrix contains the topmost a_mat_columns rows which
    # describe what linear dependencies b_mat enforces on the coefficients
    # of the linear combinations of columns of a_mat
    u_mat_ker_lincomb_deps = u_mat_ker[:a_mat_columns, :]

    gen_set = a_mat @ u_mat_ker_lincomb_deps

    if debug_mode:
        assert check_image_space_inclusion(gen_set, a_mat)
        assert check_image_space_inclusion(gen_set, b_mat)

    return gen_set


if __name__ == "__main__":

    A = np.array([
        [1, 2, 2, -2, -1],
        [-2, -3, -1, 8, 1],
        [1, 4, 8, 8, -4],
        [2, 5, 7, 2, -4]
    ], dtype=Fraction)

    b_vec = np.array([-1, 1, -4, -4], dtype=Fraction)

    aff_space = compute_affine_space(A.copy(), b_vec)
    print(aff_space)

    b_vec = np.array([0, 0, 0, 0], dtype=Fraction)

    aff_space = compute_affine_space(A.copy(), b_vec)
    print(aff_space)

    print("=" * 45)

    A = np.array([
        [1, 0]
    ], dtype=Fraction)

    b_vec = np.array([2], dtype=Fraction)

    aff_space = compute_affine_space(A, b_vec)
    print(aff_space)

    print("=" * 45)
    print("Unique solution:")

    A = np.array([
        [1, 0],
        [0, 3]
    ], dtype=Fraction)

    b_vec = np.array([2, 9], dtype=Fraction)

    aff_space = compute_affine_space(A, b_vec)
    print(aff_space)

    print("=" * 45)
    print("No solution:")

    # no solution
    A = np.array([
        [1, 2],
        [1, 2]
    ], dtype=Fraction)

    b_vec = np.array([3, 4], dtype=Fraction)

    aff_space = compute_affine_space(A, b_vec)
    print(aff_space)

    print("=" * 45)
    print("Solution is the original vector space")

    A = np.array([
        [0, 0, 0]
    ], dtype=Fraction)

    b_vec = np.array([0], dtype=Fraction)

    aff_space = compute_affine_space(A, b_vec)
    print(aff_space)

    print("=" * 45)

    assert check_image_space_inclusion(
        np.array([
            [3, -3],
            [-5, 9],
            [5, 4],
            [7, -2]
        ], dtype=Fraction),
        np.array([
            [1, 2, 2, -2, -1],
            [-2, -3, -1, 8, 1],
            [1, 4, 8, 8, -4],
            [2, 5, 7, 2, -4]
        ], dtype=Fraction)
    )

    assert not check_image_space_inclusion(
        np.array([
            [3, -3],
            [-5, 9],
            [5, 5],
            [7, -2]
        ], dtype=Fraction),
        np.array([
            [1, 2, 2, -2, -1],
            [-2, -3, -1, 8, 1],
            [1, 4, 8, 8, -4],
            [2, 5, 7, 2, -4]
        ], dtype=Fraction)
    )

    assert check_image_space_equality(
        np.array([
            [3, -3, 2, 2, -2],
            [-5, 9, -3, -1, 8],
            [5, 4, 4, 8, 8],
            [7, -2, 5, 7, 2]
        ], dtype=Fraction),
        np.array([
            [1, 2, 2, -2, -1],
            [-2, -3, -1, 8, 1],
            [1, 4, 8, 8, -4],
            [2, 5, 7, 2, -4]
        ], dtype=Fraction)
    )

    print(compute_gen_set_of_intersection_of_mat_images(
        np.array([
            [2, 1],
            [-1, 0],
            [0, 2]
        ], dtype=Fraction),
        np.array([
            [-7, 5, 9, -3],
            [-2, -2, 6, -2],
            [5, -4, -6, 2]
        ], dtype=Fraction)
    ))
