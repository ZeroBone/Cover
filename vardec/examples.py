from z3 import *

from z3_utils import is_valid, is_sat, is_unsat


def test_1():
    x, y = Reals("x y")

    phi = Not(Or(
        And(x <= -2, y > 1, x - y <= 1),
        And(x == -2, y >= 1, x - y <= 1),
        And(y <= 1, x >= 2, x - y >= 1),
        And(y >= 1, x > 2)
    ))

    return phi, [x], [y]


def simple_line_inference():
    x, y = Reals("x y")

    phi = And(
        y == 10,
        x + y <= 15
    )

    return phi, [x], [y]


def _addition_formula(n: int):

    xy = Reals("x " + " ".join("y_%d" % (i + 1) for i in range(n)))

    return And(
        Sum(*[xy[i + 1] for i in range(n)]) - xy[0] < 0,
        -1 < Sum(*[xy[i + 1] for i in range(n)]) - xy[0],
        *[
            Or(xy[i] == 0, xy[i] == 2 ** i)
            for i in range(1, n + 1)
        ]
    ), xy[:1], xy[1:]


def addition_formula_2():
    return _addition_formula(2)


def addition_formula_3():
    return _addition_formula(3)


def addition_formula_4():
    return _addition_formula(4)


def decomposable_x_complex_border():
    # this formula is monadically decomposable on x, since it is equivalent to z == 0

    x, y, z = Reals("x y z")

    return And(
        Or(x + y + 2 * z == 0, x + y + 2 * z != 0),
        z == 0
    ), [x], [y, z]


def decomposable_x_complex_border_2():
    # this formula is monadically decomposable on x, since it is equivalent to z == 0

    x, y, z = Reals("x y z")

    return And(
        Or(x + y + 2 * z == 0, x + y + 2 * z != 0),
        Or(x + y == 0, x + y != 0),
        z == 0
    ), [x], [y, z]


def h_simple_not_empty():

    x, y, z = Reals("x y z")

    return And(
        Or(x + y + 2 * z == 0, x + y + 2 * z != 0),
        Or(x + y == 0, x + y != 0),
        Or(x - z == 0, x - z != 0),
        Not(And(
            x + y + 2 * z == 0,
            x + y == 0,
            x - z == 0
        ))
    ), [x], [y, z]


def h_simple_not_empty_2():

    x, y = Reals("x y")

    return Not(And(
        x + y == 0,
        x - y == 0
    )), [x], [y]


def three_x_complex_atoms():

    x, y = Reals("x y")

    return And(
        Or(x - 2 * y == 2, x - 2 * y != 2),
        Not(And(
            x + y == 0,
            x - y == 0
        ))
    ), [x], [y]


def x_complex_region_inside_x_simple_region():

    x, y, z = Reals("x y z")

    return Or(
        And(x - y == 0, x - y + z == 0),
        z == 0
    ), [x], [y, z]


def x_equals_y():

    x, y = Reals("x y")

    return x - y == 0, [x], [y]


def x_eq_0_or_y_eq_0():

    x, y = Reals("x y")

    return Or(x == 0, y == 0), [x], [y]


def plane_parallel_intersection():

    x_1, x_2, x_3, y_1, y_2, y_3 = Reals("x_1 x_2 x_3 y_1 y_2 y_3")

    phi_1_eq = 12 * x_1 - 2 * x_2 + 16 * x_3 + 28 * y_1 + 35 * y_2 - 7 * y_3 == 0

    phi_2_eq = 6 * x_1 - x_2 + 8 * x_3 + 16 * y_1 + 20 * y_2 - 4 * y_3 == 0

    a_space = And(phi_1_eq, phi_2_eq)

    a_space_dec = And(
        6 * x_1 - x_2 + 8 * x_3 == 0,
        4 * y_1 + 5 * y_2 - y_3 == 0
    )

    assert is_valid(a_space == a_space_dec)

    b_space = And(
        6 * x_1 - x_2 + 8 * x_3 == 0,
        2 * x_1 - x_2 + 2 * y_1 + 2 * y_2 == 0,
        10 * x_1 - 5 * x_2 + 2 * y_1 + 2 * y_3 == 0
    )

    assert is_valid(Implies(b_space, a_space))

    # corresponds to p
    p_1 = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 == 0

    assert is_valid(b_space == And(a_space, p_1))

    p_2 = 6 * x_1 - 3 * x_2 + 2 * y_1 + y_2 + y_3 == 0

    assert is_valid(b_space == And(a_space, p_2))

    p_1_opposite = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 < 0
    p_2_opposite = 6 * x_1 - 3 * x_2 + 2 * y_1 + y_2 + y_3 >= 0

    assert is_unsat(And(phi_1_eq, phi_2_eq, p_1_opposite, p_2_opposite))
    assert is_valid(And(phi_1_eq, phi_2_eq) == And(phi_1_eq, phi_2_eq, Or(p_1_opposite, p_2_opposite)))
    assert not is_valid(Or(p_1_opposite, p_2_opposite))
    assert not is_valid(And(phi_1_eq) == And(phi_1_eq, Or(p_1_opposite, p_2_opposite)))
    assert not is_valid(And(phi_2_eq) == And(phi_2_eq, Or(p_1_opposite, p_2_opposite)))

    phi_3_lt = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 < 1
    phi_3_eq = 4 * x_1 - x_2 + 4 * x_3 + y_1 + y_2 == 1

    phi_4_gt = 6 * x_1 - 3 * x_2 + 2 * y_1 + y_2 + y_3 > 0

    assert is_valid(And(phi_1_eq, phi_2_eq) == And(phi_1_eq, phi_2_eq, Or(phi_3_lt, phi_4_gt)))
    assert not is_valid(And(phi_1_eq) == And(phi_1_eq, Or(phi_3_lt, phi_4_gt)))
    assert not is_valid(And(phi_2_eq) == And(phi_2_eq, Or(phi_3_lt, phi_4_gt)))

    assert is_sat(And(phi_1_eq, phi_2_eq, phi_3_lt, phi_4_gt))
    assert is_sat(And(phi_1_eq, phi_2_eq, phi_3_lt, Not(phi_4_gt)))
    assert is_sat(And(phi_1_eq, phi_2_eq, Not(phi_3_lt), phi_4_gt))

    assert is_sat(And(phi_1_eq, Not(phi_3_lt), Not(phi_4_gt)))
    assert is_sat(And(phi_2_eq, Not(phi_3_lt), Not(phi_4_gt)))
    assert is_unsat(And(phi_1_eq, phi_2_eq, Not(phi_3_lt), Not(phi_4_gt)))

    # c space

    c_space = And(
        2 * x_1 - x_2 == 0,
        x_1 + 2 * x_3 == 0,
        y_1 + y_2 == 0,
        y_1 + y_3 == 0
    )

    p_c = 3 * x_1 - x_2 + 2 * x_3 + 2 * y_1 + y_2 + y_3 == 0

    assert is_valid(c_space == And(b_space, p_c))
    assert is_valid(c_space == And(a_space, p_2, p_c))

    phi_5_neq = 3 * x_1 - x_2 + 2 * x_3 + 2 * y_1 + y_2 + y_3 != 0

    # put everything together

    phi = And(phi_1_eq, phi_2_eq, Or(phi_3_lt, phi_4_gt), Implies(phi_3_eq, phi_5_neq))

    return phi, [x_1, x_2, x_3], [y_1, y_2, y_3]


def three_sandwiched_spaces():
    x_1, x_2, x_3, x_4, x_5, x_6 = z3.Reals("x_1 x_2 x_3 x_4 x_5 x_6")

    phi_1 = x_1 + x_3 + 3 * x_4 + 9 * x_5 - 6 * x_6 == 0
    phi_2 = x_1 + x_3 + 2 * x_4 + 6 * x_5 - 4 * x_6 == 0
    phi_3 = x_1 + x_3 == 0
    phi_4 = -11 * x_1 + 5 * x_2 + 3 * x_4 == 0
    phi_5 = x_1 - x_2 + x_5 == 0
    phi_6 = x_1 + 2 * x_2 - 3 * x_6 < 0
    phi_7 = x_1 - 7 * x_2 + 3 * x_4 + 12 * x_5 == 0
    phi_8 = -x_1 + 7 * x_2 + x_4 - 8 * x_6 == 0

    phi = z3.Or(
        z3.And(phi_1, phi_2, z3.Implies(z3.And(phi_3, phi_4, phi_5), phi_6)),
        z3.And(phi_3, phi_7, phi_8)
    )

    assert is_sat(phi)

    # Check B => C subspace inclusion
    assert is_valid(z3.Implies(
        z3.And(phi_3, phi_4, phi_5, x_1 + 2 * x_2 - 3 * x_6 == 0),
        z3.And(phi_3, phi_7, phi_8)
    ))

    # Check C => A subspace inclusion
    assert is_valid(z3.Implies(
        z3.And(phi_3, phi_7, phi_8),
        z3.And(phi_1, phi_2)
    ))

    phi_decomposition = z3.And(
        x_1 + x_3 == 0,
        x_4 + 3 * x_5 - 2 * x_6 == 0
    )

    # Check equivalence
    assert is_valid(phi == phi_decomposition)

    return phi, [x_1, x_2, x_3], [x_4, x_5, x_6]
