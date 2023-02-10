from fractions import Fraction

import numpy as np
from z3 import z3


class VarDecContext:

    X = 0
    Y = 1

    def __init__(self, x: list, y: list, /):
        self.x = x
        self.y = y
        assert set(self.x).isdisjoint(set(self.y))

    def variable_count(self) -> int:
        return len(self.x) + len(self.y)

    def index_to_variable(self, i: int, /):

        assert i >= 0
        assert i < self.variable_count()

        if i >= len(self.x):
            return self.y[i - len(self.x)]

        return self.x[i]

    def variable_to_index(self, var, /) -> int:

        for i, v in enumerate(self.x + self.y):
            if v == var:
                return i

        raise IndexError

    def select_rows_corresp_x(self, m: np.ndarray):
        return m[tuple(self.variable_to_index(v) for v in self.x), :]

    def select_rows_corresp_y(self, m: np.ndarray):
        return m[tuple(self.variable_to_index(v) for v in self.y), :]

    def select_entries_corresp_x(self, m: np.ndarray):
        return m[:len(self.x)]

    def select_entries_corresp_y(self, m: np.ndarray):
        return m[len(self.x):]

    def x_or_y_linear_comb_to_z3_expr(self, lincomb_coeffs: np.ndarray, wrt_x: bool, /):

        var_list = self.x if wrt_x else self.y

        assert lincomb_coeffs.shape[0] == len(var_list)
        return z3.Sum(*(
            lincomb_coeffs[variable_id] * var
            for variable_id, var in enumerate(var_list) if lincomb_coeffs[variable_id] != 0
        ))

    def model_to_vec(self, model) -> np.ndarray:

        model_vec = np.zeros((self.variable_count()), dtype=Fraction)

        for i, v in enumerate(self.x + self.y):

            var_val = Fraction(0)

            if model[v] is not None:
                assert isinstance(model[v], z3.RatNumRef)
                var_val = Fraction(model[v].numerator_as_long(), model[v].denominator_as_long())

            model_vec[i] = var_val

        return model_vec
