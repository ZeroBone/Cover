import numpy as np


class VarDecContext:

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
