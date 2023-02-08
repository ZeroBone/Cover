class VarDecContext:

    def __init__(self, x: list, y: list, /):
        self._vars = x + y
        self.x = set(x)
        self.y = set(y)
        assert self.x.isdisjoint(self.y)

    def variable_count(self) -> int:
        return len(self._vars)

    def index_to_variable(self, i: int, /):
        return self._vars[i]

    def variable_to_index(self, var, /) -> int:

        for i, v in enumerate(self._vars):
            if v == var:
                return i

        raise IndexError
