from fractions import Fraction
from typing import Iterable
import itertools

import numpy as np
# noinspection PyPackageRequirements
from z3 import z3

from z3_utils import get_formula_variables


class VarDecContext:

    X = 0
    Y = 1

    def __init__(self, x: list, y: list, /, *,
                 debug_mode: bool = False, use_heuristics: bool = True, use_blast_heuristic: bool = True):
        assert len(x) > 0
        assert len(y) > 0
        self._x = x
        self._y = y
        self.debug_mode = debug_mode
        self.use_heuristics = use_heuristics
        self.use_blast_heuristic = use_blast_heuristic
        self._stat_cover_calls = 0
        self._stat_indistinguishable_disjuncts_count = 0
        self._stat_distinguishable_disjuncts_count = 0
        self._stat_heuristic_success_count = 0
        self._stat_heuristic_fail_count = 0
        self._stat_blast_heuristic_success_count = 0
        self._stat_blast_heuristic_fail_count = 0
        assert set(self._x).isdisjoint(set(self._y))

    def variable_count(self) -> int:
        return len(self._x) + len(self._y)

    def index_to_variable(self, i: int, /):

        assert i >= 0
        assert i < self.variable_count()

        if i >= len(self._x):
            return self._y[i - len(self._x)]

        return self._x[i]

    def variable_to_index(self, var, /) -> int:

        for i, v in enumerate(itertools.chain(self._x, self._y)):
            if v == var:
                return i

        assert False

    def project_vector_onto_block(self, vec: np.ndarray, block: int, /) -> np.ndarray:
        """ Project a one-dimensional vector onto the specified block of the partition """
        if block == VarDecContext.X:
            return vec[:len(self._x)]
        if block == VarDecContext.Y:
            return vec[len(self._x):]
        assert False

    def project_vector_back_from_block(self, vec: np.ndarray, block: int, /) -> np.ndarray:
        if block == VarDecContext.X:
            return np.concatenate((vec, np.zeros(len(self._y), dtype=Fraction)))
        if block == VarDecContext.Y:
            return np.concatenate((np.zeros(len(self._x), dtype=Fraction), vec))
        assert False

    def project_matrix_onto_block(self, mat: np.ndarray, block: int, /) -> np.ndarray:
        """ Project a one-dimensional vector onto the specified block of the partition """
        if block == VarDecContext.X:
            return mat[:len(self._x), :]
        if block == VarDecContext.Y:
            return mat[len(self._x):, :]
        assert False

    def block_variables_iter(self, block: int, /) -> Iterable:
        if block == VarDecContext.X:
            return iter(self._x)
        if block == VarDecContext.Y:
            return iter(self._y)
        assert False

    def block_linear_comb_to_expr(self, lincomb_coeffs: np.ndarray, block: int, /):

        assert block == VarDecContext.X or block == VarDecContext.Y

        var_list = self._x if block == VarDecContext.X else self._y

        assert lincomb_coeffs.shape[0] == len(var_list)

        return z3.Sum(*(
            lincomb_coeffs[var_id] * var
            for var_id, var in enumerate(var_list) if lincomb_coeffs[var_id] != 0
        ))

    def predicate_respects_pi(self, predicate) -> bool:
        pred_vars = set(get_formula_variables(predicate))
        return pred_vars.isdisjoint(self._x) or pred_vars.isdisjoint(self._y)

    def predicate_lincomb_respects_pi(self, lincomb: np.ndarray, /) -> bool:
        if not np.any(self.project_vector_onto_block(lincomb, VarDecContext.X)):
            return True
        return not np.any(self.project_vector_onto_block(lincomb, VarDecContext.Y))

    def model_to_vec(self, model) -> np.ndarray:

        model_vec = np.zeros((self.variable_count()), dtype=Fraction)

        for i, v in enumerate(itertools.chain(self._x, self._y)):

            var_val = Fraction(0)

            if model[v] is not None:
                assert isinstance(model[v], z3.RatNumRef)
                var_val = Fraction(model[v].numerator_as_long(), model[v].denominator_as_long())

            model_vec[i] = var_val

        return model_vec

    def vector_to_enforcing_expr(self, vec: np.ndarray):
        return z3.And(*(
            var == vec[var_id]
            for var_id, var in enumerate(itertools.chain(self._x, self._y))
        ))

    def stat_on_cover_call(self):
        self._stat_cover_calls += 1

    def stat_on_distinguishable_disjunct(self):
        self._stat_distinguishable_disjuncts_count += 1

    def stat_on_indistinguishable_disjunct(self):
        self._stat_indistinguishable_disjuncts_count += 1

    def stat_on_heuristic_success(self):
        self._stat_heuristic_success_count += 1

    def stat_on_heuristic_fail(self):
        self._stat_heuristic_fail_count += 1

    def stat_on_blast_heuristic_success(self):
        self._stat_blast_heuristic_success_count += 1

    def stat_on_blast_heuristic_fail(self):
        self._stat_blast_heuristic_fail_count += 1

    def print_stats(self):
        print("=== [Statistics] ===")
        print("Cover calls: %d" % self._stat_cover_calls)
        print("Distinguishable disjunct count: %d" % self._stat_distinguishable_disjuncts_count)
        print("Indistinguishable disjunct count: %d" % self._stat_indistinguishable_disjuncts_count)
        print(
            "Heuristic success rate: %d/%d" %
            (self._stat_heuristic_success_count, self._stat_heuristic_success_count + self._stat_heuristic_fail_count)
        )
        print(
            "Blast heuristic success rate: %d/%d" %
            (self._stat_blast_heuristic_success_count,
             self._stat_blast_heuristic_success_count + self._stat_blast_heuristic_fail_count)
        )


def block_str(block: int, /) -> str:
    assert block == VarDecContext.X or block == VarDecContext.Y
    return "X" if block == VarDecContext.X else "Y"
