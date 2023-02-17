from fractions import Fraction
from numbers import Rational

import numpy as np
import z3 as z3
from z3 import ExprRef, RatNumRef

from vardec_context import VarDecContext
from z3_utils import is_uninterpreted_variable


class LinearConstraint:

    PRED_LT = "<"
    PRED_EQ = "="
    PRED_GT = ">"

    def __init__(self, lhs_linear_combination: np.ndarray, rhs_constant: Rational, /):
        self._lhs_linear_combination = lhs_linear_combination
        self._rhs_constant = rhs_constant

    def get_lhs_linear_combination_expr(self, context: VarDecContext, /):
        return z3.Sum(*(
            coeff * context.index_to_variable(variable_id)
            for variable_id, coeff in enumerate(self._lhs_linear_combination) if coeff != 0
        ))

    def get_lhs_linear_combination_vector(self) -> np.ndarray:
        return self._lhs_linear_combination.copy()

    def get_rhs_constrant(self) -> Rational:
        return self._rhs_constant

    def model_satisfies_equality_version(self, model_vec: np.ndarray, /):
        lhs_value = np.dot(self._lhs_linear_combination, model_vec)
        assert isinstance(lhs_value, Rational)
        assert isinstance(self._rhs_constant, Rational)
        return lhs_value == self._rhs_constant

    def get_predicate_symbol_satisfying_model(self, model_vec: np.ndarray, /) -> str:

        lhs_value = np.dot(self._lhs_linear_combination, model_vec)

        if lhs_value < self._rhs_constant:
            return LinearConstraint.PRED_LT

        if lhs_value > self._rhs_constant:
            return LinearConstraint.PRED_GT

        return LinearConstraint.PRED_EQ

    def get_version_satisfying_model(self, context: VarDecContext, model_vec: np.ndarray, /):

        lhs_value = np.dot(self._lhs_linear_combination, model_vec)

        lin_comb_z3 = self.get_lhs_linear_combination_expr(context)

        if lhs_value < self._rhs_constant:
            return lin_comb_z3 < self._rhs_constant

        if lhs_value > self._rhs_constant:
            return lin_comb_z3 > self._rhs_constant

        return lin_comb_z3 == self._rhs_constant

    def get_equality_expr(self, context: VarDecContext, /):
        lin_comb_z3 = self.get_lhs_linear_combination_expr(context)
        return lin_comb_z3 == self._rhs_constant

    def respects_pi(self, context: VarDecContext, /) -> bool:
        return context.predicate_lincomb_respects_pi(self._lhs_linear_combination)

    def __hash__(self) -> int:
        return hash((np.dot(self._lhs_linear_combination, self._lhs_linear_combination), self._rhs_constant))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, LinearConstraint):
            return False

        if self._rhs_constant != o._rhs_constant:
            return False

        return np.array_equal(self._lhs_linear_combination, o._lhs_linear_combination)

    def __ne__(self, o: object) -> bool:
        return not (self == o)

    def __neg__(self):
        return LinearConstraint(-self._lhs_linear_combination, -self._rhs_constant)

    def __str__(self):
        return "%s = %s" % (self._lhs_linear_combination, self._rhs_constant)


def _z3_linear_combination_to_rational_vector(context: VarDecContext, z3_sum, /) -> np.ndarray:

    if is_uninterpreted_variable(z3_sum):
        # noinspection PyTypeChecker
        coeffs = np.zeros((context.variable_count(),), dtype=Fraction)

        # set the component corresponding to the variable to one
        coeffs[context.variable_to_index(z3_sum)] = Fraction(1)

        return coeffs

    node_type = z3_sum.decl().kind()

    if node_type == z3.Z3_OP_UMINUS:
        # unary minus
        return -_z3_linear_combination_to_rational_vector(context, z3_sum.children()[0])

    if node_type == z3.Z3_OP_ADD:
        # noinspection PyTypeChecker
        coeffs = np.zeros((context.variable_count(),), dtype=Fraction)

        for operand in z3_sum.children():
            operand_coeffs = _z3_linear_combination_to_rational_vector(context, operand)
            assert coeffs.dot(operand_coeffs) == 0, "Some addition in the linear combination is not simplified"
            coeffs += operand_coeffs

        return coeffs

    if node_type == z3.Z3_OP_SUB:
        assert len(z3_sum.children()) == 2
        left_arr = _z3_linear_combination_to_rational_vector(context, z3_sum.children()[0])
        right_arr = _z3_linear_combination_to_rational_vector(context, z3_sum.children()[1])
        assert left_arr.dot(right_arr) == 0, "Some subtraction in the linear combination is not simplified"
        return left_arr - right_arr

    assert node_type == z3.Z3_OP_MUL, "Unexpected operation in linear constraint"
    assert len(z3_sum.children()) == 2

    mul_lhs = z3_sum.children()[0]
    mul_rhs = z3_sum.children()[1]

    if mul_rhs.decl().kind() == z3.Z3_OP_ANUM and isinstance(mul_rhs, RatNumRef):
        mul_lhs, mul_rhs = mul_rhs, mul_lhs

    assert mul_lhs.decl().kind() == z3.Z3_OP_ANUM and isinstance(mul_lhs, RatNumRef), \
        "Linear combination coefficients must be rational numbers"

    coeff = Fraction(mul_lhs.numerator_as_long(), mul_lhs.denominator_as_long())
    return coeff * _z3_linear_combination_to_rational_vector(context, mul_rhs)


def predicate_to_linear_constraint(context: VarDecContext, predicate, /) -> LinearConstraint:

    left_operand: ExprRef = predicate.children()[0]
    right_operand: ExprRef = predicate.children()[1]

    assert left_operand.decl().kind() == z3.Z3_OP_ANUM or right_operand.decl().kind() == z3.Z3_OP_ANUM

    if right_operand.decl().kind() != z3.Z3_OP_ANUM:
        left_operand, right_operand = right_operand, left_operand

    assert right_operand.decl().kind() == z3.Z3_OP_ANUM
    assert isinstance(right_operand, RatNumRef)

    lhs_linear_combination = _z3_linear_combination_to_rational_vector(context, left_operand)
    rhs_constant = Fraction(right_operand.numerator_as_long(), right_operand.denominator_as_long())

    return LinearConstraint(lhs_linear_combination, rhs_constant)
