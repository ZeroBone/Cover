from typing import List

import z3

from disjunct_graph import render_disjunct_graph
from formula_context import FormulaContext
from linear_constraint import LinearConstraint
from vardec_context import VarDecContext


class CoverVisualizer:

    def __init__(self):
        pass

    def on_cover_init_and_ret_pi_simple(self):
        raise NotImplementedError()

    def on_cover_init_pi_complex(self, domain_eq_constraints: List[LinearConstraint], theta):
        raise NotImplementedError()

    def create_visualizer_for_recursive_call(self):
        raise NotImplementedError()


class DummyCoverVisualizer(CoverVisualizer):

    def __init__(self):
        CoverVisualizer.__init__(self)

    def on_cover_init_and_ret_pi_simple(self):
        pass

    def on_cover_init_pi_complex(self, domain_eq_constraints: List[LinearConstraint], theta):
        pass

    def create_visualizer_for_recursive_call(self):
        return self


class ActualCoverVisualizer(CoverVisualizer):

    def __init__(self, context: VarDecContext, phi_context: FormulaContext, /, *,
                 level: int = 0, rec_call_number: int = 1, gamma_tag: str):
        CoverVisualizer.__init__(self)
        self._context = context
        self._phi_context = phi_context
        self._level = level
        self._rec_call_number = rec_call_number
        self._rec_call_counter = 0
        self._gamma_tag = gamma_tag

    def on_cover_init_and_ret_pi_simple(self):
        print("Pi simple!")

    def on_cover_init_pi_complex(self, domain_eq_constraints: List[LinearConstraint], theta):
        print("Pi complex! Level: %4d Recursive call number: %4d" % (self._level, self._rec_call_number))
        print("Gamma tag: %s" % self._gamma_tag)
        if len(domain_eq_constraints) > 0:
            print("Domain equality constraints: %s" %
                  [c.get_equality_expr(self._context) for c in domain_eq_constraints])

        assert self._level == len(domain_eq_constraints)

        render_disjunct_graph(
            self._context,
            self._phi_context,
            domain_eq_constraints,
            z3.And(*(ct.get_equality_expr(self._context) for ct in domain_eq_constraints)),
            file_name="gamma_%s_%03d_%03d" % (self._gamma_tag, self._level, self._rec_call_number),
            highlight_disjunct_domain=theta
        )

    def create_visualizer_for_recursive_call(self):

        self._rec_call_counter += 1

        return ActualCoverVisualizer(
            self._context,
            self._phi_context,
            level=self._level + 1,
            rec_call_number=self._rec_call_counter,
            gamma_tag=self._gamma_tag
        )


class Visualizer:

    def __init__(self):
        self._context = None
        self._phi_context = None

    def set_contexts(self, context: VarDecContext, phi_context: FormulaContext, /):
        self._context = context
        self._phi_context = phi_context

    def get_cover_visualizer_for_next_gamma(self, gamma_model, /) -> ActualCoverVisualizer:

        assert self._context is not None
        assert self._phi_context is not None

        gamma_tag = self._phi_context.model_vec_to_tag(self._context.model_to_vec(gamma_model))

        return ActualCoverVisualizer(self._context, self._phi_context, gamma_tag=gamma_tag)
