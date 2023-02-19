import logging

import numpy as np
import z3

from vardec_context import VarDecContext
from linear_constraint import predicate_to_linear_constraint
from z3_utils import get_formula_predicates

_logger = logging.getLogger("vardec")


class FormulaContext:

    def __init__(self, phi, context: VarDecContext, /):
        self.phi = phi

        _logger.info("Formula:\n%s", phi)

        self.constraints = []

        for predicate in get_formula_predicates(phi):
            _logger.info("Predicate: %s", predicate)

            constraint = predicate_to_linear_constraint(context, predicate)

            if constraint not in self.constraints:
                self.constraints.append(constraint)

        _logger.info("Constraints:\n%s", ",\n".join(("%04d : %s" % (i, c) for i, c in enumerate(self.constraints))))

        # create solver for efficiently checking entailment queries

        self._entailment_solver = z3.Solver()
        self._entailment_solver.add(z3.Not(phi))

    def model_check(self, model_vec: np.ndarray, context: VarDecContext, /) -> bool:
        return self.query_whether_formula_entails_phi(
            context.vector_to_enforcing_expr(model_vec)
        )

    def model_vec_to_tag(self, model_vec: np.ndarray, /) -> str:
        """ Returns a short, readable and unique identifier of the disjunct containing this model"""
        return "".join(
            ct.get_predicate_symbol_satisfying_model(model_vec)
            for ct in self.constraints
        )

    def query_whether_formula_entails_phi(self, query_formula, /) -> bool:
        self._entailment_solver.push()
        self._entailment_solver.add(query_formula)
        result = self._entailment_solver.check()
        self._entailment_solver.pop()
        assert result == z3.sat or result == z3.unsat
        return result == z3.unsat

    def get_constraint_count(self) -> int:
        return len(self.constraints)
