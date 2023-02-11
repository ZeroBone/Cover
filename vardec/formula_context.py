import logging

import z3

from vardec_context import VarDecContext
from linconstraint import predicate_to_linear_constraint
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

        _logger.info("Constraints:\n%s", ",\n".join((str(c) for c in self.constraints)))

        # create solver for efficiently checking entailment queries

        self._entailment_solver = z3.Solver()
        self._entailment_solver.add(z3.Not(phi))

    def query_whether_formula_entails_phi(self, query_formula, /) -> bool:
        self._entailment_solver.push()
        self._entailment_solver.add(query_formula)
        result = self._entailment_solver.check()
        self._entailment_solver.pop()
        assert result == z3.sat or result == z3.unsat
        return result == z3.unsat

    def get_constraint_count(self) -> int:
        return len(self.constraints)
