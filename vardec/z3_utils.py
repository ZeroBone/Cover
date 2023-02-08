import ctypes

import z3 as z3


class TooDeepFormulaError(Exception):
    pass


class AstReferenceWrapper:

    def __init__(self, node, /):
        self._node = node

    def __hash__(self):
        return self._node.hash()

    def __eq__(self, other):
        return self._node.eq(other.unwrap())

    def __repr__(self):
        return str(self._node)

    def unwrap(self):
        return self._node


def wrap_ast_ref(node, /):
    assert isinstance(node, z3.AstRef)
    return AstReferenceWrapper(node)


def is_unsat(psi):
    s = z3.Solver()
    s.add(psi)
    check_result = s.check()
    assert check_result != z3.unknown
    return check_result == z3.unsat


def is_valid(psi):
    return is_unsat(z3.Not(psi))


def is_sat(psi):
    return not is_unsat(psi)


def is_uninterpreted_variable(node, /):
    return z3.is_const(node) and node.decl().kind() == z3.Z3_OP_UNINTERPRETED


def get_formula_predicates(phi, /):

    predicates_list = []
    visited = set()

    def ast_visitor(node):

        node_type = node.decl().kind()

        if node_type in [z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT]:
            predicates_list.append(node)

        for child in node.children():

            child_wrapped = wrap_ast_ref(child)

            if child_wrapped in visited:
                continue

            visited.add(child_wrapped)

            ast_visitor(child)

    visited.add(wrap_ast_ref(phi))

    try:
        ast_visitor(phi)
    except (RecursionError, ctypes.ArgumentError):
        raise TooDeepFormulaError()

    return predicates_list
