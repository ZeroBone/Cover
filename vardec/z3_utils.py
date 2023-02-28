import ctypes

# noinspection PyPackageRequirements
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


def fixes_var(psi, var):

    s = z3.Solver()
    s.add(psi)

    s.check()

    model = s.model()

    if model[var] is None:
        return False

    return is_unsat(z3.And(psi, var != model[var]))


def is_uninterpreted_variable(node, /):
    return z3.is_const(node) and node.decl().kind() == z3.Z3_OP_UNINTERPRETED


def get_formula_predicates(phi, /):

    predicates_list = []
    visited = set()

    def _ast_visitor(node):

        node_type = node.decl().kind()

        node_is_predicate = False

        if node_type in {z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT}:
            node_is_predicate = True
        elif node_type in {z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT}:
            node_is_predicate = True
            # it may be the case that the equality or inequality is actually a Boolean equivalence or
            # non-equivalence
            # thus, we need to check what operators the children have

            for child in node.children():
                if child.decl().kind() in {z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ,
                                           z3.Z3_OP_DISTINCT, z3.Z3_OP_IMPLIES, z3.Z3_OP_ITE}:
                    # the child is indeed a Boolean operator
                    node_is_predicate = False
                    break

        if node_is_predicate:
            predicates_list.append(node)
        else:
            for child in node.children():
                child_wrapped = wrap_ast_ref(child)
                if child_wrapped in visited:
                    continue
                visited.add(child_wrapped)
                _ast_visitor(child)

    visited.add(wrap_ast_ref(phi))

    try:
        _ast_visitor(phi)
    except (RecursionError, ctypes.ArgumentError):
        raise TooDeepFormulaError()

    return predicates_list


def replace_strict_inequality_by_nonstrict(predicate, /):

    pred_kind = predicate.decl().kind()

    lhs, rhs, *_ = predicate.children()

    if pred_kind == z3.Z3_OP_LT:
        return lhs <= rhs

    if pred_kind == z3.Z3_OP_GT:
        return lhs >= rhs

    if pred_kind == z3.Z3_OP_EQ:
        return predicate

    assert False, "unknown predicate type"


def get_formula_variables(phi, /):

    vars_list = []
    visited = set()

    def ast_visitor(node):
        if is_uninterpreted_variable(node):
            vars_list.append(wrap_ast_ref(node))
        else:
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

    return vars_list


def get_formula_ast_node_count(phi, /):

    visited = set()

    def ast_visitor(node):
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

    return len(visited)
