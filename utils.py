import z3 as z3

from z3_utils import is_unsat


def fixes_var(psi, var):

    s = z3.Solver()
    s.add(psi)

    s.check()

    model = s.model()

    # print(model)
    # print(model[var])

    if model[var] is None:
        print("WARN: variable is not in the model")
        return False

    # print(z3.And(psi, var != model[var]))

    return is_unsat(z3.And(psi, var != model[var]))
