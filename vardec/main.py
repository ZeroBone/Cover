import logging
import os
import sys
from pathlib import Path
from inspect import isfunction, getmodule

import z3

import examples
from vardec import vardec
from z3_utils import is_valid


_logger = logging.getLogger("vardec")


def _resolve_base_path():
    base_path = Path(__file__).parent
    return (base_path / "../").resolve()


def _main():
    # initialize the logger

    verbose_mode = "--verbose" in sys.argv[1:]
    debug_mode = "--debug" in sys.argv[1:]
    use_heuristics = "--no-heuristics" not in sys.argv[1:]

    print("Verbose mode: %s" % ("enabled" if verbose_mode else "disabled"))
    print("Debug mode: %s" % ("enabled" if debug_mode else "disabled"))
    print("Using heuristics: %s" % ("enabled" if use_heuristics else "disabled"))

    _logger.setLevel(logging.DEBUG if verbose_mode else logging.INFO)

    file_handler = logging.FileHandler(os.path.join(_resolve_base_path(), "vardec.log"), mode="w")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s %(levelname)7s]: %(message)s")
    file_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)

    example_formulas = [
        (func.__name__, func)
        for func in examples.__dict__.values()
        if isfunction(func) and getmodule(func) == examples and func.__name__[0] != "_"
    ]

    print("Following example formulas are available for testing:")
    for i, (formula_name, _) in enumerate(example_formulas):
        print("%5d: %s" % (i + 1, formula_name))

    formula_id = int(input("Enter formula id: ")) - 1

    if formula_id < 0 or formula_id >= len(example_formulas):
        print("Error: invalid formula id.")
        return

    phi, x, y = example_formulas[formula_id][1]()

    _logger.info(
        "Partition: Pi = {{%s}, {%s}}",
        ", ".join(str(x_var) for x_var in x),
        ", ".join(str(y_var) for y_var in y),
    )

    decomposition = vardec(phi, x, y, debug_mode, use_heuristics)

    _logger.info(("=" * 20) + " [RESULT] " + ("=" * 20))

    if decomposition is None:
        print("Verdict: phi is not Pi-decomposable (see logs for the details)")
        _logger.info("Verdict: phi is not Pi-decomposable")
        _logger.info("=" * 51)
        return

    print("Verdict: phi is Pi-decomposable")
    _logger.info("Verdict: phi is Pi-decomposable")

    if is_valid(phi == decomposition):
        s = "Test PASS: decomposition is equivalent to the original formula"
        _logger.info(s)
        print(s)
    else:
        s = "Test FAIL: decomposition is not equivalent to the original formula"
        _logger.error(s)
        print(s)

    _logger.info("=" * 51)

    decomposition = z3.simplify(decomposition, elim_sign_ext=False, local_ctx=True)

    _logger.info("Variable decomposition:\n%s", decomposition)


if __name__ == '__main__':
    _main()
