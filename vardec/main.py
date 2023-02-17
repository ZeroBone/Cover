import argparse
import logging
import os
import time
from pathlib import Path
from inspect import isfunction, getmodule

import z3

import examples
from vardec import vardec
from vardec_context import VarDecContext
from visualizer import Visualizer
from z3_utils import is_valid


_logger = logging.getLogger("vardec")


def _resolve_base_path():
    base_path = Path(__file__).parent
    return (base_path / "../").resolve()


def _main():
    parser = argparse.ArgumentParser(
        prog="PresVarDec",
        description="Tool for deciding monadic and variable decomposition of linear real arithmetic",
        epilog="See the GitHub repository README for more information")

    parser.add_argument("-v", "--verbose", action="store_true", help="enable logging of all debug information")
    parser.add_argument("-d", "--debug", action="store_true", help="enable expensive invariant checks")
    parser.add_argument("-i", "--vis", action="store_true", help="turn on algorithm visualization mode")
    parser.add_argument("-s", "--no-heuristics", action="store_true", help="disable the usage of heuristics")
    parser.add_argument("-b", "--no-blast", action="store_true", help="disable the usage of the blast heuristic")

    args = parser.parse_args()

    use_heuristics = not args.no_heuristics
    use_blast_heuristic = not args.no_blast

    print("Verbose mode: %s" % ("enabled" if args.verbose else "disabled"))
    print("Debug mode: %s" % ("enabled" if args.debug else "disabled"))
    print("Visualization mode: %s" % ("enabled" if args.vis else "disabled"))
    print("Using heuristics: %s" % ("yes" if use_heuristics else "no"))
    print("Using the blast heuristic: %s" % ("yes" if use_blast_heuristic else "no"))

    # initialize the logger

    _logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

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

    context = VarDecContext(
        x,
        y,
        debug_mode=args.debug,
        use_heuristics=use_heuristics,
        use_blast_heuristic=use_blast_heuristic
    )

    visualizer = Visualizer() if args.vis else None

    time_start = time.perf_counter()

    decomposition = vardec(
        phi,
        x,
        y,
        context=context,
        visualizer=visualizer
    )

    vardec_time = time.perf_counter() - time_start

    _logger.info(("=" * 20) + " [RESULT] " + ("=" * 20))

    if decomposition is None:
        print("=== [Result] ===")
        print("Verdict: phi is not Pi-decomposable (see logs for the details)")
        _logger.info("Verdict: phi is not Pi-decomposable")
        _logger.info("=" * 51)
        context.print_stats()
        print("Time: %lf" % vardec_time)
        return

    print("=== [Result] ===")
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

    _logger.info("Variable decomposition:\n%s", decomposition)

    decomposition = z3.simplify(decomposition, elim_sign_ext=False, local_ctx=True)

    _logger.info("Variable decomposition simplified:\n%s", decomposition)

    context.print_stats()
    print("Time: %lf" % vardec_time)


if __name__ == '__main__':
    _main()
