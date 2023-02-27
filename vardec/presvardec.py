import argparse
import logging
import os
import sys
import time
from pathlib import Path
from inspect import isfunction, getmodule

import numpy as np
# noinspection PyPackageRequirements
import z3

import examples
from partition import parse_formula_variable_partition, PartitionException, get_singleton_partition, Partition
from vardec import vardec
from vardec_binary import vardec_binary
from vardec_context import VarDecContext
from visualizer import Visualizer
from z3_utils import is_valid, get_formula_variables

_logger = logging.getLogger("vardec")


def _resolve_base_path():
    base_path = Path(__file__).parent
    return (base_path / "../").resolve()


def _load_formula_from_repl() -> tuple:

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
        sys.exit(0)
    else:
        formula, partition_blocks = example_formulas[formula_id][1]()
        return formula, Partition(partition_blocks)


def _load_formula_from_smt2(args):

    try:
        formula_smt = z3.parse_smt2_file(args.formula.name)
    except z3.Z3Exception:
        print("Error: could not parse the specified .smt2 file due to a Z3 error.")
        sys.exit(0)

    return z3.simplify(z3.And([f for f in formula_smt]))


def _main():
    parser = argparse.ArgumentParser(
        prog="presvardec",
        description="Tool for deciding monadic and variable decomposition of linear real arithmetic",
        epilog="See the GitHub repository README for more information")

    parser.add_argument("-f", "--formula", metavar="FILE",
                        help="path to the .smt2 file containing the formula",
                        type=argparse.FileType("r", encoding="UTF-8"))

    parser.add_argument("-p", "--pi", action="append", dest="pi", nargs="+",
                        help="specify the variables of a block of the partition")

    parser.add_argument("-v", "--verbose", action="store_true", help="enable logging of all debug information")
    parser.add_argument("-d", "--debug", action="store_true", help="enable expensive invariant checks")
    parser.add_argument("-i", "--vis", action="store_true", help="turn on algorithm visualization mode")
    parser.add_argument("-s", "--no-heuristics", action="store_true", help="disable the usage of heuristics")
    parser.add_argument("-b", "--no-blast", action="store_true", help="disable the usage of the blast heuristic")

    args = parser.parse_args()

    use_heuristics = not args.no_heuristics
    use_blast_heuristic = not args.no_blast

    # initialize the logger

    _logger.setLevel(logging.DEBUG if args.verbose else logging.ERROR)

    file_handler = logging.FileHandler(os.path.join(_resolve_base_path(), "vardec.log"), mode="w")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s %(levelname)7s]: %(message)s")
    file_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)

    _logger.info("Verbose mode: %s", "enabled" if args.verbose else "disabled")
    _logger.info("Debug mode: %s", "enabled" if args.debug else "disabled")
    _logger.info("Visualization mode: %s", "enabled" if args.vis else "disabled")
    _logger.info("Using heuristics: %s", "yes" if use_heuristics else "no")
    _logger.info("Using the blast heuristic: %s", "yes" if use_blast_heuristic else "no")

    # config numpy

    np.set_printoptions(formatter={"object": lambda _s: "%9s" % _s})

    # load the formula

    if args.formula is not None:
        phi = _load_formula_from_smt2(args)
        pi = None
    else:
        phi, pi = _load_formula_from_repl()

    phi_vars = [var.unwrap() for var in get_formula_variables(phi)]

    # did the user specify a partition?
    # if yes, then use it
    if args.pi is not None:
        try:
            pi = parse_formula_variable_partition(phi_vars, args.pi)
        except PartitionException as e:
            print("Partition is not specified correctly: %s" % e)
            return

    if pi is None:
        # still the partition is missing
        # fall back to the singleton partition
        pi = get_singleton_partition(phi_vars)

    assert isinstance(pi, Partition)

    _logger.info("Partition: Pi = %s", pi)
    print("Partition: Pi = %s" % pi)

    if pi.is_unary():
        # decomposing over a unary partition is trivial
        _time_start = time.perf_counter()
        context = None
        decomposition = phi
        is_decomposable = True
        vardec_time = time.perf_counter() - _time_start
    elif pi.is_binary():

        _blocks = pi.get_blocks_as_variable_lists()

        context = VarDecContext(
            _blocks[0],
            _blocks[1],
            debug_mode=args.debug,
            use_heuristics=use_heuristics,
            use_blast_heuristic=use_blast_heuristic
        )

        visualizer = Visualizer() if args.vis else None

        _time_start = time.perf_counter()

        decomposition = vardec_binary(
            phi,
            context,
            visualizer=visualizer
        )

        vardec_time = time.perf_counter() - _time_start

        is_decomposable = decomposition is not None

    else:
        _time_start = time.perf_counter()
        _result = vardec(phi, pi)
        vardec_time = time.perf_counter() - _time_start
        is_decomposable = _result.is_decomposable
        decomposition = _result.decomposition
        context = None

    _logger.info(("=" * 20) + " [RESULT] " + ("=" * 20))

    if not is_decomposable:
        print("=== [Result] ===")
        print("Verdict: phi is not Pi-decomposable (see logs for the details)")
        _logger.info("Verdict: phi is not Pi-decomposable")
        _logger.info("=" * 51)
        if context is not None:
            context.print_stats()
        print("Time: %lf" % vardec_time)
        return

    print("=== [Result] ===")
    print("Verdict: phi is Pi-decomposable")
    _logger.info("Verdict: phi is Pi-decomposable")

    if decomposition is not None:
        if is_valid(phi == decomposition):
            s = "Test PASS: decomposition is equivalent to the original formula"
            _logger.info(s)
            print(s)
        else:
            s = "Test FAIL: decomposition is not equivalent to the original formula"
            _logger.error(s)
            print(s)

    _logger.info("=" * 51)

    if decomposition is not None:
        _logger.info("Variable decomposition:\n%s", decomposition)
        decomposition = z3.simplify(decomposition, elim_sign_ext=False, local_ctx=True)
        _logger.info("Variable decomposition simplified:\n%s", decomposition)

    if context is not None:
        context.print_stats()

    print("Time: %lf" % vardec_time)


if __name__ == '__main__':
    _main()
