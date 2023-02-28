import argparse
import logging
import math
import os
import sys
import time
from operator import itemgetter
from pathlib import Path
from typing import Tuple

# noinspection PyPackageRequirements
import z3 as z3

from mondec_veanes import run_veanes_benchmark
from partition import get_singleton_partition
from vardec import vardec, VarDecResult
from z3_utils import get_formula_variables, get_formula_ast_node_count

_logger = logging.getLogger("benchmark")
_logger.setLevel(logging.DEBUG)


def _resolve_benchmark_root():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark/data").resolve()


def _resolve_benchmark_results_root():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark/results").resolve()


def benchmark_smts():
    for root, dirs, files in os.walk(_resolve_benchmark_root()):
        for file in files:
            full_file_path = os.path.join(root, file)

            if not os.path.isfile(full_file_path):
                _logger.warning("File '%s' could not be found, was it deleted?", full_file_path)
                continue

            if not full_file_path.endswith(".smt2"):
                _logger.warning("Cannot handle file '%s' due to unknown extention.", full_file_path)
                continue

            try:
                smt = z3.parse_smt2_file(full_file_path)
            except z3.Z3Exception:
                _logger.warning("Could not parse benchmark file '%s'", full_file_path)
                continue

            phi = z3.simplify(z3.And([f for f in smt]))

            yield phi, full_file_path, file


class BenchmarkFormulaClass:

    def __init__(self, class_name, /):
        self._class_name = class_name
        self._key_ordering = None
        self._fh = None

    def add_result(self, prop_name_value):

        if self._key_ordering is None:
            self._key_ordering = sorted(prop_name_value.keys())
            os.makedirs(_resolve_benchmark_results_root(), exist_ok=True)
            self._fh = open(os.path.join(_resolve_benchmark_results_root(), "%s.dat" % self._class_name), "w")
            # write the topmost row of the file
            self._fh.write("    ".join("%21s" % k for k in self._key_ordering))

        assert len(prop_name_value.keys()) == len(self._key_ordering)
        assert self._fh is not None

        self._fh.write("\n")
        self._fh.write("    ".join("%21d" % prop_name_value[k] for k in self._key_ordering))

    def export(self):
        if self._fh is not None:
            self._fh.close()


def _run_presvardec_benchmark(phi, /, *, use_heuristics: bool = True) -> Tuple[float, VarDecResult]:
    _time_start = time.perf_counter()
    phi_vars = [var.unwrap() for var in get_formula_variables(phi)]
    pi = get_singleton_partition(phi_vars)
    result = vardec(phi, pi, use_heuristics=use_heuristics, use_blast=use_heuristics)
    _time_end = time.perf_counter()
    return (_time_end - _time_start), result


def _run_benchmarks(class_name: str = None):

    benchmark_instances = sorted(benchmark_smts(), key=itemgetter(1))

    # _logger.info("Found following instances:\n%s", "\n".join(v[1] for v in benchmark_instances))

    formula_classes = {}

    _logger.info("Benchmark started.")

    for phi, smt_path, smt_file in benchmark_instances:

        smt_file_components = os.path.splitext(smt_file)[0].split("_")

        if len(smt_file_components) < 2:
            _logger.warning("Ignoring the file '%s' due to invalid naming.", smt_path)
            continue

        if class_name is not None and smt_file_components[0] != class_name:
            continue

        if smt_file_components[0] in formula_classes:
            phi_class = formula_classes[smt_file_components[0]]
        else:
            phi_class = BenchmarkFormulaClass(smt_file_components[0])
            formula_classes[smt_file_components[0]] = phi_class

        prop_name_value = {
            "".join([c for c in s if not c.isdigit()]): int("".join([c for c in s if c.isdigit()]))
            for s in smt_file_components[1:]
        }

        # run the algorithm by Veanes et al.
        veanes_perf, veanes_result = run_veanes_benchmark(phi)

        # run our algorithm
        presvardec_perf, presvardec_result = _run_presvardec_benchmark(phi)

        presvardec_perf_noheuristics, presvardec_result_noheuristics =\
            _run_presvardec_benchmark(phi, use_heuristics=False)

        # analyze what the Veanes et al. algorithm provided

        veanes_size = get_formula_ast_node_count(veanes_result)

        # analyze what our algorithm provided

        assert presvardec_result.is_decomposable
        assert presvardec_result_noheuristics.is_decomposable

        if presvardec_result.decomposition is None:
            presvardec_size = 0
        else:
            presvardec_size = get_formula_ast_node_count(presvardec_result.decomposition)

        if presvardec_result_noheuristics.decomposition is None:
            presvardec_size_noheuristics = 0
        else:
            presvardec_size_noheuristics = get_formula_ast_node_count(presvardec_result_noheuristics.decomposition)

        _logger.info(
            "(Performance, Size): Veanes et al.: (%lf, %8d) PresVarDec: (%lf, %8d) "
            "PresVarDec (nh): (%lf, %8d) Formula: '%s'",
            veanes_perf,
            veanes_size,
            presvardec_perf,
            presvardec_size,
            presvardec_perf_noheuristics,
            presvardec_size_noheuristics,
            smt_file
        )

        prop_name_value["veanes_perf_s"] = math.ceil(veanes_perf)
        prop_name_value["veanes_perf_ms"] = math.ceil(veanes_perf * 1000)
        prop_name_value["veanes_size"] = veanes_size
        prop_name_value["presvardec_perf_s"] = math.ceil(presvardec_perf)
        prop_name_value["presvardec_perf_ms"] = math.ceil(presvardec_perf * 1000)
        prop_name_value["presvardec_nh_perf_s"] = math.ceil(presvardec_perf_noheuristics)
        prop_name_value["presvardec_nh_perf_ms"] = math.ceil(presvardec_perf_noheuristics * 1000)
        prop_name_value["presvardec_size"] = presvardec_size
        prop_name_value["presvardec_nh_size"] = presvardec_size_noheuristics

        phi_class.add_result(prop_name_value)

    for class_name in formula_classes:
        formula_classes[class_name].export()

    _logger.info("Benchmarking complete!")


def _main():
    # setup file handler
    fh = logging.FileHandler(os.path.join(_resolve_benchmark_results_root(), "benchmark.log"), "w")
    fh.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter("[%(asctime)s %(levelname)7s]: %(message)s")
    fh.setFormatter(file_formatter)

    _logger.addHandler(fh)

    # set up the handler responsible for console logging

    stdout_fh = logging.StreamHandler(sys.stdout)
    stdout_fh.setLevel(logging.DEBUG)

    stdout_formatter = logging.Formatter("%(levelname)7s: %(message)s")
    stdout_fh.setFormatter(stdout_formatter)

    _logger.addHandler(stdout_fh)

    parser = argparse.ArgumentParser(
        prog="presvardec_benchmark",
        description="PresVarDec's benchmarking tool",
        epilog="See the GitHub repository README for more information")

    parser.add_argument("-name", "--name", metavar="NAME",
                        help="name of the benchmark instance class",
                        type=str)

    args = parser.parse_args()

    if args.name is None:
        _logger.info("No class name specified. Benchmark will consider all formulae available.")
        class_name = None
    else:
        class_name = args.name
        _logger.info("Class name: '%s'", class_name)

    _run_benchmarks(class_name)


if __name__ == "__main__":
    _main()
