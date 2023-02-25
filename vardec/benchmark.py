import logging
import os
import time
from ctypes import Union
from pathlib import Path

# noinspection PyPackageRequirements
import z3 as z3

from mondec_veanes import run_veanes_benchmark
from partition import get_singleton_partition
from vardec import vardec
from z3_utils import get_formula_variables

_logger = logging.getLogger("benchmark")
_logger.setLevel(logging.DEBUG)


def _resolve_benchmark_root():
    base_path = Path(__file__).parent
    return (base_path / "../benchmark/data").resolve()


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
                yield z3.parse_smt2_file(full_file_path), full_file_path
            except z3.Z3Exception:
                _logger.warning("Could not parse benchmark file '%s'", full_file_path)


def _run_benchmarks():

    _logger.info("Benchmark started.")

    for smt, smt_path in benchmark_smts():

        phi = z3.simplify(z3.And([f for f in smt]))

        _logger.info("Testing on formula '%s'", smt_path)

        # run the algorithm by Veanes et al.

        veanes_perf, veanes_size = run_veanes_benchmark(phi)

        # run our algorithm
        _time_start = time.perf_counter()
        phi_vars = [var.unwrap() for var in get_formula_variables(phi)]
        pi = get_singleton_partition(phi_vars)
        result = vardec(phi, pi)
        _time_end = time.perf_counter()

        presvardec_perf = _time_end - _time_start

        assert result.is_decomposable

        presvardec_decomposition: Union[z3.ExprRef, None] = result.decomposition
        presvardec_size = 0 if presvardec_decomposition is None else len(presvardec_decomposition.sexpr())

        _logger.info(
            "[Performance & Size]: Veanes et al.: (%lf, %8d) PresVarDec: (%lf, %8d) Formula: '%s'",
            veanes_perf,
            veanes_size,
            presvardec_perf,
            presvardec_size,
            smt_path
        )


if __name__ == "__main__":

    fh = logging.FileHandler("benchmark.log", "w")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s %(levelname)7s]: %(message)s")
    fh.setFormatter(formatter)

    _logger.addHandler(fh)

    _run_benchmarks()
