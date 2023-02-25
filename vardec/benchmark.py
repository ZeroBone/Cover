import logging
import os
from pathlib import Path

# noinspection PyPackageRequirements
import z3 as z3

from mondec_veanes import run_veanes_benchmark

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

        phi = z3.And([f for f in smt])

        _logger.info("Testing on formula '%s'", smt_path)

        perf, size = run_veanes_benchmark(phi)

        print(smt_path, perf, size)


if __name__ == "__main__":

    fh = logging.FileHandler("benchmark.log", "w")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s %(levelname)7s]: %(message)s")
    fh.setFormatter(formatter)

    _logger.addHandler(fh)

    _run_benchmarks()
