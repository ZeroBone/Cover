import logging

from partition import Partition
from vardec_binary import vardec_binary
from vardec_context import VarDecContext

_logger = logging.getLogger("vardec")


class VarDecResult:

    def __init__(self, is_decomposable: bool, decomposition=None, /):
        self.is_decomposable = is_decomposable
        self.decomposition = decomposition


def vardec(phi, pi: Partition) -> VarDecResult:

    if pi.is_unary():
        return VarDecResult(True, phi)

    eq_binary_partitions = pi.get_equivalent_list_of_binary_partitions()

    _logger.info("List of partitions equivalent to Pi:\n%s", eq_binary_partitions)

    assert len(eq_binary_partitions) > 0

    if len(eq_binary_partitions) == 1:
        x, y = eq_binary_partitions[0]
        context = VarDecContext(x, y)
        result = vardec_binary(phi, context)

        if result is None:
            # the formula is not decomposable
            return VarDecResult(False)

        return VarDecResult(True, result)

    for x, y in eq_binary_partitions:
        context = VarDecContext(x, y)
        result = vardec_binary(phi, context)

        if result is None:
            # the formula is not decomposable
            return VarDecResult(False)

    return VarDecResult(True)
