import logging

from partition import Partition
from vardec_binary import vardec_binary

_logger = logging.getLogger("vardec")


def vardec(phi, pi: Partition) -> bool:

    eq_binary_partitions = pi.get_equivalent_list_of_binary_partitions()

    _logger.info("List of partitions equivalent to Pi:\n%s", eq_binary_partitions)

    for x, y in eq_binary_partitions:
        result = vardec_binary(phi, x, y)

        if result is None:
            # the formula is not decomposable
            return False

    return True
