from typing import List, Callable


class PartitionException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def _build_binary_partition(blocks, index_pred: Callable[[int], bool]) -> tuple:
    first = set()
    second = set()

    for i, v in enumerate(blocks):
        if index_pred(i):
            first.update(v)
        else:
            second.update(v)

    return sorted(first, key=str), sorted(second, key=str)


class Partition:

    def __init__(self, blocks: list, /):
        self._blocks = blocks

    def is_binary_or_unary(self) -> bool:
        assert len(self._blocks) > 0
        return len(self._blocks) < 3

    def get_blocks_as_variable_lists(self) -> list:
        return [sorted(b, key=str) for b in self._blocks]

    def get_equivalent_list_of_binary_partitions(self) -> List[tuple]:
        # we assume the partition is not unary
        assert len(self._blocks) > 1

        var_count = sum(len(b) for b in self._blocks)

        q = (var_count - 1).bit_length()

        return [
            _build_binary_partition(self._blocks, lambda j: (j >> i) & 1 == 0)
            for i in range(q)
        ]

    def __str__(self):
        return "{%s}" % ", ".join(
            "{%s}" % ", ".join(sorted(v.decl().name() for v in b)) for b in self._blocks
        )


def get_singleton_partition(formula_vars: list) -> Partition:
    return Partition([
        {v} for v in formula_vars
    ])


def parse_formula_variable_partition(formula_vars: list, partition_matrix: List[List[str]]) -> Partition:

    vars_table = {}

    for v in formula_vars:
        vars_table[v.decl().name()] = v

    vars_seen_so_far = set()

    partition_blocks = []

    for i, row in enumerate(partition_matrix):
        row_set = set(row)
        if not row_set.isdisjoint(vars_seen_so_far):
            # this is not a valid partition
            raise PartitionException(
                "The %d-th block of the partition contains a variable from another block of the partition." % (i + 1)
            )

        vars_seen_so_far.update(row_set)

        block_z3_vars = set()

        for var_label in row_set:
            if var_label in vars_table:
                block_z3_vars.add(vars_table[var_label])
            else:
                # unknown variable
                raise PartitionException("Label '%s' doesn't correspond to any variable of the formula." % var_label)

        partition_blocks.append(block_z3_vars)

    missing_vars = set(fw.decl().name() for fw in formula_vars).difference(vars_seen_so_far)

    if len(missing_vars) > 0:
        # some variable is missing
        raise PartitionException("The variable '%s' is not present in the partition." % next(iter(missing_vars)))

    return Partition(partition_blocks)
