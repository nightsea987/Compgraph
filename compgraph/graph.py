from __future__ import annotations

import typing as tp
from . import operations as ops
from .external_sort import ExternalSort


class Graph:
    """Computational graph implementation"""

    def __init__(self, operations: tp.List[ops.Operation], graphs_to_join: tp.List[Graph] | None = None) -> None:
        """
        :param operations: operations that graph need to do in run
        :param graphs_to_join: graphs that current graph will join with
        """
        self.operations = operations
        if graphs_to_join is None:
            self.graphs_to_join: tp.List[Graph] = []
        else:
            self.graphs_to_join = graphs_to_join

    @staticmethod
    def graph_from_iter(name: str) -> Graph:
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        operation = ops.ReadIterFactory(name)
        return Graph([operation])

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> Graph:
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        operation = ops.Read(filename, parser)
        return Graph([operation])

    def map(self, mapper: ops.Mapper) -> Graph:
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        operation = ops.Map(mapper)
        return Graph(self.operations + [operation], self.graphs_to_join)

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> Graph:
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        operation = ops.Reduce(reducer, keys)
        return Graph(self.operations + [operation], self.graphs_to_join)

    def sort(self, keys: tp.Sequence[str]) -> Graph:
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        operation = ExternalSort(keys)
        return Graph(self.operations + [operation], self.graphs_to_join)

    def join(self, joiner: ops.Joiner, join_graph: Graph, keys: tp.Sequence[str]) -> Graph:
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        operation = ops.Join(joiner, keys)
        return Graph(self.operations + [operation], self.graphs_to_join + [join_graph])

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        index_with_data, join_index = 0, 0
        passed_data = self.operations[index_with_data](**kwargs)
        for do_operation in self.operations[index_with_data + 1:]:
            if not isinstance(do_operation, ops.Join):
                passed_data = do_operation(passed_data)
            else:
                data_to_join = self.graphs_to_join[join_index].run(**kwargs)
                passed_data = do_operation(passed_data, data_to_join)
                join_index += 1
        yield from passed_data
