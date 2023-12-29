import typing as tp

from compgraph import operations as ops
from compgraph import Graph
from compgraph import ExternalSort


def test_graph_from_iter() -> None:
    graph = Graph.graph_from_iter('data')
    assert len(graph.operations) == 1
    assert isinstance(graph.operations[0], ops.ReadIterFactory)
    assert graph.graphs_to_join == []


def test_graph_from_file(monkeypatch: tp.Any) -> None:
    def mock_read_file(filename: tp.Any, parser: tp.Any) -> tp.Any:
        return Graph([ops.ReadIterFactory('data')])

    monkeypatch.setattr(ops, 'Read', mock_read_file)
    graph = Graph.graph_from_file('example.txt', lambda x: {'column_name': x})
    assert isinstance(graph, Graph)
    assert len(graph.operations) == 1
    assert not isinstance(graph.operations[0], ops.ReadIterFactory)
    assert graph.graphs_to_join == []


def test_map() -> None:
    graph = Graph.graph_from_iter('data').map(ops.DummyMapper()).map(ops.LowerCase('text'))
    assert isinstance(graph, Graph)
    assert len(graph.operations) == 3
    assert isinstance(graph.operations[1], ops.Map)
    assert graph.graphs_to_join == []


def test_reduce() -> None:
    keys = ['key1', 'key2']
    graph = Graph.graph_from_iter('data').reduce(ops.TermFrequency('words'), keys)
    assert len(graph.operations) == 2
    assert isinstance(graph.operations[1], ops.Reduce)
    assert graph.graphs_to_join == []


def test_sort() -> None:
    keys = ['key1', 'key2']
    graph = Graph.graph_from_iter('data').sort(keys)
    assert isinstance(graph, Graph)
    assert len(graph.operations) == 2
    assert isinstance(graph.operations[1], ExternalSort)
    assert graph.graphs_to_join == []


def test_join() -> None:
    keys = ['key1', 'key2']
    other_graph = Graph.graph_from_iter('data2')
    graph = Graph.graph_from_iter('data1').join(ops.RightJoiner(), other_graph, keys)
    assert isinstance(graph, Graph)
    assert len(graph.operations) == 2
    assert isinstance(graph.operations[1], ops.Join)
    assert graph.graphs_to_join == [other_graph]


def test_run() -> None:
    def read_data() -> tp.Any:
        return [{'column1': 1, 'column2': 2}, {'column1': 3, 'column2': 4}]

    graph = Graph([ops.ReadIterFactory('data')])
    result = list(graph.run(data=read_data))
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert 'column1' in result[0]


def test_graph_split_filter_text() -> None:
    docs = [{'id': 10, 'text': 'NO GOD! plEasE! NOOOOOOOoo'}]

    ground_truth = [
        {'id': 10, 'text': 'no'},
        {'id': 10, 'text': 'god'},
        {'id': 10, 'text': 'please'},
        {'id': 10, 'text': 'nooooooooo'},
    ]

    graph = (Graph.graph_from_iter('docs')
             .map(ops.FilterPunctuation('text'))
             .map(ops.LowerCase('text'))
             .map(ops.Split('text')))

    assert list(graph.run(docs=lambda: iter(docs))) == ground_truth
