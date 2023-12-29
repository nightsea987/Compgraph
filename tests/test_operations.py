import copy
import dataclasses
import typing as tp
import pytest
from pytest import approx

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


MY_MAP_CASES = [
    MapCase(
        mapper=ops.RoadTime(),
        data=[
            {'leave_time': '20231128T190807.417100', 'enter_time': '20231128T183807.417100'},
            {'leave_time': '19991128T230807.300000', 'enter_time': '19991128T210807.300000'}
        ],
        ground_truth=[
            {'leave_time': '20231128T190807.417100', 'enter_time': '20231128T183807.417100', 'road_time': 1800},
            {'leave_time': '19991128T230807.300000', 'enter_time': '19991128T210807.300000', 'road_time': 7200}
        ],
        cmp_keys=("leave_time", "enter_time", "road_time"),
    ),
    MapCase(
        mapper=ops.HaversineDistance(),
        data=[
            {'start': [37.61729811111, 55.75582511111], 'end': [37.62729811111, 55.75682511111]},
            {'start': [37.61729811111, 55.75582511111], 'end': [37.61729811111, 55.75582511111]}
        ],
        ground_truth=[
            {'distance': approx(0.635711, abs=0.0001),
             'end': [37.62729811111, 55.75682511111],
             'start': [37.61729811111, 55.75582511111]},
            {'distance': 0.0,
             'end': [37.61729811111, 55.75582511111],
             'start': [37.61729811111, 55.75582511111]
             }
        ],
        cmp_keys=('distance', 'start', 'end')
    ),
    MapCase(
        mapper=ops.Speed(),
        data=[{'distance': 4, 'time': 360000}, {'distance': 1000, 'time': 72000}],
        ground_truth=[{'distance': 4, 'time': 360000, 'speed': 0.04}, {'distance': 1000, 'time': 72000, 'speed': 50}],
        cmp_keys=("distance", "time", "speed")
    ),
    MapCase(
        mapper=ops.InverseDocumentFrequency(total_docs_column='total_docs', docs_column='word_i'),
        data=[
            {'total_docs': 1000, 'word_i': 100},
            {'total_docs': 1000, 'word_i': 10},
            {'total_docs': 1000, 'word_i': 1}
        ],
        ground_truth=[
            {'total_docs': 1000, 'word_i': 100, 'idf': approx(2.302585, abs=0.0001)},
            {'total_docs': 1000, 'word_i': 10, 'idf': approx(4.605170, abs=0.0001)},
            {'total_docs': 1000, 'word_i': 1, 'idf': approx(6.907755, abs=0.0001)}
        ],
        cmp_keys=('idf',)
    ),
    MapCase(
        mapper=ops.Weekday(),
        data=[{'datetime': '20231128T191500.000000'}, {'datetime': '20231120T101500.000000'}],
        ground_truth=[
            {'weekday': 'Tue', 'datetime': '20231128T191500.000000'},
            {'weekday': 'Mon', 'datetime': '20231120T101500.000000'},
        ],
        cmp_keys=("datetime", "weekday")
    ),
    MapCase(
        mapper=ops.Hour(),
        data=[{'datetime': '20231128T191500.000000'}, {'datetime': '20231120T101500.000000'}],
        ground_truth=[
            {'hour': 19, 'datetime': '20231128T191500.000000'},
            {'hour': 10, 'datetime': '20231120T101500.000000'},
        ],
        cmp_keys=("datetime", "hour")
    )
]


@pytest.mark.parametrize('map_case', MY_MAP_CASES)
def test_my_mappers(map_case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(map_case.data[map_case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(map_case.ground_truth[i]) for i in map_case.mapper_ground_truth_items]
    key_func = _Key(*map_case.cmp_keys)

    mapper_result = map_case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_result, key=key_func) == sorted(mapper_ground_truth_rows, key=key_func)

    map_result = ops.Map(map_case.mapper)(iter(map_case.data))
    assert isinstance(map_result, tp.Iterator)
    assert sorted(map_result, key=key_func) == sorted(map_case.ground_truth, key=key_func)


@dataclasses.dataclass
class ReduceCase:
    reducer: ops.Reducer
    reducer_keys: tuple[str, ...]
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    reduce_data_items: tuple[int, ...] = (0,)
    reduce_ground_truth_items: tuple[int, ...] = (0,)


MY_REDUCE_CASES = [
    ReduceCase(
        reducer=ops.Average(column='b'),
        reducer_keys=('a',),
        data=[
            {'a': 1, 'b': 2, 'f': 4},
            {'a': 1, 'b': 8, 'f': 5},

            {'a': 2, 'b': 4, 'f': 4},
            {'a': 2, 'b': 8, 'f': 15},
            {'a': 2, 'b': 3, 'f': 7},
        ],
        ground_truth=[{'a': 1, 'b': 5}, {'a': 2, 'b': 5}],
        cmp_keys=('a', 'b'),
        reduce_data_items=(0, 1),
        reduce_ground_truth_items=(0,)
    ),
]


@pytest.mark.parametrize('reduce_case', MY_REDUCE_CASES)
def test_my_reducers(reduce_case: ReduceCase) -> None:
    reducer_data_rows = [copy.deepcopy(reduce_case.data[i]) for i in reduce_case.reduce_data_items]
    reducer_ground_truth_rows = \
        [copy.deepcopy(reduce_case.ground_truth[i]) for i in reduce_case.reduce_ground_truth_items]

    key_func = _Key(*reduce_case.cmp_keys)

    reducer_result = reduce_case.reducer(reduce_case.reducer_keys, iter(reducer_data_rows))
    assert isinstance(reducer_result, tp.Iterator)
    assert sorted(reducer_result, key=key_func) == sorted(reducer_ground_truth_rows, key=key_func)

    reduce_result = ops.Reduce(reduce_case.reducer, reduce_case.reducer_keys)(iter(reduce_case.data))
    assert isinstance(reduce_result, tp.Iterator)
    assert sorted(reduce_result, key=key_func) == sorted(reduce_case.ground_truth, key=key_func)
