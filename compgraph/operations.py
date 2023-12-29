from abc import abstractmethod, ABC
from collections import defaultdict
from copy import deepcopy
from heapq import nlargest
from itertools import groupby
from math import log, radians, asin, sin, pow, sqrt, cos
import datetime
import string
import typing as tp
import re

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""
    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            for mapped_row in self.mapper(row):
                yield mapped_row


class Reducer(ABC):
    """Base class for reducers"""
    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for reduce_key, reduce_group in groupby(rows, key=lambda key_elem: tuple(key_elem[key] for key in self.keys)):
            for reduced_row in self.reducer(tuple(self.keys), reduce_group):
                yield reduced_row


class Joiner(ABC):
    """Base class for joiners"""
    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass

    def join(self, keys: tp.Sequence[str], row_a: TRow, row_b: TRow) -> TRow:
        non_key_columns = set(row_a.keys()).intersection(row_b.keys()) - set(keys)

        def merge_common_columns(row: TRow, suffix: str) -> TRow:
            return {key + suffix: row[key] for key in non_key_columns}

        def merge_non_key_columns(row: TRow) -> TRow:
            return {key: row[key] for key in row.keys() if key not in non_key_columns}

        result_row = {}
        result_row.update(merge_non_key_columns(row_a))
        result_row.update(merge_non_key_columns(row_b))
        result_row.update(merge_common_columns(row_a, self._a_suffix))
        result_row.update(merge_common_columns(row_b, self._b_suffix))
        return result_row


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        first_grouper = groupby(rows, key=lambda row: tuple(row[k] for k in self.keys))
        second_grouper = groupby(args[0], key=lambda row: tuple(row[k] for k in self.keys))
        temp_first_group = next(first_grouper, (None, None))
        first_key, first_group = temp_first_group
        temp_second_group = next(second_grouper, (None, None))
        second_key, second_group = temp_second_group

        while first_key is not None or second_key is not None:
            if first_group is not None and first_key is not None and (second_key is None or first_key < second_key):
                for r in self.joiner(keys=self.keys, rows_a=first_group, rows_b=[{}]):
                    yield r
                first_key, first_group = next(first_grouper, (None, None))
            elif second_key is not None and second_group is not None and (first_key is None or second_key < first_key):
                for r in self.joiner(self.keys, [{}], second_group):
                    yield r
                second_key, second_group = next(second_grouper, (None, None))
            else:
                if first_group is not None and second_group is not None:
                    for r in self.joiner(self.keys, first_group, second_group):
                        yield r
                    first_key, first_group = next(first_grouper, (None, None))
                    second_key, second_group = next(second_grouper, (None, None))


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""
    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        copied_row: TRow = deepcopy(row)
        copied_row[self.column] = ''.join([char for char in row[self.column] if char not in string.punctuation])
        yield copied_row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        copied_row: TRow = deepcopy(row)
        copied_row[self.column] = self._lower_case(row[self.column])
        yield copied_row


class HaversineDistance(Mapper):
    """Calculate the great-circle distance between two points on a sphere given their longitudes and latitudes."""

    def __init__(self, start_coords: str = 'start', end_coords: str = 'end', result_column: str = 'distance') -> None:
        """
        :param start_coords: start coordinates in lon/lat format
        :param end_coords: end coordinate in lon/lat format
        :param result_column: name for result column
        """
        self.R = 6373.0
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.result_column = result_column

    @staticmethod
    def haversine(theta: float) -> float:
        return pow(sin(theta / 2), 2)

    def __call__(self, row: TRow) -> TRowsGenerator:
        start_lon, start_lat = [radians(start_position) for start_position in row[self.start_coords]]
        end_lon, end_lat = [radians(end_position) for end_position in row[self.end_coords]]

        archaversine = asin(sqrt(self.haversine(end_lat - start_lat) +
                                 cos(start_lat) * cos(end_lat) * self.haversine(end_lon - start_lon)))

        copied_row: TRow = deepcopy(row)
        copied_row[self.result_column] = 2 * self.R * archaversine
        yield copied_row


class RoadTime(Mapper):
    """Calculates the time on the road from enter time to leave time"""
    def __init__(
            self,
            enter_time: str = 'enter_time',
            leave_time: str = 'leave_time',
            result_column: str = 'road_time'
    ) -> None:
        """
        :param enter_time: name for column with enter time
        :param leave_time: name for column with leave time
        :param result_column: name for result column
        """
        self.enter_time = enter_time
        self.leave_time = leave_time
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        formatted_enter_time = datetime.datetime.strptime(row[self.enter_time], "%Y%m%dT%H%M%S.%f")
        formatted_leave_time = datetime.datetime.strptime(row[self.leave_time], "%Y%m%dT%H%M%S.%f")

        copied_row: TRow = deepcopy(row)
        copied_row[self.result_column] = (formatted_leave_time - formatted_enter_time).total_seconds()
        yield copied_row


class Weekday(Mapper):
    """Calculates weekday of datetime"""
    def __init__(self, datetime_column: str = 'datetime', result_column: str = 'weekday') -> None:
        """
        :param datetime_column: name of column with datetime
        :param result_column: name of result column
        """
        self.datetime_column = datetime_column
        self.result_column = result_column

    @staticmethod
    def get_weekday(row_time: tp.Any) -> str:
        all_weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return all_weekdays[datetime.datetime.strptime(row_time, "%Y%m%dT%H%M%S.%f").weekday()]

    def __call__(self, row: TRow) -> TRowsGenerator:
        copied_row: TRow = deepcopy(row)
        copied_row[self.result_column] = self.get_weekday(row[self.datetime_column])
        yield copied_row


class Hour(Mapper):
    """Calculates hour of datetime"""
    def __init__(self, datetime_column: str = 'datetime', result_column: str = 'hour') -> None:
        """
        :param datetime_column: name of column with datetime
        :param result_column: name of result column
        """
        self.datetime_column = datetime_column
        self.result_column = result_column

    @staticmethod
    def get_hour(row_time: tp.Any) -> int:
        return datetime.datetime.strptime(row_time, "%Y%m%dT%H%M%S.%f").hour

    def __call__(self, row: TRow) -> TRowsGenerator:
        copied_row: TRow = deepcopy(row)
        copied_row[self.result_column] = self.get_hour(row[self.datetime_column])
        yield copied_row


class Speed(Mapper):
    """Calculates the speed by time and distance in kilometers per hour"""
    def __init__(self, distance: str = 'distance', time: str = 'time', result_column: str = 'speed') -> None:
        """
        :param distance: name for column with distance in meters
        :param time: name for column with time in seconds
        :param result_column: name for result column with speed
        """
        self.distance = distance
        self.time = time
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        from_meters_per_second = 3600
        copied_row: TRow = deepcopy(row)
        calculated_speed = row[self.distance] / row[self.time] * from_meters_per_second
        copied_row[self.result_column] = calculated_speed
        yield copied_row


class InverseDocumentFrequency(Mapper):
    """Calculates inversion of the frequency with which a certain word occurs in the collection documents"""
    def __init__(self, total_docs_column: str, docs_column: str, result_column: str = 'idf') -> None:
        """
        :param total_docs_column: column with total docs
        :param docs_column: name where word_i is present
        :param result_column: name for result column
        """
        self.docs_column = docs_column
        self.total_docs_column = total_docs_column
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        copied_row: TRow = deepcopy(row)
        copied_row[self.result_column] = log(row[self.total_docs_column]) - log(row[self.docs_column])
        yield copied_row


class Split(Mapper):
    """Split row on multiple rows by separator"""
    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator
        self.split_row = re.compile('[^' + self.separator + ']+') if self.separator is not None else re.compile(r'\w+')

    def __call__(self, row: TRow) -> TRowsGenerator:
        flag_find = True
        for found in re.finditer(self.split_row, row[self.column]):
            dict_to_add = {self.column: found[0]}
            yield row | dict_to_add
            flag_find = False
        if flag_find:
            dict_to_add = {self.column: ''}
            yield row | dict_to_add


class Product(Mapper):
    """Calculates product of multiple columns"""
    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        copied_row: TRow = deepcopy(row)
        result_product: float = 1
        for column in self.columns:
            result_product *= row[column]
        copied_row[self.result_column] = result_product
        yield copied_row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""
    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {name: row[name] for name in self.columns}


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""
    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        n_largest = nlargest(self.n, rows, key=lambda row_elem: row_elem[self.column_max])
        for row in n_largest:
            yield row


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""
    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        grouped_data: tp.DefaultDict[tp.Tuple[tp.Any, ...], tp.DefaultDict[tp.Tuple[tp.Any, ...], int]] = (
            defaultdict(lambda: defaultdict(int)))
        group_total: tp.DefaultDict[tp.Tuple[tp.Any, ...], int] = defaultdict(int)

        for row in rows:
            key: tp.Tuple[tp.Any, ...] = tuple(row[k] for k in group_key)
            word = row[self.words_column]
            grouped_data[key][word] += 1
            group_total[key] += 1

        for key, word_counts in grouped_data.items():
            total_count = group_total[key]
            for word, count in word_counts.items():
                new_row = {k: v for k, v in zip(group_key, key)}
                new_row[self.result_column] = count / total_count
                new_row[self.words_column] = word
                yield new_row


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        grouped_data: tp.DefaultDict[tp.Tuple[tp.Any, ...], int] = defaultdict(int)

        for row in rows:
            key: tp.Tuple[tp.Any, ...] = tuple(row[k] for k in group_key)
            grouped_data[key] += 1

        for key, word_counts in grouped_data.items():
            new_row = {k: v for k, v in zip(group_key, key)}
            new_row[self.column] = grouped_data[key]
            yield new_row


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        grouped_data: tp.DefaultDict[tp.Tuple[tp.Any, ...], int] = defaultdict(int)

        for row in rows:
            key: tp.Tuple[tp.Any, ...] = tuple(row[k] for k in group_key)
            grouped_data[key] += row[self.column]

        for key, word_counts in grouped_data.items():
            new_row = {k: v for k, v in zip(group_key, key)}
            new_row[self.column] = grouped_data[key]
            yield new_row


class Average(Reducer):
    """
        Average values aggregated by key
        Example for key=('a',) and column='b'
            {'a': 1, 'b': 2, 'f': 4}
            {'a': 1, 'b': 8, 'f': 5}
            =>
            {'a': 1, 'b': 5}
        """

    def __init__(self, column: str) -> None:
        """
        :param column: name for average column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        grouped_data: tp.DefaultDict[tp.Tuple[tp.Any, ...], int] = defaultdict(int)
        group_count: tp.DefaultDict[tp.Tuple[tp.Any, ...], int] = defaultdict(int)

        for row in rows:
            key: tp.Tuple[tp.Any, ...] = tuple(row[k] for k in group_key)
            grouped_data[key] += row[self.column]
            group_count[key] += 1

        for key, word_counts in grouped_data.items():
            new_row = {k: v for k, v in zip(group_key, key)}
            new_row[self.column] = grouped_data[key] / group_count[key]
            yield new_row


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        copy_rows_b = list(rows_b)
        for first_row in rows_a:
            for second_row in copy_rows_b:
                if second_row and first_row:
                    yield self.join(keys, first_row, second_row)


class OuterJoiner(Joiner):
    """Join with outer strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        flag = True
        copy_rows_b = [row for row in rows_b]
        if len(copy_rows_b):
            for first_row in rows_a:
                for second_row in copy_rows_b:
                    yield self.join(keys, first_row, second_row)
                flag = False
        else:
            for first_row in rows_a:
                yield self.join(keys, first_row, {})
            return
        if not flag:
            return
        for second_row in copy_rows_b:
            yield self.join(keys, {}, second_row)


class LeftJoiner(Joiner):
    """Join with left strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        flag = True
        copy_rows_b = [row for row in rows_b]
        for first_row in rows_a:
            for second_row in copy_rows_b:
                if first_row:
                    yield self.join(keys, first_row, second_row)
            flag = False
        if not flag:
            return
        for second_row in copy_rows_b:
            yield self.join(keys, {}, second_row)


class RightJoiner(Joiner):
    """Join with right strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        flag = True
        copy_rows_b = [row for row in rows_b]
        for first_row in rows_a:
            for second_row in copy_rows_b:
                if second_row:
                    yield self.join(keys, first_row, second_row)
            flag = False
        if not flag:
            return
        for second_row in copy_rows_b:
            yield self.join(keys, {}, second_row)
