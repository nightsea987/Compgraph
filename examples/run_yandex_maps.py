from itertools import islice, cycle
import click
from compgraph.algorithms import yandex_maps_graph
import typing as tp
import json


@click.command()
@click.argument('input_stream_name_length', type=click.File())
@click.argument('input_stream_name_time', type=click.File())
@click.argument('output_file', type=click.File(mode='w'))
def main(input_stream_name_length: tp.Any, input_stream_name_time: tp.Any, output_file: tp.Any) -> None:
    times = json.loads(input_stream_name_time.read().replace("\'", '\"'))
    lengths = json.loads(input_stream_name_length.read().replace("\'", '\"'))
    graph = yandex_maps_graph('travel_time', 'edge_length',
                              enter_time_column='enter_time', leave_time_column='leave_time', edge_id_column='edge_id',
                              start_coord_column='start', end_coord_column='end',
                              weekday_result_column='weekday', hour_result_column='hour', speed_result_column='speed')
    result = graph.run(travel_time=lambda: islice(cycle(iter(times)), len(times)), edge_length=lambda: iter(lengths))
    output_file.write(list(result).__repr__())


if __name__ == '__main__':
    main()
