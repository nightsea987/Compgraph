import click
from compgraph.algorithms import inverted_index_graph
import typing as tp
import json


@click.command()
@click.argument('input_file', type=click.File())
@click.argument('output_file', type=click.File(mode='w'))
def main(input_file: tp.Any, output_file: tp.Any) -> None:
    ls = json.loads(input_file.read().replace("\'", '\"'))
    graph = inverted_index_graph("docs")

    result = graph.run(docs=lambda: iter(ls))
    output_file.write(list(result).__repr__())


if __name__ == '__main__':
    main()
