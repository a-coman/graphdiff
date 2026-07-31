#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run GraphDiff for every experiment contained in a dataset directory."""

from __future__ import print_function

import argparse
import os
import re
import sys

import SoilToGraph


GEN_DIRECTORY = re.compile(r"^gen([1-9][0-9]*)$")


def generation_directories(path):
    """Return (generation_number, directory_name) pairs in numeric order."""
    generations = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        match = GEN_DIRECTORY.match(item)
        if match and os.path.isdir(item_path):
            generations.append((int(match.group(1)), item))
    return sorted(generations)


def find_experiments(dataset_path):
    """Find directories that directly contain one or more gen<number> folders."""
    experiments = []
    for root, directories, _ in os.walk(dataset_path):
        if any(GEN_DIRECTORY.match(directory) for directory in directories):
            experiments.append(root)
            # Generation directories contain inputs, not nested experiments.
            directories[:] = [
                directory
                for directory in directories
                if not GEN_DIRECTORY.match(directory)
            ]
    return sorted(experiments)


def is_cot_experiment(path):
    return "CoT" in os.path.normpath(path).split(os.sep)


def should_process(filename, cot_experiment):
    if not filename.endswith(".soil"):
        return False
    if cot_experiment and (
        filename.startswith("output") or filename.startswith("temp")
    ):
        return False
    return True


def read_file(path):
    with open(path, "r") as source:
        return source.read()


def save_file(content, path):
    with open(path, "w") as destination:
        destination.write(content)


def array_to_markdown_table(matrix, names):
    display_names = [name.replace(".soil", "") for name in names]
    header = "|       | " + " | ".join(display_names) + " |"
    separator = "|-------|" + "|".join(["---"] * len(display_names)) + "|"
    rows = []

    for row_index, row in enumerate(matrix):
        cells = ["**{0}**".format(display_names[row_index])]
        for column_index, value in enumerate(row):
            if column_index >= row_index:
                cells.append("{0:.6f}".format(value))
            else:
                cells.append("")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + rows)


def process_experiment(experiment_path, output_name):
    # Import here so discovery and --dry-run work before dependencies are installed.
    from grakel import Graph
    from grakel.kernels import ShortestPath

    graphs = []
    graph_names = []
    result = ["# Adj, edge, label\n```\n"]
    cot_experiment = is_cot_experiment(experiment_path)

    for generation_number, generation_name in generation_directories(experiment_path):
        generation_path = os.path.join(experiment_path, generation_name)
        for filename in sorted(os.listdir(generation_path)):
            if not should_process(filename, cot_experiment):
                continue

            input_path = os.path.join(generation_path, filename)
            if not os.path.isfile(input_path):
                continue

            relative_name = os.path.join(generation_name, filename)
            print("  Processing {0}".format(relative_name))
            adjacency, labels, edges = SoilToGraph.soilToGraph(read_file(input_path))

            result.extend(
                [
                    "Adj{0}-{1}: {2}\n\n".format(
                        generation_number, filename, adjacency
                    ),
                    "Labels{0}-{1}: {2}\n\n".format(
                        generation_number, filename, labels
                    ),
                    "Edges{0}-{1}: {2}\n\n".format(
                        generation_number, filename, edges
                    ),
                ]
            )
            graphs.append(Graph(initialization_object=adjacency, node_labels=labels))
            graph_names.append(relative_name)

    if not graphs:
        raise ValueError("no eligible .soil files found")

    kernel = ShortestPath(normalize=True).fit_transform(graphs)
    result.extend(
        [
            "```\n# Kernel:\n```\n",
            str(kernel),
            "\n```\n# Kernel 2D table:\n",
            array_to_markdown_table(kernel, graph_names),
            "\n",
        ]
    )

    output_path = os.path.join(experiment_path, output_name)
    save_file("".join(result), output_path)
    return output_path, len(graphs)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Recursively run graph-kernel comparisons for every experiment "
            "directory containing gen<number> folders."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset or individual experiment directory",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_option",
        help="Dataset or individual experiment directory",
    )
    parser.add_argument(
        "--output-name",
        default="grakel.md",
        help="Output filename created in each experiment (default: grakel.md)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered experiments without processing or writing files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_arguments(argv)
    if arguments.dataset and arguments.dataset_option:
        print("Pass the dataset either positionally or with --dataset, not both.", file=sys.stderr)
        return 2
    dataset = arguments.dataset_option or arguments.dataset
    if not dataset:
        print("A dataset directory is required. Pass PATH or --dataset PATH.", file=sys.stderr)
        return 2

    dataset_path = os.path.abspath(os.path.expanduser(dataset))

    if not os.path.isdir(dataset_path):
        print("Dataset directory does not exist: {0}".format(dataset_path), file=sys.stderr)
        return 2

    experiments = find_experiments(dataset_path)
    if not experiments:
        print("No experiment directories containing gen<number> folders were found.", file=sys.stderr)
        return 1

    print("Found {0} experiment(s) under {1}".format(len(experiments), dataset_path))
    if arguments.dry_run:
        for experiment_path in experiments:
            print(experiment_path)
        return 0

    failures = []
    for index, experiment_path in enumerate(experiments, start=1):
        print("[{0}/{1}] {2}".format(index, len(experiments), experiment_path))
        try:
            output_path, graph_count = process_experiment(
                experiment_path, arguments.output_name
            )
            print("  Wrote {0} graphs to {1}".format(graph_count, output_path))
        except Exception as error:
            failures.append((experiment_path, error))
            print("  FAILED: {0}".format(error), file=sys.stderr)

    if failures:
        print("\n{0} experiment(s) failed:".format(len(failures)), file=sys.stderr)
        for experiment_path, error in failures:
            print("- {0}: {1}".format(experiment_path, error), file=sys.stderr)
        return 1

    print("Completed all {0} experiment(s).".format(len(experiments)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
