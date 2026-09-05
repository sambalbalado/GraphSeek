"""Create repeatable sample data and exact answers for future experiments.

Real embeddings tend to form groups of related items. This module creates a
small approximation of that shape by sampling points around random centres.
A fixed seed makes failures repeatable across runs.
"""

import argparse
import json

import numpy as np
from numpy.typing import NDArray

from graphseek.flat import FlatIndex
from graphseek.metrics import Metric


def clustered_data(
    count: int = 1000,
    queries: int = 20,
    dimensions: int = 16,
    clusters: int = 8,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return database vectors and query vectors drawn from shared clusters."""
    for name, value in (
        ("count", count),
        ("queries", queries),
        ("dimensions", dimensions),
        ("clusters", clusters),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    rng = np.random.default_rng(seed)
    # Multiplication spreads centres apart; the smaller noise below keeps
    # samples near their selected centre.
    centers = rng.normal(size=(clusters, dimensions)) * 3

    def sample(size: int) -> NDArray[np.float64]:
        labels = rng.integers(clusters, size=size)
        return centers[labels] + rng.normal(size=(size, dimensions)) * 0.3

    return sample(count), sample(queries)


def ground_truth(
    vectors: NDArray[np.float64],
    queries: NDArray[np.float64],
    k: int = 10,
    metric: Metric = "l2",
) -> list[list[int]]:
    """Return exact neighbor IDs for each query using ``FlatIndex``."""
    index = FlatIndex(metric)
    for item in vectors:
        index.add(item)
    return [[neighbor.id for neighbor in index.search(query, k)] for query in queries]


def main() -> None:
    """Parse command-line options and print a small JSON ground-truth set."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("count", 1000),
        ("queries", 20),
        ("dimensions", 16),
        ("clusters", 8),
        ("seed", 42),
        ("k", 10),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    try:
        vectors, queries = clustered_data(
            args.count,
            args.queries,
            args.dimensions,
            args.clusters,
            args.seed,
        )
        neighbors = ground_truth(vectors, queries, args.k)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"config": vars(args), "neighbors": neighbors}, indent=2))


if __name__ == "__main__":
    main()
