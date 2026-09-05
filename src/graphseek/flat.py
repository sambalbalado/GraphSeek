"""Exact nearest-neighbor search used as the baseline for HNSW.

Exact search compares a query with every stored vector. It is simple and
reliable, which makes it useful for checking the quality of faster approximate
algorithms later in the project.
"""

from dataclasses import dataclass
from heapq import nsmallest

from numpy.typing import ArrayLike

from graphseek.metrics import Metric, Vector, prepare_vector, prepared_distance


@dataclass(frozen=True)
class Neighbor:
    """One search result containing its stable ID and distance to the query."""

    id: int
    distance: float


class FlatIndex:
    """An in-memory index that checks every vector during search.

    Vectors receive consecutive IDs starting at zero. The first inserted
    vector fixes the number of dimensions accepted by the index.
    """

    def __init__(self, metric: Metric = "l2") -> None:
        if metric not in ("l2", "cosine"):
            raise ValueError("Metric must be 'l2' or 'cosine'")
        self._metric = metric
        self._vectors: list[Vector] = []
        self._dimension: int | None = None

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return len(self._vectors)

    def _prepare(self, value: ArrayLike) -> Vector:
        """Validate one vector and enforce this index's dimensionality."""
        result = prepare_vector(value, self._metric)
        if self._dimension is not None and len(result) != self._dimension:
            raise ValueError(f"Expected {self._dimension} dimensions")
        return result

    def add(self, value: ArrayLike) -> int:
        """Copy a vector into the index and return its assigned ID."""
        item = self._prepare(value)
        self._dimension = len(item)
        item_id = len(self)
        self._vectors.append(item)
        return item_id

    def search(self, query: ArrayLike, k: int = 10) -> list[Neighbor]:
        """Return up to k neighbors, breaking equal distances by ID.

        O(nd + n log k) time and O(d + k) auxiliary space, with k capped at n.
        An empty index returns an empty list after validating the query.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        item = self._prepare(query)
        # nsmallest keeps only the best k candidates in a heap. Sorting every
        # candidate would be simpler, but would cost O(n log n) instead.
        candidates = (
            Neighbor(vector_id, prepared_distance(item, stored, self._metric))
            for vector_id, stored in enumerate(self._vectors)
        )
        return nsmallest(
            min(k, len(self)),
            candidates,
            key=lambda result: (result.distance, result.id),
        )
