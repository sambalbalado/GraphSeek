"""Exact nearest-neighbor search used as the baseline for HNSW."""

from dataclasses import dataclass
from heapq import nsmallest

from numpy.typing import ArrayLike

from graphseek.metrics import Metric, Vector, distance, prepare


@dataclass(frozen=True)
class Neighbor:
    id: int
    distance: float


class FlatIndex:
    """Store copies of vectors and assign consecutive IDs starting at zero."""

    def __init__(self, metric: Metric = "l2") -> None:
        if metric not in ("l2", "cosine"):
            raise ValueError("Metric must be 'l2' or 'cosine'")
        self._metric = metric
        self._vectors: list[Vector] = []
        self._dimension: int | None = None

    def __len__(self) -> int:
        return len(self._vectors)

    def _prepare(self, value: ArrayLike) -> Vector:
        result = prepare(value, self._metric)
        if self._dimension is not None and len(result) != self._dimension:
            raise ValueError(f"Expected {self._dimension} dimensions")
        return result

    def add(self, value: ArrayLike) -> int:
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
        candidates = (
            Neighbor(i, distance(item, stored, self._metric))
            for i, stored in enumerate(self._vectors)
        )
        return nsmallest(
            min(k, len(self)),
            candidates,
            key=lambda result: (result.distance, result.id),
        )
