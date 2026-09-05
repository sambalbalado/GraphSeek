"""Validate vectors and measure how far apart they are.

GraphSeek supports two distance measures:

* ``l2`` is squared Euclidean distance. It measures straight-line distance
  without taking the final square root, which does not change result order.
* ``cosine`` measures the angle between vectors. It is useful when direction
  matters more than magnitude, as is common with embeddings.

Public functions accept Python sequences or NumPy arrays. Internal functions
work with validated float64 arrays so indexes do not repeat unnecessary work.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Metric = Literal["l2", "cosine"]
Vector = NDArray[np.float64]


def validate_vector(value: ArrayLike) -> Vector:
    """Return a safe float64 copy of a one-dimensional numeric vector.

    Copying is intentional: an index must not change if the caller later
    modifies the list or array that was passed to it.
    """
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("Expected a non-empty one-dimensional vector")
    if raw.dtype.kind not in "iuf":
        raise ValueError("Vector values must be real numbers")
    result = raw.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Vector values must be finite")
    return result


def normalize_for_cosine(value: Vector) -> Vector:
    """Return a unit-length copy suitable for cosine comparisons."""
    # Dividing very large numbers by their maximum keeps the norm calculation
    # inside float64 range. Scaling does not change the vector's direction.
    scale = float(np.max(np.abs(value)))
    if scale == 0:
        raise ValueError("Cosine distance is undefined for a zero vector")
    scaled = value / scale
    return scaled / np.linalg.norm(scaled)


def prepare_vector(value: ArrayLike, metric: Metric) -> Vector:
    """Validate a vector and apply preprocessing required by the metric."""
    result = validate_vector(value)
    return normalize_for_cosine(result) if metric == "cosine" else result


def prepared_distance(a: Vector, b: Vector, metric: Metric) -> float:
    """Compare validated vectors that were prepared for ``metric``.

    This internal fast path assumes equal dimensions. ``FlatIndex`` enforces
    that condition when vectors enter the index and when a query is prepared.
    """
    if metric == "cosine":
        return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))
    with np.errstate(over="ignore"):
        delta = a - b
        result = float(np.dot(delta, delta))
    if not np.isfinite(result):
        raise ValueError("Squared distance exceeds float64 range")
    return result


def squared_l2(a: ArrayLike, b: ArrayLike) -> float:
    """Return squared Euclidean distance between two vectors.

    For example, the distance between ``[0, 0]`` and ``[3, 4]`` is 25.
    The function uses O(d) time and temporary space.
    """
    return _compare(a, b, "l2")


def cosine_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Return cosine distance in the range [0, 2].

    Parallel vectors have distance 0, perpendicular vectors have distance 1,
    and opposite vectors have distance 2. The function uses O(d) time and
    temporary space.
    """
    return _compare(a, b, "cosine")


def _compare(a: ArrayLike, b: ArrayLike, metric: Metric) -> float:
    left = prepare_vector(a, metric)
    right = prepare_vector(b, metric)
    if left.shape != right.shape:
        raise ValueError("Vector dimensions must match")
    return prepared_distance(left, right, metric)
