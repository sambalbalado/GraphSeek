"""Distance functions for finite, one-dimensional vectors."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Metric = Literal["l2", "cosine"]
Vector = NDArray[np.float64]


def vector(value: ArrayLike) -> Vector:
    """Validate and copy input so callers cannot change stored vectors."""
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("Expected a non-empty one-dimensional vector")
    if raw.dtype.kind not in "iuf":
        raise ValueError("Vector values must be real numbers")
    result = raw.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Vector values must be finite")
    return result


def normalize(value: Vector) -> Vector:
    # Scaling first avoids overflow when computing the norm of large values.
    scale = float(np.max(np.abs(value)))
    if scale == 0:
        raise ValueError("Cosine distance is undefined for a zero vector")
    scaled = value / scale
    return scaled / np.linalg.norm(scaled)


def prepare(value: ArrayLike, metric: Metric) -> Vector:
    result = vector(value)
    return normalize(result) if metric == "cosine" else result


def distance(a: Vector, b: Vector, metric: Metric) -> float:
    """Compare already validated vectors (normalized for cosine)."""
    if metric == "cosine":
        return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))
    with np.errstate(over="ignore"):
        delta = a - b
        result = float(np.dot(delta, delta))
    if not np.isfinite(result):
        raise ValueError("Squared distance exceeds float64 range")
    return result


def squared_l2(a: ArrayLike, b: ArrayLike) -> float:
    """Squared Euclidean distance; O(d) time and auxiliary space."""
    return _compare(a, b, "l2")


def cosine_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine distance in [0, 2]; O(d) time and auxiliary space."""
    return _compare(a, b, "cosine")


def _compare(a: ArrayLike, b: ArrayLike, metric: Metric) -> float:
    left, right = prepare(a, metric), prepare(b, metric)
    if left.shape != right.shape:
        raise ValueError("Vector dimensions must match")
    return distance(left, right, metric)
