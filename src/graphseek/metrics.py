"""Validation and distance helpers used throughout GraphSeek.

Inputs are converted to one consistent representation before distance metrics
or search algorithms use them.
"""

import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

# A validated vector is always a flat NumPy array of 64-bit floating-point
# numbers. VectorInput documents the three container types callers may provide.
Vector: TypeAlias = NDArray[np.float64]
VectorInput: TypeAlias = list[int | float] | tuple[int | float, ...] | NDArray[Any]


def validate_vector(value: VectorInput) -> Vector:
    """Return a finite, one-dimensional float copy of a numeric vector."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError("Vector must be a list, tuple, or NumPy array")

    # NumPy would silently turn [1, True] into [1, 1]. Check sequences before
    # conversion so a boolean cannot lose its original type.
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, (bool, np.bool_)) for item in value
    ):
        raise TypeError("Vector values must be real numbers; booleans are not allowed")

    # This temporary array lets us inspect the input's shape and value types.
    try:
        array = np.asarray(value)
    except ValueError as error:
        raise ValueError("Vector must be one-dimensional") from error

    # A vector must be flat and contain at least one value.
    if array.ndim != 1:
        raise ValueError("Vector must be one-dimensional")
    if array.size == 0:
        raise ValueError("Vector must not be empty")

    # NumPy uses i, u, and f for signed integers, unsigned integers, and floats.
    # Every other kind includes a value GraphSeek does not accept, such as a
    # string, boolean, object, or complex number.
    if array.dtype.kind not in "iuf":
        raise TypeError("Vector values must be real numbers; booleans are not allowed")

    # copy=True guarantees that changing the result cannot change the caller's
    # original NumPy array.
    result = array.astype(np.float64, copy=True)

    # NaN and positive or negative infinity are not usable distance values.
    if not np.isfinite(result).all():
        raise ValueError("Vector values must be finite")

    return result


def _validate_pair(left: VectorInput, right: VectorInput) -> tuple[Vector, Vector]:
    """Validate two vectors and require matching dimensions."""
    left_vector = validate_vector(left)
    right_vector = validate_vector(right)

    if left_vector.shape != right_vector.shape:
        raise ValueError("Vectors must have the same dimension")

    return left_vector, right_vector


def squared_l2(left: VectorInput, right: VectorInput) -> float:
    """Return squared Euclidean distance between equal-dimensional vectors.

    The distance calculation takes O(d) time and O(1) auxiliary space after
    validation, where d is the vector dimension. Validation creates two O(d)
    copies so callers' inputs remain independent and unmodified.
    """
    left_vector, right_vector = _validate_pair(left, right)

    distance = 0.0
    for left_value, right_value in zip(left_vector, right_vector, strict=True):
        difference = float(left_value - right_value)
        distance += difference * difference

    return distance


def cosine_distance(left: VectorInput, right: VectorInput) -> float:
    """Return cosine distance in [0, 2] for two nonzero vectors.

    Cosine distance is ``1 - cosine_similarity``. The calculation takes O(d)
    time and O(1) auxiliary space after validation. Validation creates two O(d)
    copies so callers' inputs remain independent and unmodified.
    """
    left_vector, right_vector = _validate_pair(left, right)

    dot_product = 0.0
    left_squared_magnitude = 0.0
    right_squared_magnitude = 0.0

    for left_value, right_value in zip(left_vector, right_vector, strict=True):
        left_number = float(left_value)
        right_number = float(right_value)
        dot_product += left_number * right_number
        left_squared_magnitude += left_number * left_number
        right_squared_magnitude += right_number * right_number

    if left_squared_magnitude == 0.0 or right_squared_magnitude == 0.0:
        raise ValueError("Cosine distance is undefined for zero vectors")

    magnitude_product = math.sqrt(left_squared_magnitude) * math.sqrt(
        right_squared_magnitude
    )
    similarity = dot_product / magnitude_product

    # Floating-point rounding can place a mathematically valid cosine just
    # outside [-1, 1]. Clamping preserves cosine distance's [0, 2] contract.
    bounded_similarity = min(1.0, max(-1.0, similarity))
    return 1.0 - bounded_similarity
