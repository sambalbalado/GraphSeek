"""Validation helpers for vectors used throughout GraphSeek.

Inputs are converted to one consistent representation before distance metrics
or search algorithms use them.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# A validated vector is always a flat NumPy array of 64-bit floating-point
# numbers. VectorInput documents the three container types callers may provide.
Vector: TypeAlias = NDArray[np.float64]
VectorInput: TypeAlias = list[int | float] | tuple[int | float, ...] | np.ndarray


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
