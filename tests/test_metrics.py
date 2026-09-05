"""Tests for GraphSeek's vector validation rules."""

import numpy as np
import pytest

from graphseek.metrics import validate_vector


# Accepted inputs


def test_valid_integer_vector() -> None:
    result = validate_vector([1, 2, 3])

    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_valid_floating_point_vector() -> None:
    result = validate_vector((1.5, 2.5))

    np.testing.assert_array_equal(result, np.array([1.5, 2.5]))


def test_valid_numpy_array() -> None:
    result = validate_vector(np.array([1, 2]))

    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))


# Invalid shapes and values


def test_empty_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_vector([])


def test_two_dimensional_array_is_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_vector(np.array([[1, 2], [3, 4]]))


def test_string_value_is_rejected() -> None:
    with pytest.raises(TypeError, match="real numbers"):
        validate_vector([1, "hello"])  # type: ignore[list-item]


def test_complex_value_is_rejected() -> None:
    with pytest.raises(TypeError, match="real numbers"):
        validate_vector([1, 2j])  # type: ignore[list-item]


def test_boolean_value_is_rejected() -> None:
    with pytest.raises(TypeError, match="real numbers"):
        validate_vector([1, True])


def test_nan_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_vector([1.0, np.nan])


@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_infinity_is_rejected(infinity: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_vector([1.0, infinity])


# Copy behavior


def test_original_input_remains_unchanged() -> None:
    original = np.array([1.0, 2.0])

    result = validate_vector(original)

    # Mutating the returned vector must never mutate the caller's input.
    result[0] = 99

    np.testing.assert_array_equal(original, np.array([1.0, 2.0]))
