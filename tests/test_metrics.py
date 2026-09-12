"""Tests for GraphSeek's vector validation rules."""

import numpy as np
import pytest

from graphseek.metrics import cosine_distance, squared_l2, validate_vector

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


# Squared Euclidean distance


def test_squared_l2_of_identical_vectors_is_zero() -> None:
    assert squared_l2([2, -1], [2, -1]) == 0.0


def test_squared_l2_matches_three_four_five_example() -> None:
    assert squared_l2([0, 0], [3, 4]) == 25.0


def test_squared_l2_supports_negative_coordinates() -> None:
    assert squared_l2([-2, -3], [1, 1]) == 25.0


def test_squared_l2_supports_floating_point_coordinates() -> None:
    assert squared_l2([0.5, 1.5], [1.0, 2.5]) == pytest.approx(1.25)


def test_squared_l2_is_symmetric() -> None:
    assert squared_l2([1, 5], [4, 1]) == squared_l2([4, 1], [1, 5])


def test_squared_l2_rejects_different_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        squared_l2([1, 2], [1, 2, 3])


def test_squared_l2_does_not_modify_inputs() -> None:
    left = np.array([0.0, 0.0])
    right = np.array([3.0, 4.0])

    squared_l2(left, right)

    np.testing.assert_array_equal(left, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(right, np.array([3.0, 4.0]))


def test_distance_metrics_apply_shared_vector_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        squared_l2([1.0, np.nan], [1.0, 2.0])

    with pytest.raises(ValueError, match="one-dimensional"):
        cosine_distance([[1, 0]], [1, 0])  # type: ignore[list-item]


# Cosine distance


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 0], [1, 0], 0.0),
        ([1, 0], [0, 1], 1.0),
        ([1, 0], [-1, 0], 2.0),
        ([1, 0], [10, 0], 0.0),
    ],
)
def test_cosine_distance_hand_calculated_directions(
    left: list[int], right: list[int], expected: float
) -> None:
    assert cosine_distance(left, right) == pytest.approx(expected)


def test_cosine_distance_supports_floating_point_coordinates() -> None:
    assert cosine_distance([1.5, 1.5], [3.0, 0.0]) == pytest.approx(
        1.0 - 1.0 / np.sqrt(2.0)
    )


@pytest.mark.parametrize("zero_position", ["left", "right"])
def test_cosine_distance_rejects_zero_vector(zero_position: str) -> None:
    left = [0, 0] if zero_position == "left" else [1, 0]
    right = [0, 0] if zero_position == "right" else [1, 0]

    with pytest.raises(ValueError, match="undefined for zero vectors"):
        cosine_distance(left, right)


def test_cosine_distance_rejects_different_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        cosine_distance([1, 0], [1, 0, 0])


def test_cosine_distance_does_not_modify_inputs() -> None:
    left = np.array([1.0, 0.0])
    right = np.array([0.0, 1.0])

    cosine_distance(left, right)

    np.testing.assert_array_equal(left, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(right, np.array([0.0, 1.0]))
