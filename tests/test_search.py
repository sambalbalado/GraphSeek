import numpy as np
import pytest

from graphseek.datasets import clustered_data, ground_truth
from graphseek.flat import FlatIndex
from graphseek.metrics import cosine_distance, squared_l2


def test_known_distances() -> None:
    assert squared_l2([0, 0], [3, 4]) == 25
    assert cosine_distance([1, 0], [0, 1]) == 1
    assert cosine_distance([1, 0], [-1, 0]) == 2
    assert cosine_distance([1e300, 0], [1e300, 0]) == 0


@pytest.mark.parametrize(
    "bad", [[], [[1, 2]], [float("nan")], [float("inf")], ["1"], [1j], [True]]
)
def test_rejects_invalid_vectors(bad: object) -> None:
    with pytest.raises(ValueError):
        squared_l2(bad, [1])


def test_dimension_and_zero_norm_errors() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        squared_l2([1], [1, 2])
    with pytest.raises(ValueError, match="zero"):
        cosine_distance([0, 0], [1, 0])
    with pytest.raises(ValueError, match="range"):
        squared_l2([1e300], [-1e300])


def test_index_copies_input_and_orders_ties() -> None:
    index = FlatIndex()
    value = np.array([1.0, 0.0])
    assert index.add(value) == 0
    assert index.add([-1, 0]) == 1
    value[:] = 100
    query = np.array([0.0, 0.0])
    results = index.search(query, 20)
    assert [(r.id, r.distance) for r in results] == [(0, 1), (1, 1)]
    np.testing.assert_array_equal(query, [0, 0])


def test_empty_index_and_failed_insert() -> None:
    index = FlatIndex()
    assert index.search([1], 1) == []
    index.add([1])
    with pytest.raises(ValueError):
        index.add([1, 2])
    assert len(index) == 1
    assert index.add([2]) == 1


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(k: int) -> None:
    with pytest.raises(ValueError):
        FlatIndex().search([1], k)


def test_cosine_search() -> None:
    index = FlatIndex("cosine")
    for value in ([0, 1], [2, 0], [-1, 0]):
        index.add(value)
    assert [r.id for r in index.search([5, 0], 3)] == [1, 0, 2]


def test_seeded_data_and_independent_search_reference() -> None:
    vectors, queries = clustered_data(30, 5, 4, 3, seed=7)
    repeated, _ = clustered_data(30, 5, 4, 3, seed=7)
    np.testing.assert_array_equal(vectors, repeated)
    expected = [
        np.argsort(np.sum((vectors - q) ** 2, axis=1))[:4].tolist() for q in queries
    ]
    assert ground_truth(vectors, queries, 4) == expected
    assert vectors.shape == (30, 4)
    assert queries.shape == (5, 4)


def test_dataset_rejects_invalid_size() -> None:
    with pytest.raises(ValueError):
        clustered_data(count=0)
