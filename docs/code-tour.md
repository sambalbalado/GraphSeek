# Code tour

## The idea

A vector is a list of numbers that describes an item. An embedding model might
turn two source files with similar meaning into vectors that point in similar
directions. Vector search receives one query vector and finds the stored
vectors closest to it.

GraphSeek currently uses exact search. For every query it measures the distance
to every stored vector, then returns the closest results. This gives the right
answer, but the work grows with the collection. It will become the reference
used to check whether the future HNSW index is returning good approximate
answers.

## Request flow

```text
raw list or NumPy array
        |
        v
validate and copy the vector
        |
        v
normalize it when using cosine distance
        |
        v
store it with a stable integer ID
        |
query -> compare with every stored vector -> keep the closest k results
```

## `metrics.py`

`validate_vector` converts input into a one-dimensional float64 NumPy array.
It rejects empty, nested, nonnumeric, infinite, and NaN values. It returns a
copy so changing the original input cannot silently alter the index.

Squared L2 measures ordinary geometric separation without taking a square
root:

```text
(a1 - b1)^2 + (a2 - b2)^2 + ... + (ad - bd)^2
```

Removing the square root saves work and preserves ranking because square root
is increasing: the smallest squared distance is also the smallest distance.

Cosine distance measures direction. Vectors are normalized to length one when
they enter an index. Search can then use `1 - dot_product`. A zero vector has
no direction, so cosine distance rejects it.

## `flat.py`

`Neighbor` is a result with a vector ID and distance. It is frozen so a caller
cannot accidentally change a returned result.

`FlatIndex.add` validates and stores a vector. The first insertion fixes the
dimension, so a two-dimensional and three-dimensional vector cannot enter the
same index. IDs are the insertion positions: 0, 1, 2, and so on.

`FlatIndex.search` validates the query, computes one distance for every stored
vector, and uses a bounded heap to keep the closest `k`. Equal distances are
ordered by ID so repeated runs give the same result.

For `n` stored vectors of `d` dimensions, search takes approximately `n * d`
distance work plus heap maintenance. That is O(nd + n log k) time. This linear
scan is the problem HNSW will later address.

## `datasets.py`

`clustered_data` creates random centre points and samples database vectors and
queries near those centres. Passing the same seed produces the same numbers,
which makes bugs and comparisons repeatable.

`ground_truth` inserts generated vectors into `FlatIndex` and records the exact
closest IDs for each query. Future HNSW results can be compared with this list
to calculate recall.

The module also acts as a command-line program. It prints the configuration and
neighbor IDs as JSON so results can be consumed by another script.

## `tests/test_search.py`

The tests cover distances that can be calculated by hand, bad inputs, stable
ties, input copying, index dimensions, cosine ordering, seeded data, and an
independent NumPy nearest-neighbor calculation. The independent calculation is
especially useful: it checks `FlatIndex` without relying on `FlatIndex` to
define its own expected answer.

## Why this foundation matters

Approximate search cannot be evaluated without a trustworthy exact answer.
These files establish the input rules, distance behavior, result format, and
repeatable datasets that HNSW will share. When HNSW is added, recall can be
reported as a measured comparison instead of assumed correctness.
