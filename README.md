# GraphSeek

A small vector search project in Python. Exact search works; HNSW is next.

## Setup

Requires Python 3.11 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Search

```python
from graphseek.flat import FlatIndex

index = FlatIndex(metric="l2")
index.add([1, 0])
index.add([0, 1])
print(index.search([0.9, 0.1], k=1))
```

`l2` returns squared Euclidean distance; `cosine` returns cosine distance.
Lower means closer. IDs start at zero and break ties. Requests larger than the
collection return every neighbor. Empty indexes return no neighbors. Invalid
vectors or nonpositive k raise ValueError. Cosine rejects zero vectors.

## Sample data

```sh
python -m graphseek.datasets --count 100 --queries 5 --dimensions 8 --k 3
```

Prints exact neighbor IDs for seeded clustered data as JSON. This is a
correctness fixture, not a performance benchmark.

## Files

- `src/graphseek/metrics.py`: distances and input checks
- `src/graphseek/flat.py`: exact search
- `src/graphseek/datasets.py`: sample data and ground truth
- `tests/`: known examples and reference comparisons
- `TASKS.md`: next steps

Vectors stay in memory and every search checks every vector. Persistence,
HTTP endpoints, and approximate search are not implemented yet. There are no
measured performance claims.

Next: [HNSW](https://arxiv.org/abs/1603.09320). MIT license.
