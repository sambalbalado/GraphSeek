# GraphSeek

GraphSeek is a learning project for building vector search from the ground up.

The first goal is to understand how vectors and distance calculations work.
After that, the project will grow one small, tested step at a time toward exact
nearest-neighbour search and then HNSW approximate search.

## Current status

The repository has been reset to a minimal starting point. No search algorithm
has been implemented yet.

## Source

Python code will live in `src/graphseek/`.

## Project structure

- `src/graphseek/` contains the importable Python package.
- `tests/` contains automated tests.
- `pyproject.toml` describes the project, its supported Python version, and its
  dependencies.

Keeping importable code under `src/` prevents Python from accidentally importing
files directly from the repository root. Tests therefore exercise the package as
it is installed, which is closer to how other programs will use it.

## Set up the project

Python 3.11 or newer is required. From the repository root, create an isolated
virtual environment and install GraphSeek with its development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `-e` option performs an editable install, so changes under `src/` are used
without reinstalling the package. The `[dev]` extra installs tools needed while
developing the project, currently `pytest`.

Run the automated tests with:

```bash
pytest
```

NumPy is a runtime dependency because GraphSeek uses NumPy arrays as its
validated vector representation. Installing the project installs NumPy
automatically.

## Validate a vector

Use `validate_vector()` to turn a supported input into GraphSeek's standard
vector representation:

```python
from graphseek.metrics import validate_vector

vector = validate_vector([1, 2, 3])
print(vector)  # [1. 2. 3.]
```

The function accepts lists, tuples, and one-dimensional NumPy arrays. It returns
a separate NumPy `float64` array, so changing the result does not change the
original input.

Vectors must contain at least one real, finite number. Nested vectors, strings,
complex numbers, booleans, `NaN`, and positive or negative infinity are
rejected with a descriptive exception.
