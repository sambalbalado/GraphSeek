# GraphSeek

GraphSeek is an educational vector-search engine that will implement
Hierarchical Navigable Small World (HNSW) search from first principles.
The goal is to make the speed–accuracy trade-off measurable rather than hide
the algorithm behind a library call.

> Status: project scaffold. The algorithm is intentionally built through
> small, tested commits described in [`TASKS.md`](TASKS.md).

## Why this project

Modern AI systems retrieve semantically similar documents, source code, and
images from high-dimensional embedding spaces. Exact search is accurate but
becomes expensive as a collection grows. GraphSeek will compare exact search
with HNSW approximate search using recall, latency, throughput, build time,
and memory measurements.

## Planned capabilities

- Exact top-k search as a ground-truth baseline
- Cosine and Euclidean distance metrics
- HNSW insertion and querying implemented without a vector-database library
- Deterministic index persistence
- A small HTTP API
- Reproducible benchmarks against exact search and FAISS
- Experiments on synthetic vectors and source-code embeddings

## Project principles

1. Correctness comes before optimization.
2. Every algorithmic feature includes tests and complexity notes.
3. Benchmarks report the environment and never cherry-pick results.
4. Each task becomes one focused, reviewable commit.
5. AI assistance must explain decisions; generated code is verified locally.

## Quick start

GraphSeek requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
mypy src
```

## Repository layout

```text
src/graphseek/        Python package
tests/                Unit and integration tests
benchmarks/           Reproducible benchmark entry points and results
docs/                 Architecture and experimental methodology
.github/workflows/    Continuous integration
```

## Roadmap

The implementation roadmap, acceptance criteria, AI prompts, and intended
commit messages live in [`TASKS.md`](TASKS.md). Work on only one task at a
time so the Git history shows how the system developed.

## Target evaluation

The final report will include:

- recall@1 and recall@10
- median and p95 query latency
- queries per second
- index build time
- peak memory and bytes per vector
- results across multiple dataset sizes and HNSW parameter settings

## References

- Malkov and Yashunin, *Efficient and Robust Approximate Nearest Neighbor
  Search Using Hierarchical Navigable Small World Graphs*
- FAISS documentation and published HNSW benchmark methodology

## License

MIT

