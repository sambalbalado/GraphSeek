# Architecture

GraphSeek will keep algorithmic code independent from interfaces and
benchmarking infrastructure.

```text
HTTP API / CLI
      |
Index interface
  |         |
FlatIndex   HNSWIndex
      \     /
    Distance metrics
```

## Planned components

- `metrics`: validated cosine and Euclidean distance functions
- `flat`: exhaustive top-k search and ground-truth generation
- `hnsw`: multilayer graph construction and approximate search
- `persistence`: versioned, deterministic index serialization
- `api`: input validation and HTTP endpoints
- `benchmarks`: datasets, experiment runner, and result reporting

Implementation decisions and complexity analysis should be recorded here as
the design evolves.

