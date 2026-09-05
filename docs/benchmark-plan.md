# Benchmark Plan

## Question

How do HNSW construction and search parameters change retrieval quality,
latency, throughput, build time, and memory usage relative to exact search?

## Datasets

1. Seeded synthetic vectors for fast and repeatable development.
2. A public approximate-nearest-neighbor dataset for comparable results.
3. Source-code embeddings, optionally derived from PatchBench cases.

## Metrics

- recall@1 and recall@10 against exact ground truth
- median, p95, and p99 query latency
- queries per second after warm-up
- index construction time
- peak resident memory and bytes per vector

## Experimental controls

- Pin random seeds and record dependency versions.
- Use the same vectors and queries for all indexes.
- Separate index build time from query time.
- Include warm-up queries and multiple repetitions.
- Report dataset size, dimensionality, hardware, and parameters.
- Retain failed or unfavorable results and explain them.

