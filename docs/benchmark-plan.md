# Benchmark notes

Use seeded clustered data first, then a public vector dataset. Every index
must receive the same data and queries.

Report recall@k, query latency, build time, and memory. Record hardware,
versions, seeds, and parameters. Separate build and search timing, warm up,
and repeat measurements.

The current dataset command generates ground truth only. Timing and memory
measurements are not implemented yet.
