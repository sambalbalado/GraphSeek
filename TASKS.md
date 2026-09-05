# Development plan

## Week 1: exact-search foundation

- [x] Vector validation, squared L2, and cosine distance
- [x] Exact index with stable IDs and deterministic ties
- [x] Seeded clustered data and ground truth
- [x] Known examples, invalid inputs, and reference tests
- [ ] Read the code and try the README example
- [ ] Explain why exact search gets slower as the collection grows
- [ ] Add one edge-case test of your own

## Next: graph search

1. Read HNSW's layer-search pseudocode.
2. Implement candidate queues and compare them with sorting.
3. Search a small graph using a visited set.
4. Select diverse neighbors and limit degree.
5. Implement seeded levels and multilayer insertion.
6. Implement querying and compare results with FlatIndex.

## Later

- Measure recall, latency, build time, and memory.
- Save and reload indexes with validation.
- Add a search API.
- Compare with FAISS on identical data.
- Write up measured results and limitations.
