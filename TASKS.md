# GraphSeek Implementation Tasks

Complete these tasks in order. Each task is deliberately small enough to
produce a focused commit. Replace benchmark placeholders with measurements
from your own machine; never ask an AI to invent results.

## Task 01 — Define vector validation and distance metrics

**Goal:** Create typed cosine-distance and squared-Euclidean-distance
functions with shared validation for shape, dtype, empty vectors, and zero
norms.

**Acceptance criteria:** Hand-calculated unit cases pass; invalid inputs raise
clear errors; inputs are not mutated; complexity is documented as O(d) time
and O(1) auxiliary space.

**Commit:** `feat: add validated vector distance metrics`

**AI prompt:**

> Work on GraphSeek Task 01 only. Read README.md, CONTRIBUTING.md, and the
> current source and tests. Run the existing checks first. Design typed NumPy
> functions for cosine distance and squared Euclidean distance, including a
> shared validation policy for one-dimensional numeric vectors, equal shapes,
> finite values, and cosine zero norms. Add hand-calculated tests, mutation
> checks, and meaningful error tests. Document the time and auxiliary-space
> complexity. Do not implement an index or work ahead. Run pytest, Ruff, and
> mypy, show me the diff and explain the design. If I approve, commit exactly:
> `feat: add validated vector distance metrics`.

## Task 02 — Implement the exact FlatIndex baseline

**Goal:** Store vectors with stable integer IDs and return exact top-k results.

**Acceptance criteria:** Supports add, length, and search; handles ties
deterministically; validates `k`; matches a manual example; documents O(nd)
query time.

**Commit:** `feat: implement exact flat vector index`

**AI prompt:**

> Work on GraphSeek Task 02 only. Inspect the repository and run all checks.
> Define the smallest useful index interface, then implement an in-memory
> FlatIndex using Task 01 metrics. Preserve stable integer IDs and return typed
> results ordered by distance and then ID for deterministic ties. Validate
> dimensions, empty-index searches, and k. Add unit tests against manually
> computed neighbors and document construction, memory, and query complexity.
> Do not implement HNSW. Run all checks and show the diff. If I approve, commit
> exactly: `feat: implement exact flat vector index`.

## Task 03 — Add deterministic datasets and ground truth

**Goal:** Generate seeded clustered vectors and exact nearest-neighbor ground
truth for later benchmarks.

**Acceptance criteria:** Same seed produces identical arrays; dataset shape is
configurable; ground truth uses FlatIndex; generation has tests and a CLI help
message.

**Commit:** `feat: add reproducible benchmark datasets`

**AI prompt:**

> Work on GraphSeek Task 03 only. Run current checks, then add a deterministic
> synthetic dataset generator with explicit seed, vector count, query count,
> dimensions, and cluster count. Generate exact top-k ground truth through
> FlatIndex rather than duplicating search logic. Add a lightweight CLI and
> tests for reproducibility, shapes, and invalid arguments. Do not add HNSW or
> commit generated data. Run all checks, show the diff, and explain why the
> dataset is useful. If approved, commit: `feat: add reproducible benchmark datasets`.

## Task 04 — Implement bounded candidate queues

**Goal:** Build the min/max priority-queue helpers used during graph traversal.

**Acceptance criteria:** Deterministic tie handling; correct bounded behavior;
duplicate policy documented; property-style randomized tests compare results
with sorting.

**Commit:** `feat: add deterministic candidate queues`

**AI prompt:**

> Work on GraphSeek Task 04 only. Study the HNSW paper's layer-search needs and
> inspect the existing typed result structures. Implement minimal heap-based
> candidate and bounded-result queues with deterministic distance/ID tie
> ordering. State and test the duplicate policy. Add randomized tests that
> compare queue output with a simple sorted reference. Document O(log k)
> operations. Do not implement graph traversal yet. Run all checks and show
> the diff. If approved, commit: `feat: add deterministic candidate queues`.

## Task 05 — Implement search within one proximity-graph layer

**Goal:** Given entry points and a graph layer, find the best `ef` candidates.

**Acceptance criteria:** Follows the paper's stopping rule; never revisits a
node; deterministic output; tests use tiny hand-drawn graphs and disconnected
cases.

**Commit:** `feat: implement single-layer graph search`

**AI prompt:**

> Work on GraphSeek Task 05 only. Read the original HNSW layer-search
> pseudocode and map each step to existing GraphSeek types. Implement search
> within one supplied adjacency layer using the Task 04 queues and a visited
> set. Keep distance calculation injectable or metric-aware. Add tests using
> small hand-drawn graphs, multiple entry points, cycles, ties, and a
> disconnected node. Explain the stopping condition and complexity. Do not
> implement insertion or hierarchy. Run all checks and show the diff. If
> approved, commit: `feat: implement single-layer graph search`.

## Task 06 — Add heuristic neighbor selection and pruning

**Goal:** Select diverse neighbors and enforce a maximum graph degree.

**Acceptance criteria:** Maximum degree is never exceeded; selection is
deterministic; closer-but-redundant candidates can be rejected; simple and
heuristic strategies are independently tested.

**Commit:** `feat: add HNSW neighbor selection heuristic`

**AI prompt:**

> Work on GraphSeek Task 06 only. Implement both a simple nearest-candidate
> selector and the HNSW diversity heuristic as small independently tested
> functions. Add pruning that enforces maximum degree after reciprocal edges
> are inserted. Use geometric examples that demonstrate why a redundant close
> neighbor may be rejected for a more diverse one. Document deterministic tie
> behavior and complexity. Do not create the multilayer index. Run all checks
> and show the diff. If approved, commit:
> `feat: add HNSW neighbor selection heuristic`.

## Task 07 — Build deterministic HNSW insertion

**Goal:** Insert vectors into a multilayer HNSW graph using seeded level
assignment.

**Acceptance criteria:** Same seed gives the same topology; entry point and
maximum layer update correctly; reciprocal degree limits hold; invariants are
validated after every test insertion.

**Commit:** `feat: implement seeded HNSW insertion`

**AI prompt:**

> Work on GraphSeek Task 07 only. Assemble the existing layer search and
> neighbor-selection pieces into an HNSWIndex insertion path. Use a private,
> explicitly seeded random generator for exponential level assignment. Handle
> the empty index, descent through upper layers, reciprocal links, pruning,
> entry-point replacement, and dimensional validation. Add an invariant
> checker used heavily in tests. Test topology reproducibility without locking
> tests to unnecessary implementation details. Do not add querying or an API.
> Run all checks, show the diff, and explain the insertion algorithm. If
> approved, commit: `feat: implement seeded HNSW insertion`.

## Task 08 — Implement HNSW top-k querying

**Goal:** Search from the top entry point down to layer zero and return the
best k approximate neighbors.

**Acceptance criteria:** Search is deterministic; `ef_search >= k` is
enforced; results improve or remain stable as `ef_search` grows on test data;
recall is calculated against FlatIndex.

**Commit:** `feat: implement HNSW approximate search`

**AI prompt:**

> Work on GraphSeek Task 08 only. Implement end-to-end HNSW top-k querying:
> greedy descent through upper layers followed by an ef_search exploration at
> layer zero. Return the same result type as FlatIndex and validate all public
> arguments. Add tests for empty/singleton indexes, k boundaries,
> determinism, and recall against FlatIndex on a seeded dataset. Include a
> monotonic-quality test that avoids flaky timing. Run all checks, show the
> diff, and explain how ef_search changes work and recall. If approved, commit:
> `feat: implement HNSW approximate search`.

## Task 09 — Create a correctness and recall evaluation harness

**Goal:** Run both indexes over a query set and compute retrieval-quality
metrics.

**Acceptance criteria:** Correct recall@k intersection formula; machine-readable
output; validation prevents incomparable runs; tests cover perfect, partial,
and zero recall.

**Commit:** `feat: add recall evaluation harness`

**AI prompt:**

> Work on GraphSeek Task 09 only. Add an evaluation layer that builds exact
> ground truth, queries another index, and reports recall@k plus per-query
> summary data. Keep quality evaluation separate from performance timing.
> Provide JSON output with configuration and random seed. Add hand-calculated
> tests for perfect, partial, and zero overlap and reject mismatched query or k
> configurations. Do not optimize or add charts. Run all checks and show the
> diff. If approved, commit: `feat: add recall evaluation harness`.

## Task 10 — Add versioned index persistence

**Goal:** Save and load FlatIndex and HNSWIndex without losing behavior.

**Acceptance criteria:** Round trips preserve vectors, IDs, parameters,
topology, entry point, and results; corrupt and unknown-version files fail
clearly; loading does not execute code.

**Commit:** `feat: add safe versioned index persistence`

**AI prompt:**

> Work on GraphSeek Task 10 only. Design a documented, versioned persistence
> format that does not rely on pickle or executable deserialization. Implement
> save/load for both indexes using explicit metadata and numeric arrays.
> Validate schema, dimensions, node references, and graph invariants when
> loading. Add round-trip, corruption, and unknown-version tests. Ensure
> searches before and after loading are identical. Run all checks and show the
> format and diff. If approved, commit:
> `feat: add safe versioned index persistence`.

## Task 11 — Expose a minimal HTTP search service

**Goal:** Serve health, vector insertion, and top-k search endpoints.

**Acceptance criteria:** Typed request validation; correct status codes;
dependency-injected index; endpoint tests require no live server; algorithmic
package remains independent of FastAPI.

**Commit:** `feat: expose vector search HTTP API`

**AI prompt:**

> Work on GraphSeek Task 11 only. Add FastAPI as an optional service dependency
> and expose health, vector insertion, and top-k search endpoints. Use typed
> request/response models, explicit size limits, and dependency injection so
> tests can use either index. Keep FastAPI imports out of algorithm modules.
> Add endpoint tests for successful and invalid requests without starting a
> real network server. Update setup instructions. Run all checks and show the
> diff. If approved, commit: `feat: expose vector search HTTP API`.

## Task 12 — Build the reproducible performance benchmark

**Goal:** Measure latency, throughput, construction time, and memory across
dataset sizes and HNSW parameters.

**Acceptance criteria:** Warm-up and repeated trials; p50/p95/p99 reporting;
environment metadata; raw JSON and chart generation; timing tests do not make
CI flaky.

**Commit:** `feat: add reproducible performance benchmarks`

**AI prompt:**

> Work on GraphSeek Task 12 only. Implement a configurable benchmark runner
> for FlatIndex and HNSWIndex across vector counts, dimensions, M,
> ef_construction, and ef_search. Separate build and query phases, warm up
> before measuring, repeat trials, and report p50/p95/p99, QPS, recall, build
> time, and memory with environment metadata. Save raw JSON and generate clear
> charts from it. Unit-test calculations and configuration, but keep timing
> assertions out of CI. Run checks on a tiny dataset and show the diff. If
> approved, commit: `feat: add reproducible performance benchmarks`.

## Task 13 — Profile and optimize one verified bottleneck

**Goal:** Use profiling evidence to improve performance without reducing
correctness.

**Acceptance criteria:** Before/after profile retained; optimization is scoped
to a measured hotspot; all correctness and recall tests pass; report includes
the non-cherry-picked change.

**Commit:** `perf: optimize measured HNSW search bottleneck`

**AI prompt:**

> Work on GraphSeek Task 13 only. Run the benchmark and profile a representative
> workload. Identify the single largest actionable hotspot and show the
> evidence before editing. Propose the smallest optimization, implement it,
> and rerun correctness, recall, and performance measurements under identical
> settings. Record both favorable and unfavorable changes and avoid claiming
> significance from noise. Do not rewrite unrelated code. Run all checks and
> show before/after evidence and the diff. If approved, commit:
> `perf: optimize measured HNSW search bottleneck`.

## Task 14 — Compare GraphSeek with FAISS

**Goal:** Benchmark GraphSeek, FAISS exact search, and FAISS HNSW fairly on the
same data.

**Acceptance criteria:** Same vectors/queries/metric; parameters and library
version recorded; comparison discusses implementation and hardware limits;
no claim that GraphSeek is production-superior without evidence.

**Commit:** `bench: compare GraphSeek with FAISS indexes`

**AI prompt:**

> Work on GraphSeek Task 14 only. Add FAISS as an optional benchmark dependency
> and adapters for its exact and HNSW indexes. Feed every implementation the
> same normalized vectors, queries, ground truth, k, metric, and warm-up
> policy. Record FAISS version and parameters. Produce a table and recall/QPS
> trade-off plot, then write an honest interpretation including Python/native
> implementation differences and hardware limitations. Never fabricate or
> import someone else's numbers. Run all checks and show the results and diff.
> If approved, commit: `bench: compare GraphSeek with FAISS indexes`.

## Task 15 — Evaluate source-code embeddings

**Goal:** Demonstrate a relevant AI retrieval use case using code snippets or
PatchBench cases.

**Acceptance criteria:** Dataset provenance and license documented; embedding
model/version pinned; no secrets or private code committed; qualitative
examples supplement—not replace—recall and latency results.

**Commit:** `feat: evaluate source code similarity search`

**AI prompt:**

> Work on GraphSeek Task 15 only. Design a small, reproducible code-similarity
> evaluation using public code or explicitly exportable PatchBench cases.
> Document provenance and license, pin the embedding model and revision, cache
> only redistributable artifacts, and keep credentials out of Git. Compare
> exact and HNSW retrieval using the existing harness, then add a few clearly
> labeled qualitative examples. Do not change algorithm behavior to favor the
> examples. Run all checks and show the report and diff. If approved, commit:
> `feat: evaluate source code similarity search`.

## Task 16 — Publish the final engineering report and demo

**Goal:** Make the repository understandable and reproducible in five minutes.

**Acceptance criteria:** README contains architecture, usage, measured results,
limitations, and demo; fresh-environment instructions work; release tag follows
all checks; résumé bullets use real numbers only.

**Commit:** `docs: publish GraphSeek benchmark report`

**AI prompt:**

> Work on GraphSeek Task 16 only. Audit the finished repository from a
> recruiter's perspective. Re-run the final benchmark, then update README.md
> with the problem, architecture diagram, algorithm explanation, reproducible
> setup, real results, trade-offs, limitations, and a short demo. Verify every
> command in a clean environment. Draft two résumé bullets using only measured
> numbers. Do not hide failed experiments or overstate this educational Python
> implementation. Run every check and show the final diff. If approved, commit:
> `docs: publish GraphSeek benchmark report`, then propose—but do not create—a
> `v1.0.0` tag until I confirm.

