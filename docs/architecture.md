# Design notes

FlatIndex stores float64 copies and fixes the dimension on the first insertion.
Cosine vectors are normalized once on insertion. IDs break distance ties.

Search computes every distance and selects results with a heap. For n vectors
of dimension d and k results (capped at n), time is O(nd + n log k), storage
is O(nd), and temporary space is O(d + k). Public distance functions copy
inputs and use O(d) temporary space.

Invalid insertions leave the index unchanged. Cosine normalization scales
before computing norms to avoid overflow. Squared L2 reports overflow.

HNSW will reuse the distance functions and Neighbor type. FlatIndex remains
the exact reference for recall measurements.
