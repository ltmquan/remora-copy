# Additional eval targets

Inspected the local `darpa-mocha` repository read-only. These are candidates, not implemented Remora backends or measured scaling results.

| Target | Why try it | Useful partitions | Main comparison constraint |
|---|---|---|---|
| `65_hnsw` vector search | Large corpus and many independent queries; clean first distributed workload | Query batches with replicated corpus; corpus shards plus global top-k merge | Separate exact brute force from approximate ANN; report recall@k alongside throughput. A path named hnswlib uses GPU brute force, so labels alone are misleading. |
| `15_transformers` language modeling | Large weights, sequence activations and KV cache make memory and communication meaningful | Independent requests; tensor parallel; pipeline stages | Separate prefill from decode, fix model/token counts/precision, report tokens/s and latency. Existing inference uses one device; Mirage branch specifies world size 1. |
| `46_pyg` node classification | GCN/GAT/GraphSAGE on large graphs can be memory constrained | Uniform node, degree/edge-balanced, community partitions | Preserve cross-partition neighbors or control sampling. Report accuracy and communication, not just speed. Existing code moves graph data onto one device. |
| `49_pygod` graph anomaly detection | Offers minibatch and neighborhood controls | Graph minibatches and sampled neighborhoods | Keep graph, anomaly seed and sampling policy fixed; report AUC/precision@k. |

Start with vector search for a simple, interpretable scaling experiment, then language-model inference for model partitioning. Graph targets are useful after communication and correctness instrumentation is established. YOLOv8 adds another vision implementation but less variety in partition structure. `41_ortools` vehicle routing uses a CPU solver, so it is not a good first GPU-pipeline target.

The vector-search defaults include a million corpus vectors, 100,000 queries and dimension 128. The language-model benchmark's long default generation budget should be reduced for a smoke test and then specified explicitly. These repositories contain environment-specific dependencies/paths; copy any adaptation into this fork rather than modifying the evaluation originals.

A new eval target is not automatically a new Remora test: we still need its Remora program and compiler support for its operations. Label standalone backend baselines separately until that exists.
