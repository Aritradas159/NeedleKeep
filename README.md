# NeedleKeep
### Keep the context that matters.

**Sentence-Boundary-Aware KV-Cache Selection for Efficient LLM Inference**

NeedleKeep is an experimental KV-cache selection approach that prioritizes sentence-boundary-aware context retention to reduce the amount of key-value (KV) cache retained during long-context language model inference.

Instead of retaining every token or selecting tokens using a simple fixed-window strategy, NeedleKeep's **SNAP-BND** approach aims to preserve useful context by respecting sentence boundaries.

> **Core idea:** Retain a smaller, structured subset of the context while evaluating how well the model preserves answer accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Benchmark Results](#benchmark-results)
- [Experimental Setup](#experimental-setup)
- [Repository Structure](#repository-structure)
- [Reproduce the Experiments](#reproduce-the-experiments)
- [Verification](#verification)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Overview

Large language models use a key-value cache to store intermediate attention states during inference. This cache helps avoid recomputing previous tokens, but its memory requirements grow with context length.

NeedleKeep investigates whether a sentence-boundary-aware selection strategy can retain substantially less KV-cache data while preserving task performance.

The project compares SNAP-BND against four policies:

- **Full:** Retain the full context.
- **SNAP-BND:** Sentence-boundary-aware selection.
- **SNAP-W16:** Fixed-window selection baseline.
- **Recent:** Retain the most recent context.
- **Stream4:** Simplified streaming-style baseline.

The implementation includes benchmark scripts, core verification, experiment outputs, and a notebook for exploring the approach.

## Problem Statement

As prompts grow longer, KV-cache memory can become a constraint on inference efficiency. However, removing context indiscriminately can discard information needed to answer a query correctly.

NeedleKeep explores the following question:

**Can sentence-boundary-aware KV-cache selection reduce retained cache memory while maintaining useful answer accuracy?**

The evaluation focuses on strict answer correctness, exact-match accuracy, retained-cache estimates, selector overhead, and inference timing.

## How It Works

At a high level, the experiment follows this workflow:

1. **Prepare the context:** Construct benchmark examples with different context lengths and query types.
2. **Select context positions:** Apply a cache-retention policy to determine which positions to retain.
3. **Run inference:** Evaluate the language model using the selected context.
4. **Score the response:** Measure strict correctness and exact-match accuracy.
5. **Compare efficiency:** Track retained KV-cache estimates and timing metrics across policies.

SNAP-BND is compared with the full-context policy and simpler selection baselines under the same benchmark cases.

## Benchmark Results

The benchmark used **120 records**, with 24 records per policy across five policies.

### Strict answer accuracy

| Policy | Correct | Accuracy |
|---|---:|---:|
| Full | 14/24 | 58.33% |
| **SNAP-BND** | **16/24** | **66.67%** |
| SNAP-W16 | 9/24 | 37.50% |
| Recent | 0/24 | 0.00% |
| Stream4 | 1/24 | 4.17% |

### Exact-match accuracy

| Policy | Exact matches | Accuracy |
|---|---:|---:|
| Full | 14/24 | 58.33% |
| **SNAP-BND** | **20/24** | **83.33%** |
| SNAP-W16 | 12/24 | 50.00% |
| Recent | 0/24 | 0.00% |
| Stream4 | 1/24 | 4.17% |

### Retained KV-cache estimate

| Metric | Full | Selection policies |
|---|---:|---:|
| Mean retained positions | 3,091.5 | 96 |
| Mean estimated KV-cache | 36.229 MiB | 1.125 MiB |
| Estimated cache reduction vs. full | — | 96.48% |

The estimated reduction is in **retained KV-cache memory**, not total GPU memory consumption.

### What these results suggest

- SNAP-BND achieved higher strict and exact-match accuracy than the full-context baseline in this benchmark run.
- The selection policies retained substantially fewer positions than the full-context policy.
- SNAP-BND had a mean selector time of approximately **0.355 seconds**, compared with approximately **0.056 seconds** for SNAP-W16.
- The results motivate further evaluation on larger and more diverse workloads.

**Important:** The benchmark contains only 24 records per policy. The reported accuracy difference does not establish statistically significant superiority or guarantee similar performance on other models and datasets.

## Experimental Setup

| Component | Configuration |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Benchmark size | 120 records |
| Policies | Full, SNAP-BND, SNAP-W16, Recent, Stream4 |
| Records per policy | 24 |
| Random seeds | 3000–3003 |
| Context lengths | 112 and 224 |
| Query types | `id`, `mgr`, `bud` |
| Selection-policy budget | 96 retained positions |
| Reproduced run | 120/120 records completed |

The benchmark also records paired comparisons and a SNAP-BND ablation output. See the CSV files in `results_reproduced/` for the detailed results.

## Repository Structure

```text
NeedleKeep/
├── README.md
├── verification_report.json
├── code/
│   ├── recallguard_core.py
│   ├── run_benchmark.py
│   └── verify_core.py
├── notebooks/
│   └── RecallGuard-KV-experiment.ipynb
└── results_reproduced/
    ├── fresh_seed_stretch_comparison.csv
    ├── fresh_seed_stretch_summary.csv
    ├── fresh_seed_paired_comparisons.csv
    └── fresh_seed_snap_ablation.csv
```

The Python filenames and notebook retain some of their original RecallGuard-KV names from the project's development stage.

## Reproduce the Experiments

### Requirements

- Python 3
- The dependencies required by the benchmark implementation
- Access to the Hugging Face model `Qwen/Qwen2.5-0.5B-Instruct`
- A compatible runtime with sufficient memory for model inference

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Aritradas159/NeedleKeep.git
   cd NeedleKeep
   ```

2. Review the implementation and install the required dependencies for your environment.

3. Run the core verification:

   ```bash
   python code/verify_core.py
   ```

4. Run the benchmark:

   ```bash
   python code/run_benchmark.py
   ```

Check the script's configuration and output paths before running it. Model downloads, dependency versions, hardware, and runtime configuration may affect execution and timing.

The `results_reproduced/` directory contains the saved benchmark outputs from the reproduced run.

## Verification

The core verification report records **four checks passed and zero failures**.

The benchmark was also reproduced on a Google Colab T4 runtime, completing all 120 records. The reproduced run matched the original run's case-policy coverage and strict and exact correctness counts, although the generated answer text differed.

This supports reproducibility of the reported scores under the tested setup, not deterministic reproduction of every generated response.

See [`verification_report.json`](verification_report.json) for the core verification details.

## Limitations

NeedleKeep is an experimental prototype, and the current results have important limitations:

- The evaluation uses a small benchmark with 24 records per policy.
- The model tested is `Qwen/Qwen2.5-0.5B-Instruct`; results may differ for larger models.
- The estimated KV-cache reduction is not equivalent to a measured reduction in peak GPU memory.
- Selector overhead may offset some of the benefits in latency-sensitive workloads.
- Stream4 is a simplified streaming-style baseline, not a faithful implementation of the official StreamingLLM method.
- The benchmark's H2O-style baseline, where applicable in the development code, should not be treated as a faithful official H2O implementation.
- Broader datasets, longer contexts, additional architectures, and stronger statistical evaluation are needed.

These limitations are important when interpreting the benchmark numbers.

## Future Work

Potential next steps include:

- Evaluate on larger language models and longer contexts.
- Expand the benchmark with more tasks, records, and random seeds.
- Measure actual peak memory usage and end-to-end latency.
- Compare against faithful implementations of established KV-cache methods.
- Investigate the trade-off between selection overhead, retained memory, and answer quality.
- Improve experiment automation and configuration management.

## Acknowledgements

Built as an experimental project for exploring efficient language-model inference and KV-cache selection.

**Repository:** [Aritradas159/NeedleKeep](https://github.com/Aritradas159/NeedleKeep)

---

*NeedleKeep — Keep the context that matters.*
