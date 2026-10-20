# RecallGuard-KV

**Question-aware KV-cache selection using sentence-boundary-aware spans for long-context recall.**

## Overview

RecallGuard-KV explores whether selecting a small subset of a language model's
key-value (KV) cache can preserve factual recall in long contexts. The proposed
variation, **SNAP-BND**, selects question-relevant cache positions while
respecting sentence boundaries.

Model: `Qwen/Qwen2.5-0.5B-Instruct`.

## Method and benchmark

The fresh-seed comparison contains 120 records: 24 cases for each of five
policies. The small synthetic recall task uses contexts of 112 or 224 sentences,
four seeds (3000–3003), and three query types (identifier, manager, and budget).

All compressed policies in this comparison retain 96 cache positions before
the question is processed. FULL is the uncompressed reference.

- **FULL:** retain the original KV cache.
- **SNAP-BND:** question-aware, sentence-boundary-aware selection.
- **SNAP-W16:** question-aware selection using 16-token spans.
- **Recent-96:** retain the most recent 96 positions.
- **Streaming-style (4 + 92):** retain the first four sink positions and the
  latest 92 positions. This is a simplified, post-prefill baseline, not the
  official online StreamingLLM implementation.

### Scoring

`exact` means the expected value is found by a boundary-aware regular
expression in the first line of the generated answer. `strict` requires an
exact match and excludes answers whose first line contains a known decoy value.
These metrics are not interchangeable.

## Fresh-seed results

| Policy | Strict accuracy | Strict 95% Wilson interval | Exact accuracy | Mean retained positions | Mean KV size (MiB) | Reported mean KV savings | Mean selector time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL | 14/24 (58.33%) | 38.83–75.53% | 14/24 (58.33%) | 3091.5 | 36.229 | 0.00% | 0.000 |
| SNAP-BND | 16/24 (66.67%) | 46.71–82.03% | 20/24 (83.33%) | 96.0 | 1.125 | 96.48% | 0.355 |
| SNAP-W16 | 9/24 (37.50%) | 21.16–57.29% | 12/24 (50.00%) | 96.0 | 1.125 | 96.48% | 0.056 |
| Recent-96 | 0/24 (0.00%) | 0.00–13.80% | 0/24 (0.00%) | 96.0 | 1.125 | 96.48% | 0.000 |
| Streaming-style (4 + 92) | 1/24 (4.17%) | 0.74–20.24% | 1/24 (4.17%) | 96.0 | 1.125 | 96.48% | 0.000 |

The summary CSV reports **96.48% mean retained-KV savings** for the compressed
policies. This is a measure of stored KV-cache size, not peak GPU-memory savings.

SNAP-BND achieved 16/24 strict-correct cases (66.67%), compared with 14/24
(58.33%) for FULL. The 95% Wilson intervals overlap, and this small benchmark
does not establish that SNAP-BND is generally more accurate than FULL.
SNAP-BND's exact accuracy was 20/24 (83.33%); this is a different scoring metric
from strict accuracy.

## Memory and latency limitations

Compression is performed after the full context cache has been created.
Therefore, these experiments demonstrate a reduction in the **retained KV cache**,
not a reduction in the peak memory needed to prefill the complete context.

The recorded mean `peak_delta_mib` is approximately 1450.718 MiB for every
policy. These results do not demonstrate peak GPU-memory savings. Mean prefill
times are also similar across policies. SNAP-BND's mean selector time is
approximately 0.355 seconds, compared with 0.056 seconds for SNAP-W16.

## Verification

`code/verify_core.py` contains automated checks for core import, attention-setting
restoration, saved case-truth reproduction, and compressed-cache generation.

The recorded verification passed all four checks. Its cache-generation smoke
test retained 48 positions; that is a separate functional test, not the
96-position setting used in the fresh-seed benchmark.

## Project files

- `code/recallguard_core.py` — model loading, selectors, baselines, and evaluation helpers.
- `code/verify_core.py` — automated core verification checks.
- `results/fresh_seed_stretch_comparison.csv` — per-case benchmark results.
- `results/fresh_seed_stretch_summary.csv` — aggregate accuracy, cache, and timing metrics.
- `results/fresh_seed_paired_comparisons.csv` — paired strict-correctness comparisons.
- `results/fresh_seed_snap_ablation.csv` — SNAP-BND comparisons against the other methods.
- `verification_report.json` — result of the automated verification.
- `notebooks/RecallGuard-KV-experiment.ipynb` — experiment notebook, when available.

The Colab experiment notebook used for the benchmark is included at `notebooks/RecallGuard-KV-experiment.ipynb`. It contains experiment cells; review the notebook before rerunning resource-intensive cells.

## Reproducibility notes

The core file is extracted from the executed Colab notebook. The verification
function expects an already-loaded `model` and `tokenizer`; it is not a
standalone command-line runner. Full benchmark execution requires running the
appropriate notebook cells and can be resource-intensive.

## Limitations

- The benchmark is small, synthetic, and uses one 0.5B model.
- FULL itself is imperfect on these recall cases.
- The confidence intervals are wide and overlap for SNAP-BND and FULL.
- SNAP-BND is question-aware; Recent-96 and the streaming-style baseline are
  question-blind.
- The reported cache-size savings should not be presented as peak VRAM savings.
- Results support conclusions about this benchmark only, not general model quality.
