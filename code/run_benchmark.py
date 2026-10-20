#!/usr/bin/env python3
"""Re-run RecallGuard-KV's 120-case fresh-seed benchmark.

This recreates the evaluation protocol in the project notebook. It does not
overwrite the shipped reference results: by default, new outputs go to
<project-root>/results_reproduced/.

Requires a CUDA runtime compatible with the installed PyTorch build and the
packages listed in requirements.txt. Use --dry-run to inspect the plan without
loading the model or running inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "code" / "recallguard_core.py"
MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
POLICIES = ("full", "snap_bnd", "snap_w16", "recent", "stream4")
QUERY_TYPES = ("id", "mgr", "bud")
KEYS = ("n_sent", "seed", "qtype")
RAW_COLUMNS = [
    "n_sent", "seed", "qtype", "policy", "budget", "L", "exact", "strict",
    "id_kept", "n_pos_pre", "kv_mib_pre", "selector_s", "prefill_s",
    "continue_s", "peak_delta_mib", "answer", "truth",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def import_core():
    import importlib.util

    require(CORE_PATH.is_file(), f"Core module not found: {CORE_PATH}")
    spec = importlib.util.spec_from_file_location("recallguard_core_runner", CORE_PATH)
    require(spec is not None and spec.loader is not None, "Could not create core import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "make_case", "eval_case", "select_snap_span_indices", "snap_fn",
        "boundary_fn", "recent_indices", "streaming_indices", "pat", "first_line",
        "wilson", "_compress_by_indices",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    require(not missing, f"Core module missing required callables: {missing}")
    return module


def case_keys(seeds: list[int], lengths: list[int]) -> list[tuple[int, int, str]]:
    out = []
    for n_sent in lengths:
        for seed in seeds:
            for qtype in QUERY_TYPES:
                out.append((int(n_sent), int(seed), qtype))
    return out


def validate_case_truths(core, reference_path: Path, seeds: list[int], lengths: list[int]) -> None:
    """If the shipped comparison exists, make sure its truth labels match generator output."""
    if not reference_path.is_file():
        print("NOTE: reference CSV not found; skipping comparison-truth cross-check.")
        return
    df = pd.read_csv(reference_path)
    if not {"policy", "n_sent", "seed", "qtype", "truth"}.issubset(df.columns):
        print("NOTE: reference CSV lacks truth-check columns; skipping cross-check.")
        return
    full = df[df["policy"].astype(str).str.lower().eq("full")]
    matches = 0
    mismatches = []
    for (n_sent, seed), group in full.groupby(["n_sent", "seed"]):
        if int(seed) not in seeds or int(n_sent) not in lengths:
            continue
        case = core.make_case(int(seed), int(n_sent))
        truths = {str(qt): str(truth) for qt, _, truth in case["qa"]}
        for row in group.itertuples():
            actual = truths.get(str(row.qtype))
            expected = str(row.truth)
            if actual == expected:
                matches += 1
            else:
                mismatches.append((int(n_sent), int(seed), str(row.qtype), expected, actual))
    expected_n = len(seeds) * len(lengths) * len(QUERY_TYPES)
    require(not mismatches and matches == expected_n,
            f"Reference truth validation failed: {matches}/{expected_n}; examples={mismatches[:3]}")
    print(f"PASS: reference truth labels reproduce ({matches}/{expected_n})")


def save_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def parse_existing(raw_path: Path, case_set: set[tuple[int, int, str]], budget: int) -> tuple[list[dict[str, Any]], set[tuple[int, int, str, str, int]]]:
    if not raw_path.exists():
        return [], set()
    frame = pd.read_csv(raw_path)
    missing = set(RAW_COLUMNS) - set(frame.columns)
    require(not missing, f"Existing output CSV is incompatible; missing columns: {sorted(missing)}")
    rows = frame[RAW_COLUMNS].to_dict("records")
    done = set()
    for row in rows:
        n_sent, seed, qtype = int(row["n_sent"]), int(row["seed"]), str(row["qtype"])
        policy, row_budget = str(row["policy"]), int(row["budget"])
        key = (n_sent, seed, qtype, policy, row_budget)
        require((n_sent, seed, qtype) in case_set, f"Unexpected case in existing output: {key}")
        require(policy in POLICIES, f"Unexpected policy in existing output: {policy}")
        require(key not in done, f"Duplicate result key in existing output: {key}")
        require(row_budget == (0 if policy == "full" else budget), f"Unexpected budget in existing output: {key}")
        done.add(key)
    return rows, done


def run_benchmark(core, model, tokenizer, device, out_dir: Path, seeds: list[int], lengths: list[int], budget: int, n_new: int, model_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "fresh_seed_stretch_comparison.csv"
    case_list = case_keys(seeds, lengths)
    case_set = set(case_list)
    expected_rows = len(case_list) * len(POLICIES)
    rows, done = parse_existing(raw_path, case_set, budget)
    print(f"Output directory: {out_dir}")
    print(f"Previously saved rows found: {len(done)} / {expected_rows}; remaining rows will be evaluated.")

    # Keep module globals in sync: the policy helpers in the extracted core use them.
    core.model = model
    core.tokenizer = tokenizer
    core.device = device
    model.eval()

    for n_sent in lengths:
        for seed in seeds:
            case = core.make_case(seed, n_sent)
            for qtype, question, truth in case["qa"]:
                q_ids = tokenizer(
                    "\nQuestion: " + question + "\nAnswer:",
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(device)

                # FULL is the uncompressed reference. All compressed methods use
                # the same 96-position budget and the notebook's absolute position mode.
                jobs = [
                    ("full", 0, None, "default"),
                    ("snap_bnd", budget, core.boundary_fn(budget, q_ids, case["ctx"]), "absolute"),
                    ("snap_w16", budget, core.snap_fn(budget, 16, q_ids), "absolute"),
                    ("recent", budget,
                     lambda cache, B=budget: core.recent_indices(cache.layers[0].keys.shape[2], B),
                     "absolute"),
                    ("stream4", budget,
                     lambda cache, B=budget: core.streaming_indices(cache.layers[0].keys.shape[2], 4, B),
                     "absolute"),
                ]

                for policy, row_budget, policy_fn, mode in jobs:
                    key = (n_sent, seed, qtype, policy, row_budget)
                    if key in done:
                        continue
                    print(f"RUN {len(done) + 1}/{expected_rows}: n_sent={n_sent}, seed={seed}, qtype={qtype}, policy={policy}", flush=True)
                    try:
                        result = core.eval_case(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            ctx=case["ctx"],
                            question=question,
                            ident=truth,
                            policy_fn=policy_fn,
                            mode=mode,
                            n_new=n_new,
                        )
                        answer = str(result["answer"])
                        line = core.first_line(answer)
                        exact = core.pat(str(truth)).search(line) is not None
                        strict = exact and not any(
                            core.pat(str(decoy)).search(line) is not None
                            for decoy in case["decoys"][qtype]
                        )
                        row = {
                            "n_sent": int(n_sent),
                            "seed": int(seed),
                            "qtype": str(qtype),
                            "policy": policy,
                            "budget": int(row_budget),
                            "L": int(result["L"]),
                            "exact": int(exact),
                            "strict": int(strict),
                            "id_kept": float(result["id_kept"]),
                            "n_pos_pre": int(result["n_pos_pre"]),
                            "kv_mib_pre": float(result["kv_mib_pre"]),
                            "selector_s": float(result["selector_s"]),
                            "prefill_s": float(result["prefill_s"]),
                            "continue_s": float(result["continue_s"]),
                            "peak_delta_mib": float(result["peak_delta_mib"]),
                            "answer": answer[:60],
                            "truth": str(truth),
                        }
                        rows.append(row)
                        done.add(key)
                        save_csv_atomic(pd.DataFrame(rows, columns=RAW_COLUMNS), raw_path)
                    except Exception:
                        # Persist all completed cases before surfacing the failure.
                        save_csv_atomic(pd.DataFrame(rows, columns=RAW_COLUMNS), raw_path)
                        raise
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            print(f"Completed context: n_sent={n_sent}, seed={seed}; saved rows={len(done)}/{expected_rows}", flush=True)

    require(len(done) == expected_rows, f"Incomplete benchmark: {len(done)}/{expected_rows} rows saved")
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df = df.sort_values(["n_sent", "seed", "qtype", "policy"], kind="stable").reset_index(drop=True)
    save_csv_atomic(df, raw_path)
    write_aggregate_files(df, out_dir)
    write_config(out_dir, model_name, seeds, lengths, budget, n_new, device)
    print("\nBENCHMARK COMPLETE")
    print(f"Raw rows: {len(df)}")
    print(f"Summary: {out_dir / 'fresh_seed_stretch_summary.csv'}")
    print(f"Paired comparisons: {out_dir / 'fresh_seed_paired_comparisons.csv'}")
    print(f"SNAP ablation: {out_dir / 'fresh_seed_snap_ablation.csv'}")
    print(f"Raw results: {raw_path}")


def write_aggregate_files(df: pd.DataFrame, out_dir: Path) -> None:
    require(len(df) == 120, f"Expected 120 raw records, got {len(df)}")
    expected_counts = df["policy"].value_counts().to_dict()
    require(all(expected_counts.get(p, 0) == 24 for p in POLICIES), f"Unexpected policy counts: {expected_counts}")

    full = df[df["policy"] == "full"]
    full_kv = {tuple(getattr(row, k) for k in KEYS): float(row.kv_mib_pre) for row in full.itertuples()}
    summary_rows = []
    for policy in POLICIES:
        part = df[df["policy"] == policy]
        n = len(part)
        strict_correct = int(part["strict"].sum())
        exact_correct = int(part["exact"].sum())
        p = strict_correct / n
        z = 1.96
        den = 1 + z*z/n
        center = (p + z*z/(2*n)) / den
        half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
        if policy == "full":
            saved_pct = 0.0
        else:
            saved = []
            for row in part.itertuples():
                key = tuple(getattr(row, k) for k in KEYS)
                saved.append(100 * (1 - float(row.kv_mib_pre) / full_kv[key]))
            saved_pct = sum(saved) / len(saved)
        summary_rows.append({
            "policy": policy,
            "cases": n,
            "strict_correct": strict_correct,
            "strict_accuracy_pct": round(100*p, 2),
            "strict_95CI_low": round(100*(center-half), 2),
            "strict_95CI_high": round(100*(center+half), 2),
            "exact_correct": exact_correct,
            "exact_accuracy_pct": round(100*exact_correct/n, 2),
            "mean_positions": round(float(part["n_pos_pre"].mean()), 1),
            "mean_kv_mib": round(float(part["kv_mib_pre"].mean()), 3),
            "mean_kv_saved_vs_full_pct": round(saved_pct, 2),
            "mean_selector_seconds": round(float(part["selector_s"].mean()), 3),
            "mean_continuation_seconds": round(float(part["continue_s"].mean()), 3),
            "mean_prefill_seconds": round(float(part["prefill_s"].mean()), 3),
            "mean_peak_delta_mib": round(float(part["peak_delta_mib"].mean()), 3),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "fresh_seed_stretch_summary.csv", index=False)

    lookup = {
        policy: {tuple(getattr(row, k) for k in KEYS): int(row.strict) for row in df[df["policy"] == policy].itertuples()}
        for policy in POLICIES
    }
    paired_rows = []
    for method in POLICIES[1:]:
        wins = losses = both = neither = 0
        for key, value in lookup[method].items():
            full_value = lookup["full"][key]
            if value and full_value:
                both += 1
            elif value and not full_value:
                wins += 1
            elif not value and full_value:
                losses += 1
            else:
                neither += 1
        paired_rows.append({
            "method": method,
            "paired_cases": 24,
            "method_correct_full_wrong": wins,
            "full_correct_method_wrong": losses,
            "both_correct": both,
            "both_wrong": neither,
        })
    pd.DataFrame(paired_rows).to_csv(out_dir / "fresh_seed_paired_comparisons.csv", index=False)

    ablation_rows = []
    for other in ("snap_w16", "recent", "stream4", "full"):
        a, b = lookup["snap_bnd"], lookup[other]
        snap_only = sum(a[k] == 1 and b[k] == 0 for k in a)
        other_only = sum(a[k] == 0 and b[k] == 1 for k in a)
        both = sum(a[k] == 1 and b[k] == 1 for k in a)
        neither = sum(a[k] == 0 and b[k] == 0 for k in a)
        acc_diff = 100 * (sum(a.values()) - sum(b.values())) / 24
        ablation_rows.append({
            "comparison": f"snap_bnd vs {other}",
            "paired_cases": 24,
            "snap_bnd_only_correct": snap_only,
            "other_only_correct": other_only,
            "both_correct": both,
            "both_wrong": neither,
            "accuracy_difference_pp": round(acc_diff, 2),
        })
    pd.DataFrame(ablation_rows).to_csv(out_dir / "fresh_seed_snap_ablation.csv", index=False)


def write_config(out_dir: Path, model_name: str, seeds: list[int], lengths: list[int], budget: int, n_new: int, device) -> None:
    config = {
        "benchmark": "RecallGuard-KV fresh-seed stretch comparison",
        "model": model_name,
        "seeds": seeds,
        "context_sentence_counts": lengths,
        "query_types": list(QUERY_TYPES),
        "policies": list(POLICIES),
        "compressed_budget_positions": budget,
        "full_budget": 0,
        "max_new_tokens": n_new,
        "scoring": "boundary-aware regex on first answer line; strict excludes known same-question decoy values on that line",
        "device": str(device),
        "torch_version": torch.__version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_records": len(seeds) * len(lengths) * len(QUERY_TYPES) * len(POLICIES),
        "note": "A new inference run may vary slightly across hardware or software revisions; this records the same protocol, not a bitwise guarantee.",
    }
    try:
        import transformers
        config["transformers_version"] = transformers.__version__
    except Exception:
        config["transformers_version"] = "unknown"
    if torch.cuda.is_available():
        config["gpu_name"] = torch.cuda.get_device_name(0)
    (out_dir / "benchmark_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT, help="Hugging Face model ID (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[3000, 3001, 3002, 3003])
    parser.add_argument("--lengths", nargs="+", type=int, default=[112, 224])
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--n-new", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_reproduced",
                        help="Separate output directory; shipped reference results are not overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and print the run plan without loading the model")
    args = parser.parse_args()

    require(args.budget == 96, "This script is configured for the documented 96-position benchmark; do not silently change the budget.")
    require(args.n_new == 40, "The documented scoring protocol uses max 40 new tokens; change only if intentionally designing a different benchmark.")
    require(len(set(args.seeds)) == len(args.seeds), "Seeds contain duplicates")
    require(len(set(args.lengths)) == len(args.lengths), "Context lengths contain duplicates")
    require(all(x > 0 for x in args.seeds) and all(x >= 10 for x in args.lengths), "Invalid seed or context length")

    core = import_core()
    cases = case_keys(args.seeds, args.lengths)
    expected_rows = len(cases) * len(POLICIES)
    require(expected_rows > 0, "No benchmark cases requested")
    out_dir = args.output_dir.expanduser().resolve()
    print("RecallGuard-KV reproducible benchmark runner")
    print(f"Project root: {ROOT}")
    print(f"Model: {args.model}")
    print(f"Seeds: {args.seeds}")
    print(f"Context sentence counts: {args.lengths}")
    print(f"Query types: {list(QUERY_TYPES)}")
    print(f"Policies: {list(POLICIES)}")
    print(f"Compressed budget: {args.budget}; FULL retains full context")
    print(f"Max new tokens: {args.n_new}")
    print(f"Expected records: {expected_rows}")
    print(f"Output directory: {out_dir}")

    validate_case_truths(core, ROOT / "results" / "fresh_seed_stretch_comparison.csv", args.seeds, args.lengths)

    # Ensure generator and row schema produce the expected unique case-policy keys.
    planned = [(n, s, q, p) for n, s, q in cases for p in POLICIES]
    require(len(planned) == len(set(planned)), "Duplicate planned cases")
    require(all(q in QUERY_TYPES and p in POLICIES for _, _, q, p in planned), "Invalid planned key")
    print("PASS: planned case-policy keys are unique")

    if args.dry_run:
        print("DRY RUN ONLY: model not loaded, no inference performed, no benchmark CSV written.")
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. In Colab, choose Runtime > Change runtime type > T4 GPU (or another available GPU), then rerun.")

    print("Loading model; this may download model files if they are not cached...", flush=True)
    model, tokenizer, device = core.load_model(args.model)
    model.eval()
    print(f"Model loaded on {device}. Starting benchmark; each completed row is saved for resume.", flush=True)
    run_benchmark(core, model, tokenizer, device, out_dir, args.seeds, args.lengths, args.budget, args.n_new, args.model)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BENCHMARK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
