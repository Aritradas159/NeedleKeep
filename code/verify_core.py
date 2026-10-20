
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import json
import pandas as pd
import torch


def run_verification(model, tokenizer, repo_path):
    repo = Path(repo_path)
    checks = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append({
                "test": name,
                "status": "PASS",
                "detail": detail
            })
            print("PASS:", name, "-", detail)
            return detail
        except Exception as exc:
            checks.append({
                "test": name,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}"
            })
            print("FAIL:", name, "-", type(exc).__name__, str(exc))
            return None

    core_path = repo / "code" / "recallguard_core.py"
    if not core_path.exists():
        raise FileNotFoundError(f"Missing {core_path}")

    def import_core():
        spec = importlib.util.spec_from_file_location(
            "recallguard_core_automated_test", core_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.model = model
        module.tokenizer = tokenizer
        module.device = next(model.parameters()).device
        return module

    core = check("Core module import", import_core)
    if core is None:
        raise RuntimeError("Core import failed; see error above.")

    def check_context_manager():
        class DummyModel:
            def __init__(self):
                self.config = SimpleNamespace(
                    _attn_implementation="sdpa"
                )

            def set_attn_implementation(self, value):
                self.config._attn_implementation = value

        dummy = DummyModel()
        with core.eager_attn(dummy):
            assert dummy.config._attn_implementation == "eager"
        assert dummy.config._attn_implementation == "sdpa"

        try:
            with core.eager_attn(dummy):
                raise ValueError("intentional restoration test")
        except ValueError:
            pass

        assert dummy.config._attn_implementation == "sdpa"
        return "enters, exits, and restores after exceptions"

    check("eager_attn context manager", check_context_manager)

    def reproduce_cases():
        csv_path = repo / "results" / "fresh_seed_stretch_comparison.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing current benchmark: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"policy", "n_sent", "seed", "qtype", "truth"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Benchmark missing columns: {sorted(missing)}")

        full = df[df["policy"].astype(str).str.lower().eq("full")]

        if len(full) != 24:
            raise AssertionError(
                f"Expected 24 FULL reference records; found {len(full)}"
            )

        matches = 0
        mismatches = []

        for (n_sent, seed), group in full.groupby(["n_sent", "seed"]):
            case = core.make_case(int(seed), int(n_sent))
            truths = {str(qt): str(truth) for qt, _, truth in case["qa"]}

            for row in group.itertuples():
                expected = str(row.truth)
                actual = truths.get(str(row.qtype))

                if actual == expected:
                    matches += 1
                else:
                    mismatches.append({
                        "seed": int(seed),
                        "n_sent": int(n_sent),
                        "qtype": str(row.qtype),
                        "expected": expected,
                        "actual": actual
                    })

        if matches != len(full) or mismatches:
            raise AssertionError(
                f"Matched {matches}/{len(full)}; "
                f"mismatches={mismatches[:3]}"
            )

        return (
            f"{matches}/{len(full)} FULL reference truths reproduced "
            f"from {csv_path.name}"
        )

    check("Reproduce saved case truths", reproduce_cases)

    def selector_and_generation():
        required_functions = [
            "select_snap_span_indices",
            "_compress_by_indices"
        ]
        missing = [n for n in required_functions
                   if not callable(getattr(core, n, None))]
        if missing:
            raise AttributeError(f"Missing functions: {missing}")

        device = next(model.parameters()).device
        model.eval()

        lines = [
            f"Operational report {i}: the team reviewed inventory, "
            f"shipment schedules, supplier records, and maintenance "
            f"procedures for region {i % 9}."
            for i in range(80)
        ]
        lines[38] = (
            "The approved budget for Project Osprey is 4.2 million dollars."
        )
        context = "\n".join(lines)
        question = "What is the approved budget for Project Osprey?"

        input_ids = tokenizer(
            context, return_tensors="pt"
        ).input_ids.to(device)
        q_ids = tokenizer(
            "\nQuestion: " + question + "\nAnswer:",
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(device)

        with torch.inference_mode():
            full_cache = model(
                input_ids=input_ids, use_cache=True
            ).past_key_values

            original_length = full_cache.layers[0].keys.shape[2]
            original_key = full_cache.layers[0].keys.clone()
            original_attention = getattr(
                model.config, "_attn_implementation", None
            )

            if original_length <= 48:
                raise AssertionError(
                    f"Need >48 context positions; got {original_length}"
                )

            indices = core.select_snap_span_indices(
                full_cache, model, q_ids,
                start_keep=8, num_mid_spans=3,
                span_size=8, recent_keep=16, pool=5
            )
            indices_valid = (
                len(indices) == 48
                and len(set(indices)) == 48
                and indices == sorted(indices)
                and min(indices) >= 0
                and max(indices) < original_length
            )
            if not indices_valid:
                raise AssertionError("Selector did not return 48 valid indices")

            compressed = core._compress_by_indices(full_cache, indices)
            retained = compressed.layers[0].keys.shape[2]
            if retained != 48:
                raise AssertionError(f"Compressed cache has {retained} positions")
            if not torch.equal(full_cache.layers[0].keys, original_key):
                raise AssertionError("Original cache was modified")
            if getattr(model.config, "_attn_implementation", None) != original_attention:
                raise AssertionError("Attention implementation was not restored")

            # Test question processing and one generated token.
            working_cache = copy.deepcopy(compressed)
            q_len = q_ids.shape[1]
            out = model(
                input_ids=q_ids,
                past_key_values=working_cache,
                use_cache=True,
                attention_mask=torch.ones(
                    1, retained + q_len, dtype=torch.long, device=device
                ),
                position_ids=torch.arange(
                    original_length, original_length + q_len,
                    device=device
                ).unsqueeze(0)
            )
            after_question = out.past_key_values.layers[0].keys.shape[2]
            if after_question != retained + q_len:
                raise AssertionError(
                    f"After-question cache={after_question}, "
                    f"expected={retained + q_len}"
                )

            next_token = out.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            next_position = original_length + q_len
            out2 = model(
                input_ids=next_token,
                past_key_values=out.past_key_values,
                use_cache=True,
                attention_mask=torch.ones(
                    1, after_question + 1, dtype=torch.long, device=device
                ),
                position_ids=torch.tensor(
                    [[next_position]], dtype=torch.long, device=device
                )
            )
            after_token = out2.past_key_values.layers[0].keys.shape[2]
            if after_token != after_question + 1:
                raise AssertionError(
                    f"After-generation cache={after_token}; "
                    f"expected={after_question + 1}"
                )
            if not torch.isfinite(out.logits).all().item():
                raise AssertionError("Question logits are non-finite")
            if not torch.isfinite(out2.logits).all().item():
                raise AssertionError("Generation logits are non-finite")

        return {
            "original_positions": original_length,
            "retained_positions": retained,
            "question_tokens": q_len,
            "cache_after_question": after_question,
            "cache_after_one_token": after_token,
            "original_cache_unchanged": True,
            "attention_setting_restored": True
        }

    check("SNAP selector and compressed-cache generation",
          selector_and_generation)

    report = {
        "report": "RecallGuard-KV automated core verification",
        "checks": checks,
        "passed": all(c["status"] == "PASS" for c in checks)
    }
    report_path = repo / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print("\nReport saved to:", report_path)
    print(
        "AUTOMATED_CORE_VERIFICATION_PASSED"
        if report["passed"]
        else "AUTOMATED_CORE_VERIFICATION_FAILED"
    )
    return report
