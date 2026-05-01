"""Phase 3: Mechanism Refinement

Test the best-performing variants from Phase 2 on ALL 8 tasks.
Compare refined mechanism vs standard vs full Rossler.
"""

import mlx.core as mx
import numpy as np
from attention import ATTENTION_VARIANTS, StandardAttention, RosslerAttention
from engine import run_experiment, DEFAULT_CONFIG
from tasks import ALL_TASKS


def run_refinement(best_variant_names):
    """Run the best variants from Phase 2 on all 8 tasks.

    Args:
        best_variant_names: list of variant names to test (from Phase 2 winners)
    """
    # Always include standard and rossler for comparison
    variants_to_test = ["standard"] + list(set(best_variant_names)) + ["rossler"]
    # Deduplicate while preserving order
    seen = set()
    unique_variants = []
    for v in variants_to_test:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)

    print("=" * 70)
    print("  PHASE 3 — Mechanism Refinement")
    print(f"  Testing {unique_variants} on all 8 tasks")
    print("=" * 70)

    # Results: task -> variant -> result
    all_results = {}

    for task_name, task_fn in ALL_TASKS.items():
        train_seqs, train_masks, info = task_fn(2048, seed=42)
        test_seqs, test_masks, _ = task_fn(512, seed=123)

        train_x = mx.array(train_seqs[:, :-1])
        train_y = mx.array(train_seqs[:, 1:])
        test_x = mx.array(test_seqs[:, :-1])
        test_y = mx.array(test_seqs[:, 1:])
        target_masks = {k: v[:, 1:] for k, v in test_masks.items()}
        cfg = {**DEFAULT_CONFIG, "vocab_size": info["vocab_size"]}
        crit = info["critical_mask"]

        task_results = {}
        for vn in unique_variants:
            attn_cls = ATTENTION_VARIANTS[vn]
            r = run_experiment(
                f"{vn} / {task_name}", attn_cls, vn,
                train_x, train_y, test_x, test_y, target_masks, config=cfg,
            )
            r["critical_mask"] = crit
            task_results[vn] = r

        all_results[task_name] = task_results

    # Full comparison table
    print(f"\n{'=' * 70}")
    print(f"  REFINEMENT RESULTS — Critical accuracy across all 8 tasks")
    print(f"{'=' * 70}")

    task_names = list(ALL_TASKS.keys())

    # Header
    header = f"\n  {'Task':<22}"
    for vn in unique_variants:
        header += f" {vn[:12]:>12}"
    print(header)
    print(f"  {'-' * (22 + 13 * len(unique_variants))}")

    # Per-task results
    wins = {vn: 0 for vn in unique_variants}
    totals = {vn: 0.0 for vn in unique_variants}

    for tn in task_names:
        crit = all_results[tn][unique_variants[0]]["critical_mask"]
        row = f"  {tn:<22}"
        best_acc = -1
        best_vn = ""
        for vn in unique_variants:
            acc = all_results[tn][vn]["acc"].get(crit, 0)
            totals[vn] += acc
            row += f" {acc:>12.4f}"
            if acc > best_acc:
                best_acc = acc
                best_vn = vn
        wins[best_vn] += 1
        print(row + f"  <- {best_vn}")

    # Aggregates
    print(f"\n  {'AGGREGATE':<22}", end="")
    for vn in unique_variants:
        print(f" {totals[vn] / len(task_names):>12.4f}", end="")
    print("  (mean)")

    print(f"  {'WINS':<22}", end="")
    for vn in unique_variants:
        print(f" {wins[vn]:>12d}", end="")
    print("  (tasks won)")

    # Verdict
    print(f"\n  {'─' * 60}")
    best_overall = max(unique_variants, key=lambda v: totals[v])
    std_mean = totals["standard"] / len(task_names)
    best_mean = totals[best_overall] / len(task_names)
    delta = best_mean - std_mean

    if best_overall == "standard":
        print(f"  Verdict: Standard attention wins overall. No Rossler component provides")
        print(f"  consistent improvement across the task battery.")
    elif delta > 0.02:
        print(f"  Verdict: {best_overall} outperforms standard by {delta:+.4f} mean accuracy.")
        print(f"  This is the recommended mechanism for further development.")
    else:
        print(f"  Verdict: {best_overall} is marginally better ({delta:+.4f}) but not decisive.")

    return all_results, wins, totals, unique_variants


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        best = sys.argv[1:]
    else:
        best = ["fold_differential", "differential"]
        print("  No variants specified, defaulting to fold_differential + differential")
    run_refinement(best)
