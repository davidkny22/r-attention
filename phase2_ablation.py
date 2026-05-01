"""Phase 2: Component Ablation

Run all 6 attention variants on the hard tasks identified by Phase 1.
Identify which components help on which failure modes.
"""

import mlx.core as mx
import numpy as np
from attention import ATTENTION_VARIANTS
from engine import run_experiment, compare_results, DEFAULT_CONFIG
from tasks import ALL_TASKS


def run_ablation(hard_task_names):
    print("=" * 70)
    print("  PHASE 2 — Component Ablation")
    print(f"  6 variants x {len(hard_task_names)} hard tasks")
    print("=" * 70)

    # Results matrix: task -> list of variant results
    matrix = {}

    for task_name in hard_task_names:
        task_fn = ALL_TASKS[task_name]
        train_seqs, train_masks, info = task_fn(2048, seed=42)
        test_seqs, test_masks, _ = task_fn(512, seed=123)

        train_x = mx.array(train_seqs[:, :-1])
        train_y = mx.array(train_seqs[:, 1:])
        test_x = mx.array(test_seqs[:, :-1])
        test_y = mx.array(test_seqs[:, 1:])
        target_masks = {k: v[:, 1:] for k, v in test_masks.items()}
        cfg = {**DEFAULT_CONFIG, "vocab_size": info["vocab_size"]}
        crit = info["critical_mask"]

        task_results = []
        for variant_name, attn_cls in ATTENTION_VARIANTS.items():
            print(f"\n{'─' * 70}")
            print(f"  {task_name} / {variant_name}")
            print(f"{'─' * 70}")

            r = run_experiment(
                f"{variant_name} / {task_name}", attn_cls, variant_name,
                train_x, train_y, test_x, test_y, target_masks, config=cfg,
            )
            r["task_name"] = task_name
            r["critical_mask"] = crit
            task_results.append(r)

        matrix[task_name] = task_results

    # Print comparison matrix
    print(f"\n{'=' * 70}")
    print(f"  ABLATION MATRIX — Critical mask accuracy")
    print(f"{'=' * 70}")

    variant_names = list(ATTENTION_VARIANTS.keys())

    # Header
    header = f"  {'Variant':<22}"
    for tn in hard_task_names:
        header += f" {tn[:14]:>14}"
    print(header)
    print(f"  {'-' * (22 + 15 * len(hard_task_names))}")

    # Standard baseline values
    baselines = {}
    for tn in hard_task_names:
        std_result = [r for r in matrix[tn] if r["variant"] == "standard"][0]
        baselines[tn] = std_result["acc"].get(std_result["critical_mask"], 0)

    for vn in variant_names:
        row = f"  {vn:<22}"
        for tn in hard_task_names:
            result = [r for r in matrix[tn] if r["variant"] == vn][0]
            crit_acc = result["acc"].get(result["critical_mask"], 0)
            row += f" {crit_acc:>14.4f}"
        print(row)

    # Delta vs standard
    print(f"\n  Delta vs standard:")
    print(f"  {'Variant':<22}", end="")
    for tn in hard_task_names:
        print(f" {tn[:14]:>14}", end="")
    print()
    print(f"  {'-' * (22 + 15 * len(hard_task_names))}")

    component_wins = {vn: 0 for vn in variant_names if vn != "standard"}

    for vn in variant_names:
        if vn == "standard":
            continue
        row = f"  {vn:<22}"
        for tn in hard_task_names:
            result = [r for r in matrix[tn] if r["variant"] == vn][0]
            crit_acc = result["acc"].get(result["critical_mask"], 0)
            delta = crit_acc - baselines[tn]
            row += f" {delta:>+14.4f}"
            if delta > 0.01:
                component_wins[vn] += 1
        print(row)

    # Summary
    print(f"\n  Component wins (delta > 1% on any hard task):")
    for vn, wins in sorted(component_wins.items(), key=lambda x: -x[1]):
        print(f"    {vn:<22} {wins}/{len(hard_task_names)} tasks")

    # Find best variant per task
    print(f"\n  Best variant per task:")
    best_variants = {}
    for tn in hard_task_names:
        best = max(matrix[tn], key=lambda r: r["acc"].get(r["critical_mask"], 0))
        best_acc = best["acc"].get(best["critical_mask"], 0)
        print(f"    {tn:<25} {best['variant']:<22} {best_acc:.4f}")
        best_variants[tn] = best["variant"]

    return matrix, component_wins, best_variants


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        hard = sys.argv[1:]
    else:
        hard = list(ALL_TASKS.keys())
        print("  No hard tasks specified, running all tasks.")
    run_ablation(hard)
