"""Phase 1: Failure Mode Discovery

Run standard attention on all 8 synthetic tasks.
Rank by critical-mask accuracy to find where standard attention breaks.
Tasks with critical accuracy < 85% are Phase 2 targets.
"""

import mlx.core as mx
import numpy as np
from attention import StandardAttention
from engine import run_experiment, DEFAULT_CONFIG
from tasks import ALL_TASKS


def main():
    print("=" * 70)
    print("  PHASE 1 — Failure Mode Discovery")
    print("  Standard attention on all 8 tasks")
    print("=" * 70)

    results = []

    for task_name, task_fn in ALL_TASKS.items():
        print(f"\n{'─' * 70}")
        print(f"  Task: {task_name}")
        print(f"{'─' * 70}")

        train_seqs, train_masks, info = task_fn(2048, seed=42)
        test_seqs, test_masks, _ = task_fn(512, seed=123)

        train_x = mx.array(train_seqs[:, :-1])
        train_y = mx.array(train_seqs[:, 1:])
        test_x = mx.array(test_seqs[:, :-1])
        test_y = mx.array(test_seqs[:, 1:])
        target_masks = {k: v[:, 1:] for k, v in test_masks.items()}

        cfg = {**DEFAULT_CONFIG, "vocab_size": info["vocab_size"]}

        r = run_experiment(
            f"Standard / {task_name}", StandardAttention, "standard",
            train_x, train_y, test_x, test_y, target_masks, config=cfg,
        )
        r["task_name"] = task_name
        r["critical_mask"] = info["critical_mask"]
        r["task_description"] = info["description"]
        results.append(r)

    # Ranked table
    print(f"\n{'=' * 70}")
    print(f"  PHASE 1 RESULTS — Ranked by critical accuracy (hardest first)")
    print(f"{'=' * 70}")
    print(f"\n  {'Task':<25} {'Overall':>8} {'Critical':>10} {'Crit Mask':>15} {'Time':>6}")
    print(f"  {'-' * 65}")

    ranked = sorted(results, key=lambda r: r["acc"].get(r["critical_mask"], 0))
    hard_tasks = []

    for r in ranked:
        crit = r["acc"].get(r["critical_mask"], 0)
        marker = " <-- HARD" if crit < 0.85 else ""
        print(f"  {r['task_name']:<25} {r['acc']['overall']:>8.4f} {crit:>10.4f} {r['critical_mask']:>15} {r['train_time']:>5.0f}s{marker}")
        if crit < 0.85:
            hard_tasks.append(r["task_name"])

    print(f"\n  Hard tasks (critical < 85%): {hard_tasks if hard_tasks else 'NONE'}")
    print(f"  These are the Phase 2 targets.")
    print()

    return results, hard_tasks


if __name__ == "__main__":
    main()
