"""Master runner: execute all 4 experimental phases and write results."""

import sys
import time
import os

def main():
    t_start = time.time()

    # ── Failure Mode Discovery ──────────────────────────────
    print("\n" + "=" * 70)
    print("  STARTING FULL EXPERIMENTAL PIPELINE")
    print("=" * 70 + "\n")

    from phase1_discovery import main as phase1_main
    results_p1, hard_tasks = phase1_main()

    if not hard_tasks:
        print("\n  WARNING: No hard tasks found (all > 85% critical accuracy).")
        print("  Lowering threshold to 95% and retrying...")
        hard_tasks = [r["task_name"] for r in results_p1
                      if r["acc"].get(r["critical_mask"], 1.0) < 0.95]
        if not hard_tasks:
            print("  Still no hard tasks. Using all tasks for ablation.")
            hard_tasks = [r["task_name"] for r in results_p1]

    print(f"\n  Hard tasks for ablation: {hard_tasks}")
    p1_time = time.time() - t_start

    # ── Component Ablation ──────────────────────────────────
    t2 = time.time()
    from phase2_ablation import run_ablation
    matrix_p2, component_wins, best_variants = run_ablation(hard_tasks)
    p2_time = time.time() - t2

    # Determine which variants to carry forward
    # Pick variants that won on at least 1 task, plus fold_differential and differential
    carry_forward = set()
    for vn, wins in component_wins.items():
        if wins > 0:
            carry_forward.add(vn)
    carry_forward.add("fold_differential")
    carry_forward.add("differential")
    carry_forward = list(carry_forward)
    print(f"\n  Carrying forward to refinement: {carry_forward}")

    # ── Mechanism Refinement ────────────────────────────────
    t3 = time.time()
    from phase3_refinement import run_refinement
    results_p3, wins_p3, totals_p3, variants_p3 = run_refinement(carry_forward)
    p3_time = time.time() - t3

    # Determine best variant for scale test
    best_non_standard = max(
        [(v, totals_p3[v]) for v in variants_p3 if v != "standard"],
        key=lambda x: x[1]
    )[0]

    # ── Scale Test ──────────────────────────────────────────
    t4 = time.time()
    from phase4_scale import run_scale_test
    scale_variants = ["standard", best_non_standard]
    if best_non_standard != "rossler":
        scale_variants.append("rossler")
    results_p4 = run_scale_test(scale_variants)
    p4_time = time.time() - t4

    total_time = time.time() - t_start

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Failure mode discovery: {p1_time / 60:.1f} min")
    print(f"  Component ablation:     {p2_time / 60:.1f} min")
    print(f"  Mechanism refinement:   {p3_time / 60:.1f} min")
    print(f"  Scale test:             {p4_time / 60:.1f} min")
    print(f"  Total:                  {total_time / 60:.1f} min")
    print()


if __name__ == "__main__":
    main()
