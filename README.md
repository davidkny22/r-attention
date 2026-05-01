# R-Attention: Attention from Rössler Dynamics

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![MLX](https://img.shields.io/badge/MLX-0.31+-orange.svg)](https://ml-explore.github.io/mlx/)

**An attention mechanism derived from the Rössler attractor: values become active participants, attention scores self-regulate through a fold gate, and heads learn where to sit on the periodic-to-chaotic spectrum.**

The Rössler attractor is one of the simplest chaotic systems—three coupled ODEs with a single nonlinear term. That one nonlinearity creates the entire folding structure: a near-2D spiral that periodically ejects trajectories upward, then reinjects them back down. The question was whether those dynamics could be mapped onto attention.

This project came from an X post showing the basic properties of the Rössler equation. I thought about whether it had novel applications for machine learning, and the answer was yes—at least in theory. Three principles were derived from the ODEs and mapped onto attention:

1. **Subtractive coupling** (from `ẋ = -y - z`): Attention scores incorporate a subtractive term from the values. Values can suppress attention, not just receive it.

2. **Self-feedback with tunable gain** (from `ẏ = x + ay`): Keys have a self-reinforcement channel with a learned gain parameter `a`. This controls how focused vs. diffuse the attention pattern is—the "spiral tightness."

3. **Single gated nonlinearity** (from `ż = b + z(x - c)`): Values self-amplify only when queries exceed a learned threshold `c`. This is the fold—the only nonlinearity, the source of all structure. The parameter `c` controls where on the periodic-to-chaotic spectrum each head sits.

A reinjection mechanism completes the analogy: after the fold modifies attention, the output blends fold-modified attention with spiral-only (unmodified) attention, mirroring how Rössler trajectories return to the spiral plane after ejection.

**Result:** The score-based fold formulation is mechanistically sound—genuine sparsity, head specialization through learned thresholds—but no Rössler-derived component consistently outperformed standard attention across an 8-task synthetic battery. See `docs/results/experiment-log.md` for the full experimental story, component ablations, and conclusions.

## Quick Start

```bash
git clone https://github.com/davidkny22/r-attention.git
cd r-attention
pip install -r requirements.txt

# Original validation test (the standalone rossler_attention.py script)
python rossler_attention.py

# Phase 1: Discover where standard attention breaks
python phase1_discovery.py

# Phase 2: Component ablation on hard tasks
python phase2_ablation.py

# Phase 3: Mechanism refinement
python phase3_refinement.py

# Phase 4: Scale test on Shakespeare character LM
python phase4_scale.py

# Or run the full pipeline
python run_all.py
```

## Module Reference

Every module can be used standalone. Below are practical examples for each domain.

### Attention Variants (`attention.py`)

```python
from attention import ATTENTION_VARIANTS, StandardAttention, RosslerAttention

# Standard baseline
attn = StandardAttention(dim=64, num_heads=4)
output = attn(x)

# Full Rossler mechanism (fold + spiral + reinjection)
rossler = RosslerAttention(dim=64, num_heads=4)
output = rossler(x)

# Diagnostics: attention matrix, fold activation, reinjection ratio
diag = rossler.get_diagnostics(x)
print(diag["A"].shape)    # (B, H, T, T)
print(diag["F"].shape)    # (B, H, T, T) fold activation
```

### Training & Evaluation (`engine.py`)

```python
from engine import run_experiment, DEFAULT_CONFIG
from attention import RosslerAttention

# Run a single experiment with full diagnostics
result = run_experiment(
    name="associative_recall",
    attention_cls=RosslerAttention,
    variant_name="rossler",
    train_x=train_x, train_y=train_y,
    test_x=test_x, test_y=test_y,
    target_masks=masks,
    config={"epochs": 200},
)

# Result keys: losses, acc, diag, train_time, param_history, model
print(f"Overall accuracy: {result['acc']['overall']:.4f}")
print(f"Fold sparsity: {result['diag']['fold_sparsity']:.4f}")
print(f"Reinjection ratio: {result['diag']['reinjection_ratio']:.4f}")
```

### Task Generators (`tasks.py`)

```python
from tasks import ALL_TASKS

# Generate data for any of the 8 tasks
task_fn = ALL_TASKS["associative_recall"]
train_seqs, train_masks, info = task_fn(num_seqs=2048, seed=42)

# info contains vocab_size, critical_mask, description
critical = info["critical_mask"]  # which positions test the core capability
```

## How It Works

Standard attention is a passive value retrieval model:

```
A = softmax(QK^T / sqrt(d))
O = A @ V
```

Values have zero influence on where attention goes. The Rössler mapping changes this entirely.

### Spiral Component

From `ẏ = x + ay`: keys reinforce themselves with a learned per-head gain.

```
K_spiral = K + a * (K @ W_self)     # self-feedback on keys
S = Q @ K_spiral^T / sqrt(d)        # spiral scores
```

### Fold Component

From `ż = b + z(x - c)`: value magnitude gates score exceedance.

```
v_energy = ||V_j||                   # how "loud" each value position is
F = v_energy * relu(S - c)          # fold fires where scores exceed threshold
```

This directly mirrors `z(x - c)`: value magnitude (z) multiplies score exceedance (x - c). The fold is genuinely sparse—dormant below threshold, proportional to exceedance above it.

### Coupling and Reinjection

From `ẋ = -y - z`: subtractive combination. Values can suppress attention.

```
A = softmax(S - lambda * F)         # high scores get suppressed by fold
spiral_only = softmax(S) @ V
fold_output = A @ V
O = fold_output + b * spiral_only   # reinjection: blend fold-modified + spiral-only
```

Four learnable scalars per head: `a` (spiral tightness), `b` (reinjection strength), `c` (fold threshold), `lambda` (fold coupling).

## Project Structure

```
r-attention/
  attention.py              # All 6 attention variants (standard, fold-only,
                            #   spiral-only, differential, fold+differential, full Rossler)
  engine.py                 # Model, TransformerBlock, FeedForward, training loop,
                            #   evaluation, diagnostics, experiment runner
  tasks.py                  # All 8 synthetic task generators
  rossler_attention.py      # Original standalone validation script
  phase1_discovery.py       # Run standard attention on all 8 tasks, rank by difficulty
  phase2_ablation.py        # Run 6 variants on hard tasks from Phase 1
  phase3_refinement.py      # Build refined mechanism, test on all 8 tasks
  phase4_scale.py           # Scale test on character-level language modeling
  run_all.py                # Master runner for the full 4-phase pipeline
  data/
    shakespeare.txt         # Tiny Shakespeare corpus for Phase 4
  docs/
    results/
      experiment-log.md     # Full experiment log with all 7 experiments
```

## Installation

```bash
git clone https://github.com/davidkny22/r-attention.git
cd r-attention
pip install -r requirements.txt
```

For development:
```bash
pip install -e .
```

## Related Work

The Rössler attractor was introduced by [Rössler (1976)](https://doi.org/10.1016/0375-9601(76)90101-8) as one of the simplest continuous chaotic systems. The "edge of chaos" principle— that computation is most powerful near the boundary between order and disorder—has been central to reservoir computing since [Jaeger (2001)](https://www.ai.rug.nl/minds/uploads/Jaeger01.pdf) and [Maass, Natschläger & Markram (2002)](https://doi.org/10.1162/089976602760407955). [Singh, Sankaranarayanan & Raman (2025)](https://arxiv.org/abs/2507.18467) recently linked Lyapunov spectra and memory-capacity spectra in Echo-State Networks, formalizing the edge-of-chaos design heuristic. More broadly, [Chen et al. (2018)](https://arxiv.org/abs/1806.07366) formalized neural networks as continuous dynamical systems via Neural ODEs, while [Essex et al. (2026)](https://arxiv.org/abs/2508.10765) performed comprehensive bifurcation analysis of learning Hopfield networks, revealing attractor destruction and catastrophic forgetting via pitchfork cascades. To our knowledge, directly mapping Rössler fold-and-spiral dynamics onto transformer attention scores has not been previously attempted.

## License

[AGPL-3.0-only](LICENSE)

## Citation

```bibtex
@misc{kogan2026rossler,
  author = {Kogan, David},
  title = {R-Attention: Attention from {R}{\"o}ssler Dynamics},
  year = {2026},
  url = {https://github.com/davidkny22/r-attention}
}
```
