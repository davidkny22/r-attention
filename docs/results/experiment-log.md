# R-Attention: Experiment Log

## Origin and Question

The Rössler attractor is one of the simplest chaotic dynamical systems—three coupled ODEs with a single nonlinear term, `z(x - c)`. That one nonlinearity creates the entire folding structure: a near-2D spiral that periodically ejects trajectories upward, then reinjects them back down to the spiral plane. Minimal nonlinearity, maximal structural complexity.

The core question: can the Rössler's dynamics be mapped onto attention to produce a mechanism with properties standard attention lacks?

Standard attention uses a passive value retrieval model—queries match keys, values get weighted and summed. Values have zero influence on where attention goes. The Rössler mapping would change this: values become active participants, attention scores self-regulate through a fold mechanism, and the system self-organizes its capacity allocation through bifurcation dynamics.

Three principles were derived from the Rössler ODEs:

1. **Subtractive coupling** (from `ẋ = -y - z`): Attention scores incorporate a subtractive term from the values. Values can suppress attention, not just receive it.
2. **Self-feedback with tunable gain** (from `ẏ = x + ay`): Keys have a self-reinforcement channel with a learned gain parameter `a`. This controls how focused vs. diffuse the attention pattern is—the "spiral tightness."
3. **Single gated nonlinearity** (from `ż = b + z(x - c)`): Values self-amplify only when queries exceed a learned threshold `c`. This is the fold—the only nonlinearity, the source of all structure. The parameter `c` controls where on the periodic-to-chaotic spectrum each head sits.

The reinjection mechanism completes the analogy: after the fold modifies attention, the output blends fold-modified attention with spiral-only (unmodified) attention, mirroring how Rössler trajectories return to the spiral plane after ejection.

---

## Experiment 1: Initial Mechanism Validation

**Goal:** Determine if the Rössler attention mechanism can train at all, whether heads specialize through different learned `c` values, and whether it outperforms standard attention.

**Setup:** Vocab 16, dim 64, 4 heads, 1 layer, sequence length 256. Synthetic task: periodic sequences (period 3–5) interrupted by random "fold" bursts (3–8 tokens), with the model predicting the next token. Two models trained: standard attention and Rössler attention. MLX on Apple Silicon M1.

**Initial fold formulation:** `gate = sigmoid((Q - c) @ V^T)`, where values gate queries exceeding a threshold. The fold activation F (shape T x T) modifies attention scores: `A = softmax(S - lambda * F)`. The reinjection blends fold-modified and spiral-only outputs: `O = A @ V + b * softmax(S) @ V`.

**Results at 50 epochs:**

Both models achieved identical performance—23% accuracy (above random 6.25% but not strong). The Rössler mechanism showed no advantage. But the learned parameters told a clear story:

| Parameter | Init | Learned | Interpretation |
|-----------|------|---------|----------------|
| a (spiral) | 0.1 | ~0.01 | Self-feedback killed |
| b (reinjection) | 0.01 | ~-0.12 | Reinjection inverted (subtracted, not added) |
| c (threshold) | 1.0 | ~1.14 | Slight threshold shift |
| lambda (coupling) | 0.1 | ~0.16 | Fold coupling slightly strengthened |

The network dismantled three of four Rössler components. It killed the spiral, inverted the reinjection, and kept the fold but could not extract useful signal from it.

**At 200 epochs:** Same story, more pronounced. Both models converged to identical loss (~1.62). The spiral (a) went to ~0 or negative. The reinjection (b) went to -0.23. The network was actively subtracting the spiral-only output rather than using it as intended.

**Root cause analysis:** The fold gate `sigmoid((Q - c) @ V^T)` produced near-uniform values across positions. The dot product over d=16 dimensions averaged out to moderate values where sigmoid ≈ 0.5. Softmax is invariant to constant shifts, so `softmax(S - lambda * 0.5) = softmax(S - constant) = softmax(S)`. The fold was doing nothing.

The deeper issue: the fold was disconnected from the attention scores. In the Rössler, the fold monitors the system's own state (x) and fires when it crosses a threshold. In the initial formulation, the fold was a separate Q-V computation with no relationship to the attention scores.

### Decision Log

**Decision:** Redesign the fold to be a function of the attention scores themselves, not a separate computation. Change from Q-V gating to score-based activation: `F = v_energy * relu(S - c)`.

**Rationale:** The current fold is disconnected from the attention scores and produces uniform activations. A score-based fold directly mirrors the Rössler ODE structure—value magnitude (z) multiplies score exceedance (x - c)—and creates genuine sparsity. This also changes `c` from operating on raw query elements to operating on attention scores, which requires reinitializing from 1.0 to 0.0.

---

## Experiment 2: Score-Based Fold with Value Weighting

**Goal:** Test whether the score-based fold formulation trains, whether heads specialize, and whether accuracy improves.

**Setup:** Same task and model config as Experiment 1. The fold now uses `F = v_energy * relu(S - c)`, where `v_energy = ||V_j||` and `c` is initialized to 0.0.

**Results at 200 epochs:**

The mechanism is qualitatively alive now. Compare the learned parameters:

| Parameter | V1 (old fold) | V2 (score-based fold) | Interpretation |
|-----------|--------------|----------------------|----------------|
| a | ~0.00 (killed) | -0.17 to +0.07 (varied) | Spiral being used |
| b | -0.18 (inverted) | +0.03 to +0.07 (positive) | Reinjection working as intended |
| c | 1.17 (uniform) | -0.51 to +0.13 (spread) | Real bifurcation specialization |
| lambda | 0.28 (uniform) | 0.16 to 0.19 (stable) | Fold coupling active |

Key finding: `b` stayed positive—the network is no longer trying to subtract the spiral-only output. The reinjection is being used as intended. And `c` values spread across heads (std 0.24 vs 0.07 in V1), confirming bifurcation-based head specialization.

However, overall accuracy was still identical to standard attention (21.9% vs 21.9%). The fold sparsity was 64.2%—about 36% of attention score positions had the fold active, which is reasonable selectivity.

### Decision Log

**Decision:** Design a harder task specifically intended to stress the capabilities Rössler attention claims to provide: long-range attention across noise disruptions.

**Rationale:** The periodic sequence task may be too easy for standard attention. Both models solved it equally because the task did not require any capability that the Rössler mechanism uniquely provides. A harder task with adversarial noise and mode transitions might expose an advantage.

---

## Experiment 3: Harder Task — Multi-Scale Periodic with Mode Transitions

**Goal:** Test whether Rössler attention provides an advantage when the task specifically requires attending across noise disruptions to recover context.

**Setup:** Four modes with non-overlapping token sets, each with a unique repeating pattern of period 3–5. Sequences alternate between modes via transition signals (token 15) followed by 3–6 noise tokens. To predict the first tokens of a new mode after noise, the model must attend past the noise to identify the pre-transition mode and infer which mode comes next.

This task introduces adversarial noise (drawn from all modes), requires attending across 4–7 positions of noise, and allows within-mode vs. post-transition accuracy to be measured separately.

**Measurement battery:** Overall accuracy, within-mode accuracy, post-transition accuracy (critical metric), transition detection, attention entropy per head, fold activation sparsity, head specialization divergence, reinjection utilization, loss curve smoothness, and parameter trajectory.

**Results at 200 epochs:**

Within-mode accuracy reached 98.5% (standard) and 98.8% (Rössler)—both models nearly solved the within-mode pattern prediction. Post-transition accuracy: standard 91.2%, Rössler 90.2%. Standard attention slightly outperformed on the critical metric.

The parameter trajectory revealed systematic dismantling again:

```
c:      0.07 -> 0.43   (fold getting more selective, threshold climbing)
lambda: 0.14 -> 0.08   (fold coupling weakening)
b:     -0.02 -> -0.24  (reinjection inverting again, strongly negative)
a:      0.13 -> 0.03   (spiral dying)
```

The network was again trying to eliminate the Rössler components. The negative `b` was particularly telling: the network was computing a contrast between fold-modified and unmodified attention—a form of differential attention that emerged spontaneously.

Head divergence was also revealing: standard attention had more head specialization (sym-KL 3.66 vs 3.08) than Rössler. The extra Rössler parameters may have added noise to the specialization process rather than enhancing it.

**Key observation:** The fold suppresses high attention scores, but in this task, the high scores were correct—they pointed to the right mode tokens. The fold was fighting the attention rather than helping it. Standard attention naturally ignored noise tokens by learning discriminative QK embeddings. The fold mechanism may not be solving a problem this task actually has.

### Decision Log

**Decision:** Pivot from "prescribe a mechanism and test it" to "discover where standard attention breaks, then test individual components on those failure modes." Design a systematic four-phase pipeline: failure mode discovery, component ablation, mechanism refinement, and scale testing.

**Rationale:** Three experiments show a consistent pattern: the network dismantles Rössler components when they are not useful, and the negative `b` (differential attention) emerges spontaneously every time. Rather than continue iterating the full mechanism, the approach should isolate which components help on which tasks—starting by finding where standard attention actually breaks.

**The 8 synthetic tasks:**

- **Associative recall:** Key-value pairs followed by a query key; predict the associated value.
- **Selective copy:** Copy specific tokens while ignoring interleaved distractors.
- **Dual-stream tracking:** Two interleaved periodic streams with different periods.
- **Nested periodicity:** Three nested periodic scales (period 3 inside period 12 inside period 36).
- **Sparse needle retrieval:** One critical token buried in 200+ tokens of noise.
- **Pattern with confounders:** Establish a pattern, show a confounding pattern, resume the original.
- **Mode interference:** Two modes share 50% of tokens but have different orderings.
- **Compositional lookup:** Two-hop pointer indirection.

---

## Experiment 4: Systematic Failure Mode Discovery

**Goal:** Find where standard attention actually breaks, rather than guessing what tasks should favor Rössler attention.

**Setup:** All tasks used vocab 14–18, dim 64, 4 heads, 1 layer, seq len 256, 200 epochs, Adam 1e-3. Training set: 2048 sequences. Test set: 512 sequences. Each task has a "critical mask"—the specific token positions that test the task's core capability.

**Results—ranked by critical accuracy (hardest first):**

| Task | Overall Acc | Critical Acc | What It Tests |
|------|------------|-------------|---------------|
| Associative recall | 95.0% | **23.4%** | Long-range key-value retrieval |
| Selective copy | 82.7% | **62.4%** | Copy past distractors |
| Nested periodicity | 96.9% | 88.5% | Multi-scale pattern detection |
| Mode interference | 92.6% | 97.1% | Contextual disambiguation |
| Pattern confounders | 98.2% | 99.9% | Resistance to recency bias |
| Dual-stream | 99.7% | 100.0% | Head specialization |
| Sparse needle | 99.6% | 100.0% | Extreme selectivity |
| Compositional lookup | 99.7% | 100.0% | Two-hop composition |

Standard attention completely solved 5 of 8 tasks. It partially solved nested periodicity (88.5%) and selective copy (62.4%). It failed hard on associative recall (23.4%)—barely above random for a multi-class prediction. The two hard tasks (< 85% critical accuracy) were carried forward for ablation.

Notably, the compositional lookup task—designed as a ceiling test that 1-layer models structurally cannot solve—was solved perfectly at 100%. This may mean the task design was not actually requiring two-hop composition, or the model found a shortcut. Either way, the task was not useful as a ceiling test.

The hard tasks share a common characteristic: they require precise retrieval of specific token content from earlier in the sequence, not just pattern matching.

### Decision Log

**Decision:** Run all 6 attention variants on the 2 hard tasks (associative recall and selective copy) to determine which components help.

**Rationale:** Standard attention breaks on associative recall and selective copy. If any Rössler-derived component provides an advantage, it should appear on these tasks. Testing 6 variants (standard, fold-only, spiral-only, differential, fold+differential, full Rossler) isolates which specific mechanisms matter.

---

## Experiment 5: Component Ablation on Hard Tasks

**Goal:** For each hard task, test all 6 attention variants to determine which Rössler-derived components (if any) help.

**The 6 variants:**
1. **Standard** — vanilla dot-product attention (baseline)
2. **Fold only** — standard + score-based fold (`F = v_energy * relu(S - c)`)
3. **Spiral only** — standard + key self-feedback (`K_spiral = K + a * (K @ W_self)`)
4. **Differential** — two independent attention computations, output is their difference
5. **Fold + differential** — fold suppression combined with differential contrast
6. **Full Rossler** — all components enabled (fold + spiral + reinjection)

**Results—critical mask accuracy on each hard task:**

| Variant | Associative Recall | Selective Copy |
|---------|-------------------|----------------|
| **Standard** | **24.4%** | **63.2%** |
| Fold only | 22.9% | 57.2% |
| Spiral only | 19.5% | 44.6% |
| Differential | 24.2% | 40.6% |
| Fold + differential | 20.7% | 52.0% |
| Full Rossler | 23.1% | 54.6% |

**Delta vs standard:**

| Variant | Associative Recall | Selective Copy |
|---------|-------------------|----------------|
| Fold only | -1.6% | -6.0% |
| Spiral only | -4.9% | -18.7% |
| Differential | -0.2% | -22.7% |
| Fold + differential | -3.7% | -11.3% |
| Full Rossler | -1.4% | -8.7% |

Every variant tested on these two hard tasks performed worse than standard attention. The spiral was the most destructive (-4.9% and -18.7%). Differential attention—the pattern the network kept discovering spontaneously—performed the worst on selective copy (-22.7%).

The fold was the least harmful variant, losing by only 1.6% on associative recall and 6.0% on selective copy.

### Decision Log

**Decision:** Test the most promising candidates (differential and fold+differential) across all 8 tasks, even though they did not win on hard tasks. Also include full Rossler for completeness.

**Rationale:** A component that loses on hard tasks might still help on easy tasks, or might not hurt overall accuracy. Testing across all 8 tasks checks for any wins and ensures the refined mechanism does not catastrophically fail on already-solved tasks.

---

## Experiment 6: Full Task Battery Refinement

**Goal:** Check for wins on any task, including easy ones, and verify that promising candidates do not hurt already-solved tasks.

**Results—critical accuracy across all 8 tasks:**

| Task | Standard | Differential | Fold+Diff | Rossler | Winner |
|------|----------|-------------|-----------|---------|--------|
| Associative recall | **25.8%** | 24.0% | 20.7% | 22.3% | Standard |
| Selective copy | **63.1%** | 40.3% | 51.2% | 57.5% | Standard |
| Dual-stream | 100% | 100% | 100% | 100% | Tie |
| Nested periodicity | 88.4% | 88.4% | 88.5% | **88.5%** | Rossler (marginal) |
| Sparse needle | 100% | 100% | 100% | 100% | Tie |
| Pattern confounders | 99.5% | 99.4% | **100%** | 99.5% | Fold+Diff |
| Mode interference | 97.1% | 96.4% | **98.4%** | 97.6% | Fold+Diff |
| Compositional lookup | 100% | **10.4%** | 100% | 100% | Standard (Diff catastrophic) |

**Aggregate:**

| Variant | Mean Accuracy | Tasks Won |
|---------|--------------|-----------|
| **Standard** | **84.2%** | **5** |
| Rossler | 83.2% | 1 |
| Fold+Differential | 82.3% | 2 |
| Differential | 69.9% | 0 |

Standard attention won overall with the highest mean accuracy and the most task wins in this battery.

**Notable observations:**

- Differential attention catastrophically failed on compositional lookup (10.4% vs 100%). This may be because splitting the computation into two separate QKV projections halved the effective capacity for tasks requiring all heads to work together.
- Fold+differential showed marginal wins on pattern confounders (+0.5%) and mode interference (+1.3%). These are tasks where score suppression could theoretically help, but the margins are small.

### Decision Log

**Decision:** Run a scale test on real data (character-level Shakespeare language modeling) to see if findings transfer.

**Rationale:** No variant proved consistently useful in the synthetic battery. However, synthetic tasks may not capture all the dynamics of real language. A scale test on real data at larger capacity (dim 128, 8 heads, 2 layers) provides a different test surface.

---

## Experiment 7: Scale Test on Real Data

**Goal:** Test whether findings transfer to a real language modeling task at larger scale.

**Setup:** Character-level language modeling on Tiny Shakespeare (~1MB text, 65-character vocab). Model: dim 128, 8 heads, 2 layers, 446K–450K parameters. Trained for 50 epochs with Adam 3e-4. Only standard and full Rossler tested.

**Results:**

| Variant | BPC (bits-per-char) | Parameters | Training Time |
|---------|--------------------:|------------|---------------|
| Standard | 2.482 | 446,145 | 267s |
| Rossler | 2.464 | 450,305 | 585s |

Delta: Rossler -0.018 BPC. Rossler has 4,160 extra parameters (+0.9%) and takes 2.2x longer to train. The BPC improvement is within noise for this small model and dataset, and the extra parameters could account for the difference.

---

## Observations Across Experiments

### Confirmed

- The score-based fold formulation (`F = v_energy * relu(S - c)`) creates genuine sparsity (approximately 64% of positions dormant) and allows heads to learn different thresholds.
- The network consistently modifies or dismantles Rössler components that do not provide useful signal: the spiral (a) is driven toward ~0, the reinjection (b) inverts to negative values, and the fold coupling (lambda) weakens.
- Across this 8-task synthetic battery and one real-data scale test, standard attention matched or outperformed all Rössler-derived variants.
- The measurement battery (10 metrics across accuracy, attention patterns, fold dynamics, and parameter trajectories) provides clear signal about what the mechanism is doing internally.

### Observed

- The spiral component (key self-feedback) appears consistently harmful across experiments. In every test, `a` was driven toward 0 or negative. This may indicate that adding a learnable projection to keys introduces noise without useful signal in this setup.
- The reinjection parameter `b` inverted in every long-run experiment. The network spontaneously discovered a contrast between fold-modified and unmodified attention. This emergent pattern may be a training artifact rather than a generalizable mechanism.
- The differential attention variant (two independent QKV computations, output is their difference) catastrophically failed on compositional lookup. This may suggest that splitting capacity across two attention computations reduces effective model size, though this was not isolated.
- Fold+differential showed marginal gains on pattern confounders and mode interference—tasks where suppressing misleading high-attention tokens could theoretically help. The gains were small (0.5% and 1.3%) and were not replicated at scale.

### Unexplained

- **Does the fold solve a problem standard attention actually has?** On the multi-scale periodic task, the fold suppressed high attention scores that were actually correct. On confounders and mode interference, the fold showed marginal benefit. Whether there exists a task where score suppression genuinely helps remains open.
- **Would Rössler dynamics work across layers or timesteps?** All experiments applied the full mechanism within a single attention computation. The temporal iteration that makes the Rössler attractor work—spiral, eject, reinject—does not exist in a single forward pass. Whether it would help if applied across layers or in a recurrent architecture was not tested.
- **Why does the network always invert `b`?** The spontaneous emergence of negative `b` (differential attention) in every long experiment may indicate that the contrast structure is useful, or it may be an artifact of how the reinjection interacts with the fold during training. This was not isolated.

---

## Where This Stands

**What has been tested:**

- The Rössler mapping applied within a single attention computation, tested on 8 synthetic tasks and one real-data scale test.
- Individual components (fold, spiral, reinjection, differential) isolated and ablated.
- The score-based fold formulation trains and produces genuine sparsity.

**What has not been tested:**

- Rössler dynamics applied across transformer layers or timesteps.
- The fold as a standalone technique without the spiral or reinjection framing.
- Tasks specifically designed to require score suppression (stronger adversarial interference).
- Recurrent or state-space models with Rössler-structured transitions.

**What the next experiments should address:**

1. **Test the fold as a standalone technique.** Remove the Rössler framing and test `v_energy * relu(S - c)` as a lightweight attention modification on tasks with misleading high-attention tokens.
2. **Apply dynamics across layers or timesteps.** The temporal iteration that makes the attractor work could manifest if the state evolves over multiple steps.
3. **Design tasks with stronger adversarial interference.** The fold showed marginal benefit on tasks with misleading tokens. A task specifically engineered to require score suppression might expose a real advantage.

---

## Open Questions

1. **Would Rössler dynamics work across layers or timesteps?** The temporal iteration that makes the attractor work could manifest in a recurrent architecture or across transformer layers. This was not tested.

2. **Are there tasks where attention score suppression genuinely helps?** The fold showed marginal wins on pattern confounders and mode interference. A task with stronger adversarial interference might expose a real advantage.

3. **Is the spontaneous emergence of differential attention meaningful?** The network inverted `b` in every long experiment, computing a contrast between fold-modified and unmodified attention. Whether this is a useful pattern or a training artifact remains unclear.

4. **Does the spiral component have any valid use case?** Key self-feedback was consistently harmful in these tests. Whether it helps in a different setting (e.g., very long sequences, many heads, recurrent application) is unknown.

---

## Experimental Timeline

| Experiment | Duration | Key Finding |
|------------|----------|-------------|
| Initial mechanism | ~10 min | Fold gate disconnected from scores; network dismantled all components |
| Score-based fold | ~10 min | Fold alive, heads specialize, but no accuracy gain |
| Harder task | ~6 min | Standard wins; network inverts reinjection |
| Failure mode discovery | 37 min | 2 of 8 tasks hard for standard attention |
| Component ablation | 71 min | Zero wins for any variant on hard tasks |
| Full task battery | 179 min | Standard wins 5/8 tasks, highest mean accuracy |
| Scale test (Shakespeare) | 14 min | -0.018 BPC (marginal, within noise) |
| **Total** | **~327 min** | |

All experiments run on Apple M1, MLX 0.31.1. Rössler experiments: April 12, 2026.
