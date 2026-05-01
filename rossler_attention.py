"""
Rossler Attention — Validation Test v2

Multi-scale periodic sequences with state-dependent mode transitions.
The task requires attending across noise disruptions to recover mode
context, which should expose the fold mechanism's advantages.

Measurement battery:
  1.  Overall accuracy
  2.  Within-mode accuracy (stable periodic tokens)
  3.  Post-transition accuracy (first 3 tokens after mode change — THE critical metric)
  4.  Transition detection (can the model predict transition signals?)
  5.  Attention entropy per head
  6.  Fold activation sparsity (Rossler only)
  7.  Head specialization divergence
  8.  Reinjection utilization (Rossler only)
  9.  Loss curve smoothness
  10. Parameter trajectory (Rossler only)

Usage:
  python rossler_attention.py
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time


# ============================================================
# Configuration
# ============================================================

VOCAB_SIZE = 16
EMBED_DIM = 64
NUM_HEADS = 4
SEQ_LEN = 256
BATCH_SIZE = 32
NUM_TRAIN = 2048
NUM_TEST = 512
EPOCHS = 200
LR = 1e-3
SEED = 42

NUM_MODES = 4
TRANSITION_TOKEN = 15

# Fixed mode patterns (tokens 0-14, deliberately overlapping across modes)
MODE_PATTERNS = [
    [0, 3, 7],              # Mode 0, period 3
    [5, 1, 9, 6],           # Mode 1, period 4
    [8, 4, 11],             # Mode 2, period 3
    [2, 10, 13, 14, 12],    # Mode 3, period 5
]

# Transition order: mode -> next mode
NEXT_MODE = [1, 2, 3, 0]  # 0->1->2->3->0


# ============================================================
# Data Generation
# ============================================================

def generate_sequences(num_seqs, seq_len, seed=0):
    """Generate multi-scale periodic sequences with mode transitions.

    Each sequence starts in a random mode, follows its repeating pattern,
    then transitions: transition_token (15) -> 3-6 noise tokens -> new mode.

    The task: predict the next token. Post-transition tokens require attending
    across noise to identify the pre-transition mode and infer the next one.

    Returns:
        sequences: (num_seqs, seq_len) int32
        masks: dict of (num_seqs, seq_len) bool arrays:
            'within_mode':     stable periodic tokens (not near transitions)
            'post_transition': first 3 tokens of new mode after noise
            'noise':           transition signal + noise tokens
    """
    rng = np.random.RandomState(seed)
    sequences = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {
        "within_mode": np.zeros((num_seqs, seq_len), dtype=bool),
        "post_transition": np.zeros((num_seqs, seq_len), dtype=bool),
        "noise": np.zeros((num_seqs, seq_len), dtype=bool),
    }

    for i in range(num_seqs):
        mode = rng.randint(0, NUM_MODES)
        pattern = MODE_PATTERNS[mode]
        period = len(pattern)
        phase = 0
        pos = 0
        tokens_in_mode = 0
        tokens_since_mode_start = 999  # initial tokens are within_mode
        next_transition_at = rng.randint(15, 26)

        while pos < seq_len:
            if tokens_in_mode >= next_transition_at:
                # === TRANSITION SIGNAL ===
                if pos >= seq_len:
                    break
                sequences[i, pos] = TRANSITION_TOKEN
                masks["noise"][i, pos] = True
                pos += 1

                # === NOISE TOKENS ===
                noise_len = rng.randint(3, 7)
                noise_end = min(pos + noise_len, seq_len)
                noise_tokens = rng.randint(0, VOCAB_SIZE - 1, size=noise_end - pos)
                sequences[i, pos:noise_end] = noise_tokens
                for j in range(pos, noise_end):
                    masks["noise"][i, j] = True
                pos = noise_end

                # === NEW MODE ===
                mode = NEXT_MODE[mode]
                pattern = MODE_PATTERNS[mode]
                period = len(pattern)
                phase = 0
                tokens_in_mode = 0
                tokens_since_mode_start = 0
                next_transition_at = rng.randint(15, 26)
            else:
                # === PERIODIC TOKEN ===
                sequences[i, pos] = pattern[phase % period]

                if tokens_since_mode_start < 3:
                    masks["post_transition"][i, pos] = True
                else:
                    masks["within_mode"][i, pos] = True

                phase += 1
                tokens_in_mode += 1
                tokens_since_mode_start += 1
                pos += 1

    return sequences, masks


# ============================================================
# Causal Mask
# ============================================================

def make_causal_mask(T):
    """Lower-triangular causal mask: 0 where attending is allowed, -1e9 where blocked."""
    rows = mx.arange(T).reshape(T, 1)
    cols = mx.arange(T).reshape(1, T)
    return mx.where(rows >= cols, mx.array(0.0), mx.array(-1e9))


# ============================================================
# Standard Attention
# ============================================================

class StandardAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def _qkv(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        return Q, K, V

    def __call__(self, x):
        B, T, D = x.shape
        Q, K, V = self._qkv(x)
        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale
        S = S + make_causal_mask(T)
        A = mx.softmax(S, axis=-1)
        O = (A @ V).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.o_proj(O)

    def get_diagnostics(self, x):
        B, T, D = x.shape
        Q, K, V = self._qkv(x)
        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale
        S = S + make_causal_mask(T)
        A = mx.softmax(S, axis=-1)
        return {"A": A}


# ============================================================
# Rossler Attention
# ============================================================

class RosslerAttention(nn.Module):
    """Attention mechanism derived from Rossler attractor dynamics.

    The fold gate is F = v_energy * relu(S - c), directly mirroring z(x-c)
    from the ODE: value magnitude (z) multiplies score exceedance (x-c).
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # Rossler dynamics parameters (per-head)
        self.rossler_a = mx.array([0.1] * num_heads)
        self.rossler_b = mx.array([0.01] * num_heads)
        self.rossler_c = mx.array([0.0] * num_heads)
        self.rossler_lam = mx.array([0.1] * num_heads)

        # Key self-feedback projection (H, d, d)
        self.w_self = mx.random.normal(
            shape=(num_heads, self.head_dim, self.head_dim)
        ) * (2.0 / (self.head_dim + self.head_dim)) ** 0.5

    def _forward_internals(self, x):
        """Shared forward logic returning all intermediates."""
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)

        a = self.rossler_a.reshape(1, H, 1, 1)
        b = self.rossler_b.reshape(1, H, 1, 1)
        c = self.rossler_c.reshape(1, H, 1, 1)
        lam = self.rossler_lam.reshape(1, H, 1, 1)

        # Spiral
        w = self.w_self.reshape(1, H, d, d)
        K_spiral = K + a * (K @ w)
        S = (Q @ K_spiral.transpose(0, 1, 3, 2)) * self.scale
        S = S + make_causal_mask(T)

        # Fold: F = v_energy * relu(S - c)
        v_energy = mx.sqrt(mx.sum(V * V, axis=-1, keepdims=True))  # (B,H,T,1)
        v_energy_t = v_energy.transpose(0, 1, 3, 2)                # (B,H,1,T)
        F = v_energy_t * nn.relu(S - c)                             # (B,H,T,T)

        # Coupling
        A = mx.softmax(S - lam * F, axis=-1)

        # Reinjection
        spiral_only = mx.softmax(S, axis=-1) @ V
        fold_output = A @ V
        O = fold_output + b * spiral_only

        return O, {"A": A, "F": F, "S": S, "V": V, "v_energy": v_energy_t,
                    "spiral_only": spiral_only, "fold_output": fold_output,
                    "b": b}

    def __call__(self, x):
        B, T, D = x.shape
        O, _ = self._forward_internals(x)
        O = O.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.o_proj(O)

    def get_diagnostics(self, x):
        _, diag = self._forward_internals(x)
        return diag


# ============================================================
# Feed-Forward, Block, Model
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4)
        self.w2 = nn.Linear(dim * 4, dim)

    def __call__(self, x):
        return self.w2(nn.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, attention_cls):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_cls(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, max_seq_len, attention_cls):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.block = TransformerBlock(dim, num_heads, attention_cls)
        self.norm = nn.LayerNorm(dim)
        self.out_head = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        B, T = x.shape
        positions = mx.arange(T)
        x = self.token_embed(x) + self.pos_embed(positions)
        x = self.block(x)
        x = self.norm(x)
        return self.out_head(x)

    def get_attn_diagnostics(self, x):
        B, T = x.shape
        positions = mx.arange(T)
        h = self.token_embed(x) + self.pos_embed(positions)
        h_normed = self.block.norm1(h)
        return self.block.attn.get_diagnostics(h_normed)


# ============================================================
# Training
# ============================================================

def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def train_epoch(model, optimizer, loss_grad_fn, train_x, train_y, batch_size):
    num_batches = train_x.shape[0] // batch_size
    total_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        x = train_x[start:start + batch_size]
        y = train_y[start:start + batch_size]
        loss, grads = loss_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
    return total_loss / num_batches


def record_rossler_params(model):
    """Snapshot current Rossler parameters."""
    attn = model.block.attn
    return {
        "a": [attn.rossler_a[h].item() for h in range(attn.num_heads)],
        "b": [attn.rossler_b[h].item() for h in range(attn.num_heads)],
        "c": [attn.rossler_c[h].item() for h in range(attn.num_heads)],
        "lam": [attn.rossler_lam[h].item() for h in range(attn.num_heads)],
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, test_x, test_y, target_masks, batch_size=BATCH_SIZE):
    """Compute accuracy overall and per mask category."""
    total_correct = 0
    total_tokens = 0
    cat_correct = {k: 0.0 for k in target_masks}
    cat_tokens = {k: 0 for k in target_masks}

    # Transition detection: accuracy where target == TRANSITION_TOKEN
    trans_correct = 0
    trans_tokens = 0

    num_batches = (test_x.shape[0] + batch_size - 1) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        x = test_x[start:start + batch_size]
        y = test_y[start:start + batch_size]

        logits = model(x)
        preds = mx.argmax(logits, axis=-1)
        correct = (preds == y).astype(mx.float32)
        mx.eval(correct)

        total_correct += mx.sum(correct).item()
        total_tokens += correct.size

        for k in target_masks:
            m = target_masks[k][start:start + batch_size]
            m_mx = mx.array(m.astype(np.float32))
            cat_correct[k] += mx.sum(correct * m_mx).item()
            cat_tokens[k] += int(m.sum())

        # Transition detection
        trans_mask = (y == TRANSITION_TOKEN).astype(mx.float32)
        trans_correct += mx.sum(correct * trans_mask).item()
        trans_tokens += int(mx.sum(trans_mask).item())

    results = {
        "overall": total_correct / max(total_tokens, 1),
    }
    for k in target_masks:
        results[k] = cat_correct[k] / max(cat_tokens[k], 1)
    results["transition_detect"] = trans_correct / max(trans_tokens, 1)
    results["_counts"] = {k: cat_tokens[k] for k in target_masks}
    results["_counts"]["total"] = total_tokens
    results["_counts"]["transition"] = trans_tokens

    return results


# ============================================================
# Diagnostics
# ============================================================

def compute_diagnostics(model, test_x, batch_size=BATCH_SIZE, is_rossler=False):
    """Compute attention diagnostics on test data."""
    all_entropy = []
    all_divergence = []
    all_fold_sparsity = []
    all_reinj_ratio = []

    num_batches = min(4, (test_x.shape[0] + batch_size - 1) // batch_size)

    for i in range(num_batches):
        start = i * batch_size
        x = test_x[start:start + batch_size]
        diag = model.get_attn_diagnostics(x)
        A = diag["A"]  # (B, H, T, T)
        mx.eval(A)

        B, H, T, _ = A.shape

        # --- Metric 5: Attention entropy per head ---
        eps = 1e-8
        entropy = -mx.sum(A * mx.log(A + eps), axis=-1)  # (B, H, T)
        per_head_entropy = mx.mean(entropy, axis=(0, 2))  # (H,)
        mx.eval(per_head_entropy)
        all_entropy.append(np.array([per_head_entropy[h].item() for h in range(H)]))

        # --- Metric 7: Head specialization divergence ---
        # Symmetric KL between all head pairs, averaged
        divs = []
        for h1 in range(H):
            for h2 in range(h1 + 1, H):
                p = A[:, h1] + eps
                q = A[:, h2] + eps
                kl_pq = mx.sum(p * mx.log(p / q), axis=-1)  # (B, T)
                kl_qp = mx.sum(q * mx.log(q / p), axis=-1)
                sym_kl = (mx.mean(kl_pq) + mx.mean(kl_qp)) / 2.0
                mx.eval(sym_kl)
                divs.append(sym_kl.item())
        all_divergence.append(np.mean(divs))

        if is_rossler:
            F = diag["F"]
            fold_output = diag["fold_output"]
            spiral_only = diag["spiral_only"]
            b = diag["b"]
            mx.eval(F, fold_output, spiral_only)

            # --- Metric 6: Fold activation sparsity ---
            # Exclude causally masked positions (they're -1e9 * relu -> 0 trivially)
            # Only count unmasked positions
            causal = mx.arange(T).reshape(T, 1) >= mx.arange(T).reshape(1, T)  # (T,T)
            causal_f = causal.astype(mx.float32)
            unmasked_count = mx.sum(causal_f).item() * B * H
            # F is zero where relu(S-c) <= 0 (sparse by design)
            zeros_in_unmasked = mx.sum((F < 1e-6).astype(mx.float32) * causal_f).item() * 1
            # Need to broadcast causal over B, H
            causal_bh = causal_f.reshape(1, 1, T, T)
            zeros_in_unmasked = mx.sum(
                (F < 1e-6).astype(mx.float32) * causal_bh
            ).item()
            unmasked_count = mx.sum(causal_bh).item() * B * H
            sparsity = zeros_in_unmasked / max(unmasked_count, 1)
            all_fold_sparsity.append(sparsity)

            # --- Metric 8: Reinjection utilization ---
            reinj = b * spiral_only  # (B, H, T, d)
            reinj_norm = mx.sqrt(mx.sum(reinj * reinj, axis=-1) + eps)  # (B, H, T)
            fold_norm = mx.sqrt(mx.sum(fold_output * fold_output, axis=-1) + eps)
            ratio = mx.mean(reinj_norm / fold_norm)
            mx.eval(ratio)
            all_reinj_ratio.append(ratio.item())

    results = {
        "entropy_per_head": np.mean(all_entropy, axis=0),  # (H,)
        "head_divergence": float(np.mean(all_divergence)),
    }
    if is_rossler:
        results["fold_sparsity"] = float(np.mean(all_fold_sparsity))
        results["reinjection_ratio"] = float(np.mean(all_reinj_ratio))

    return results


# ============================================================
# Utilities
# ============================================================

def count_params(tree):
    total = 0
    if isinstance(tree, mx.array):
        return tree.size
    elif isinstance(tree, dict):
        for v in tree.values():
            total += count_params(v)
    elif isinstance(tree, list):
        for v in tree:
            total += count_params(v)
    return total


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(name, attention_cls, train_x, train_y, test_x, test_y,
                   target_masks, is_rossler=False):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    mx.random.seed(SEED)
    model = Model(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, SEQ_LEN - 1, attention_cls)
    mx.eval(model.parameters())

    num_params = count_params(model.parameters())
    print(f"  Parameters: {num_params:,}")

    optimizer = optim.Adam(learning_rate=LR)
    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    losses = []
    param_history = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, optimizer, loss_grad_fn, train_x, train_y, BATCH_SIZE)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch + 1:>3}/{EPOCHS}  loss: {avg_loss:.4f}  ({elapsed:.1f}s)")

            if is_rossler:
                snapshot = record_rossler_params(model)
                snapshot["epoch"] = epoch + 1
                param_history.append(snapshot)

    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s")

    # Evaluation
    acc = evaluate(model, test_x, test_y, target_masks)
    print(f"  Overall accuracy:        {acc['overall']:.4f}")
    print(f"  Within-mode accuracy:    {acc['within_mode']:.4f}")
    print(f"  Post-transition accuracy:{acc['post_transition']:.4f}")
    print(f"  Transition detection:    {acc['transition_detect']:.4f}")

    # Diagnostics
    print(f"\n  Computing diagnostics...")
    diag = compute_diagnostics(model, test_x, is_rossler=is_rossler)

    ent = diag["entropy_per_head"]
    print(f"  Attention entropy per head: [{', '.join(f'{e:.3f}' for e in ent)}]")
    print(f"  Head specialization (sym-KL): {diag['head_divergence']:.4f}")

    if is_rossler:
        print(f"  Fold activation sparsity: {diag['fold_sparsity']:.4f}")
        print(f"  Reinjection utilization:  {diag['reinjection_ratio']:.4f}")

        # Print final Rossler params
        attn = model.block.attn
        print(f"\n  Rossler parameters (final):")
        print(f"  {'Head':<6} {'a':>8} {'b':>8} {'c':>8} {'lambda':>8}")
        print(f"  {'-' * 38}")
        c_vals = []
        for h in range(NUM_HEADS):
            a_v = attn.rossler_a[h].item()
            b_v = attn.rossler_b[h].item()
            c_v = attn.rossler_c[h].item()
            l_v = attn.rossler_lam[h].item()
            c_vals.append(c_v)
            print(f"  {h:<6} {a_v:>8.4f} {b_v:>8.4f} {c_v:>8.4f} {l_v:>8.4f}")

        c_std = float(np.std(c_vals))
        print(f"\n  c std: {c_std:.4f} ({'differentiated' if c_std > 0.1 else 'not differentiated'})")

        # Parameter trajectory (mean across heads)
        if param_history:
            print(f"\n  Parameter trajectory (mean across heads):")
            print(f"  {'Epoch':<8} {'a':>8} {'b':>8} {'c':>8} {'lam':>8}")
            print(f"  {'-' * 40}")
            for snap in param_history:
                print(f"  {snap['epoch']:<8} "
                      f"{np.mean(snap['a']):>8.4f} "
                      f"{np.mean(snap['b']):>8.4f} "
                      f"{np.mean(snap['c']):>8.4f} "
                      f"{np.mean(snap['lam']):>8.4f}")

    return {
        "losses": losses,
        "acc": acc,
        "diag": diag,
        "model": model,
        "train_time": train_time,
        "param_history": param_history if is_rossler else None,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("Rossler Attention — Validation Test v2")
    print(f"Config: vocab={VOCAB_SIZE}, dim={EMBED_DIM}, heads={NUM_HEADS}, "
          f"seq_len={SEQ_LEN}, epochs={EPOCHS}, lr={LR}")
    print(f"Task: {NUM_MODES} modes, transition signal=token {TRANSITION_TOKEN}")
    print(f"Framework: MLX on Apple Silicon\n")

    # Generate data
    print("Generating data...")
    train_seqs, train_masks = generate_sequences(NUM_TRAIN, SEQ_LEN, seed=42)
    test_seqs, test_masks = generate_sequences(NUM_TEST, SEQ_LEN, seed=123)

    train_x = mx.array(train_seqs[:, :-1])
    train_y = mx.array(train_seqs[:, 1:])
    test_x = mx.array(test_seqs[:, :-1])
    test_y = mx.array(test_seqs[:, 1:])

    # Align masks with target positions (shift by 1)
    target_masks = {k: v[:, 1:] for k, v in test_masks.items()}

    # Data stats
    total = target_masks["within_mode"].size
    for k in target_masks:
        n = int(target_masks[k].sum())
        print(f"  {k}: {n} tokens ({100 * n / total:.1f}%)")
    trans_count = int((test_seqs[:, 1:] == TRANSITION_TOKEN).sum())
    print(f"  transition signals: {trans_count} tokens ({100 * trans_count / total:.1f}%)")
    print(f"  total: {total} tokens")

    # Run experiments
    standard = run_experiment(
        "Standard Attention", StandardAttention,
        train_x, train_y, test_x, test_y, target_masks,
        is_rossler=False,
    )
    rossler = run_experiment(
        "Rossler Attention", RosslerAttention,
        train_x, train_y, test_x, test_y, target_masks,
        is_rossler=True,
    )

    # ── Full Comparison ─────────────────────────────────────
    sa = standard["acc"]
    ra = rossler["acc"]
    sd = standard["diag"]
    rd = rossler["diag"]

    print(f"\n{'=' * 70}")
    print(f"  FULL COMPARISON")
    print(f"{'=' * 70}")

    def row(label, s_val, r_val, fmt=".4f"):
        delta = r_val - s_val
        print(f"  {label:<30} {s_val:>10{fmt}} {r_val:>10{fmt}} {delta:>+10{fmt}}")

    # Accuracy metrics
    print(f"\n  {'ACCURACY':<30} {'Standard':>10} {'Rossler':>10} {'Delta':>10}")
    print(f"  {'-' * 60}")
    row("Overall", sa["overall"], ra["overall"])
    row("Within-mode", sa["within_mode"], ra["within_mode"])
    row("Post-transition", sa["post_transition"], ra["post_transition"])
    row("Transition detection", sa["transition_detect"], ra["transition_detect"])

    # Attention diagnostics
    print(f"\n  {'ATTENTION DIAGNOSTICS':<30} {'Standard':>10} {'Rossler':>10} {'Delta':>10}")
    print(f"  {'-' * 60}")
    for h in range(NUM_HEADS):
        row(f"Entropy head {h}", sd["entropy_per_head"][h], rd["entropy_per_head"][h])
    row("Mean entropy", np.mean(sd["entropy_per_head"]), np.mean(rd["entropy_per_head"]))
    row("Head divergence (sym-KL)", sd["head_divergence"], rd["head_divergence"])

    # Rossler-only metrics
    print(f"\n  {'ROSSLER-SPECIFIC':<30} {'Value':>10}")
    print(f"  {'-' * 40}")
    print(f"  {'Fold sparsity':<30} {rd['fold_sparsity']:>10.4f}")
    print(f"  {'Reinjection ratio':<30} {rd['reinjection_ratio']:>10.4f}")

    # Loss curve analysis
    s_losses = np.array(standard["losses"])
    r_losses = np.array(rossler["losses"])

    # Smoothness: std of epoch-to-epoch changes
    s_diffs = np.diff(s_losses)
    r_diffs = np.diff(r_losses)
    s_smooth = float(np.std(s_diffs))
    r_smooth = float(np.std(r_diffs))

    print(f"\n  {'LOSS CURVE':<30} {'Standard':>10} {'Rossler':>10} {'Delta':>10}")
    print(f"  {'-' * 60}")
    row("Final loss", s_losses[-1], r_losses[-1])
    row("Loss smoothness (std diffs)", s_smooth, r_smooth)
    row("Min loss achieved", s_losses.min(), r_losses.min())

    # Verdict
    print(f"\n  {'─' * 60}")
    delta_post = ra["post_transition"] - sa["post_transition"]
    if delta_post > 0.03:
        verdict = f"PASS — Rossler advantage on post-transition: +{delta_post:.4f}"
    elif delta_post > 0.01:
        verdict = f"PROMISING — Rossler marginal advantage: +{delta_post:.4f}"
    elif delta_post > -0.01:
        verdict = f"NEUTRAL — no significant difference: {delta_post:+.4f}"
    else:
        verdict = f"FAIL — Standard outperforms: {delta_post:+.4f}"
    print(f"  {verdict}")
    print()


if __name__ == "__main__":
    main()
