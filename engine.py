"""Shared infrastructure: model, training, evaluation, diagnostics."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

from attention import ROSSLER_VARIANTS


# ── Default config ──────────────────────────────────────────

DEFAULT_CONFIG = {
    "embed_dim": 64,
    "num_heads": 4,
    "seq_len": 256,
    "batch_size": 32,
    "epochs": 200,
    "lr": 1e-3,
    "seed": 42,
}


# ── Model components ───────────────────────────────────────

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
        x = self.token_embed(x) + self.pos_embed(mx.arange(T))
        x = self.block(x)
        x = self.norm(x)
        return self.out_head(x)

    def get_attn_diagnostics(self, x):
        B, T = x.shape
        h = self.token_embed(x) + self.pos_embed(mx.arange(T))
        h_normed = self.block.norm1(h)
        return self.block.attn.get_diagnostics(h_normed)


def count_params(tree):
    if isinstance(tree, mx.array):
        return tree.size
    elif isinstance(tree, dict):
        return sum(count_params(v) for v in tree.values())
    elif isinstance(tree, list):
        return sum(count_params(v) for v in tree)
    return 0


# ── Training ───────────────────────────────────────────────

def loss_fn(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))


def train_epoch(model, optimizer, grad_fn, train_x, train_y, batch_size):
    n = train_x.shape[0] // batch_size
    total = 0.0
    for i in range(n):
        s = i * batch_size
        loss, grads = grad_fn(model, train_x[s:s + batch_size], train_y[s:s + batch_size])
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total += loss.item()
    return total / n


def has_rossler_params(model):
    attn = model.block.attn
    return hasattr(attn, "rossler_c") or hasattr(attn, "alpha")


def record_params(model):
    attn = model.block.attn
    snap = {}
    for name in ["rossler_a", "rossler_b", "rossler_c", "rossler_lam", "alpha"]:
        if hasattr(attn, name):
            arr = getattr(attn, name)
            snap[name] = [arr[h].item() for h in range(arr.shape[0])]
    return snap


def run_training(model, train_x, train_y, config, quiet=False):
    optimizer = optim.Adam(learning_rate=config["lr"])
    grad_fn = nn.value_and_grad(model, loss_fn)
    losses = []
    param_history = []
    t0 = time.time()

    for epoch in range(config["epochs"]):
        avg = train_epoch(model, optimizer, grad_fn, train_x, train_y, config["batch_size"])
        losses.append(avg)
        if not quiet and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"    Epoch {epoch + 1:>3}/{config['epochs']}  loss: {avg:.4f}  ({time.time() - t0:.0f}s)")
        if (epoch + 1) % 10 == 0 and has_rossler_params(model):
            param_history.append({"epoch": epoch + 1, **record_params(model)})

    return losses, param_history, time.time() - t0


# ── Evaluation ─────────────────────────────────────────────

def evaluate(model, test_x, test_y, masks, batch_size=32, signal_token=None):
    total_correct = 0
    total_tokens = 0
    cat_correct = {k: 0.0 for k in masks}
    cat_tokens = {k: 0 for k in masks}
    sig_correct = 0
    sig_tokens = 0

    n_batches = (test_x.shape[0] + batch_size - 1) // batch_size
    for i in range(n_batches):
        s = i * batch_size
        x = test_x[s:s + batch_size]
        y = test_y[s:s + batch_size]
        preds = mx.argmax(model(x), axis=-1)
        correct = (preds == y).astype(mx.float32)
        mx.eval(correct)

        total_correct += mx.sum(correct).item()
        total_tokens += correct.size

        for k in masks:
            m = masks[k][s:s + batch_size]
            m_mx = mx.array(m.astype(np.float32))
            cat_correct[k] += mx.sum(correct * m_mx).item()
            cat_tokens[k] += int(m.sum())

        if signal_token is not None:
            sm = (y == signal_token).astype(mx.float32)
            sig_correct += mx.sum(correct * sm).item()
            sig_tokens += int(mx.sum(sm).item())

    results = {"overall": total_correct / max(total_tokens, 1)}
    for k in masks:
        results[k] = cat_correct[k] / max(cat_tokens[k], 1)
    if signal_token is not None:
        results["signal_detect"] = sig_correct / max(sig_tokens, 1)
    results["_counts"] = {k: cat_tokens[k] for k in masks}
    results["_counts"]["total"] = total_tokens
    return results


# ── Diagnostics ────────────────────────────────────────────

def compute_diagnostics(model, test_x, variant_name="standard", batch_size=32):
    is_rossler = variant_name in ROSSLER_VARIANTS
    all_entropy = []
    all_div = []
    all_sparsity = []
    all_reinj = []

    n = min(4, (test_x.shape[0] + batch_size - 1) // batch_size)
    for i in range(n):
        s = i * batch_size
        x = test_x[s:s + batch_size]
        diag = model.get_attn_diagnostics(x)
        A = diag["A"]
        mx.eval(A)
        B, H, T, _ = A.shape
        eps = 1e-8

        ent = -mx.sum(A * mx.log(A + eps), axis=-1)
        per_head = mx.mean(ent, axis=(0, 2))
        mx.eval(per_head)
        all_entropy.append(np.array([per_head[h].item() for h in range(H)]))

        divs = []
        for h1 in range(H):
            for h2 in range(h1 + 1, H):
                p, q = A[:, h1] + eps, A[:, h2] + eps
                kl = (mx.mean(mx.sum(p * mx.log(p / q), axis=-1))
                      + mx.mean(mx.sum(q * mx.log(q / p), axis=-1))) / 2.0
                mx.eval(kl)
                divs.append(kl.item())
        all_div.append(np.mean(divs))

        if is_rossler and "F" in diag:
            F = diag["F"]
            mx.eval(F)
            causal = mx.arange(T).reshape(T, 1) >= mx.arange(T).reshape(1, T)
            causal_f = causal.astype(mx.float32).reshape(1, 1, T, T)
            zeros = mx.sum((F < 1e-6).astype(mx.float32) * causal_f).item()
            total_unmasked = mx.sum(causal_f).item() * B * H
            all_sparsity.append(zeros / max(total_unmasked, 1))

            if "fold_output" in diag and "spiral_only" in diag and "b" in diag:
                fo = diag["fold_output"]
                so = diag["spiral_only"]
                b = diag["b"]
                r_norm = mx.sqrt(mx.sum((b * so) ** 2, axis=-1) + eps)
                f_norm = mx.sqrt(mx.sum(fo ** 2, axis=-1) + eps)
                mx.eval(r_norm, f_norm)
                all_reinj.append(mx.mean(r_norm / f_norm).item())

    results = {
        "entropy_per_head": np.mean(all_entropy, axis=0),
        "head_divergence": float(np.mean(all_div)),
    }
    if all_sparsity:
        results["fold_sparsity"] = float(np.mean(all_sparsity))
    if all_reinj:
        results["reinjection_ratio"] = float(np.mean(all_reinj))
    return results


# ── Experiment runner ──────────────────────────────────────

def run_experiment(name, attention_cls, variant_name, train_x, train_y,
                   test_x, test_y, target_masks, config=None, quiet=False,
                   signal_token=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if not quiet:
        print(f"\n  {'=' * 50}")
        print(f"  {name}")
        print(f"  {'=' * 50}")

    mx.random.seed(cfg["seed"])
    vocab_size = cfg.get("vocab_size", int(train_y.max()) + 1)
    model = Model(vocab_size, cfg["embed_dim"], cfg["num_heads"],
                  cfg["seq_len"] - 1, attention_cls)
    mx.eval(model.parameters())

    if not quiet:
        print(f"    Params: {count_params(model.parameters()):,}")

    losses, param_hist, train_time = run_training(model, train_x, train_y, cfg, quiet=quiet)

    acc = evaluate(model, test_x, test_y, target_masks, cfg["batch_size"],
                   signal_token=signal_token)
    diag = compute_diagnostics(model, test_x, variant_name, cfg["batch_size"])

    if not quiet:
        print(f"    Time: {train_time:.0f}s")
        print(f"    Overall acc: {acc['overall']:.4f}")
        for k in target_masks:
            if k.startswith("_"):
                continue
            print(f"    {k}: {acc.get(k, 0):.4f}")

    return {
        "name": name,
        "variant": variant_name,
        "losses": losses,
        "acc": acc,
        "diag": diag,
        "train_time": train_time,
        "param_history": param_hist,
        "model": model,
    }


# ── Comparison utilities ───────────────────────────────────

def compare_results(results_list, critical_mask):
    print(f"\n  {'Variant':<25} {'Overall':>8} {critical_mask:>16} {'Entropy':>8} {'Loss':>8}")
    print(f"  {'-' * 65}")
    for r in results_list:
        ent = float(np.mean(r["diag"]["entropy_per_head"]))
        crit = r["acc"].get(critical_mask, 0)
        print(f"  {r['name']:<25} {r['acc']['overall']:>8.4f} {crit:>16.4f} {ent:>8.3f} {r['losses'][-1]:>8.4f}")

    if len(results_list) > 1:
        base_crit = results_list[0]["acc"].get(critical_mask, 0)
        print(f"\n  Delta vs {results_list[0]['name']}:")
        for r in results_list[1:]:
            d = r["acc"].get(critical_mask, 0) - base_crit
            print(f"    {r['name']:<25} {d:>+.4f}")


def rank_experiments(results_list, metric_key):
    return sorted(results_list, key=lambda r: r["acc"].get(metric_key, 0), reverse=True)
