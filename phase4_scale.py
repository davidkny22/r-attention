"""Phase 4: Scale Test

Character-level language modeling on Shakespeare.
Compare refined mechanism vs standard at larger scale.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import os
import urllib.request

from attention import ATTENTION_VARIANTS, StandardAttention
from engine import (FeedForward, TransformerBlock, count_params,
                    loss_fn, train_epoch)


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/shakespeare.txt"


class MultiLayerModel(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len, attention_cls):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.blocks = [TransformerBlock(dim, num_heads, attention_cls) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(dim)
        self.out_head = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        B, T = x.shape
        x = self.token_embed(x) + self.pos_embed(mx.arange(T))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.out_head(x)

    def get_attn_diagnostics(self, x):
        B, T = x.shape
        h = self.token_embed(x) + self.pos_embed(mx.arange(T))
        h_normed = self.blocks[0].norm1(h)
        return self.blocks[0].attn.get_diagnostics(h_normed)


def download_shakespeare():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"  Downloading Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    with open(DATA_PATH, "r") as f:
        text = f.read()
    return text


def prepare_data(text, seq_len=256):
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    encoded = np.array([char_to_idx[c] for c in text], dtype=np.int32)

    split = int(len(encoded) * 0.9)
    train_data = encoded[:split]
    test_data = encoded[split:]

    def make_sequences(data, sl):
        n = len(data) // sl
        return data[:n * sl].reshape(n, sl)

    return make_sequences(train_data, seq_len), make_sequences(test_data, seq_len), len(chars), chars


def compute_bpc(model, test_x, test_y, batch_size=32):
    total_loss = 0.0
    total_tokens = 0
    n = (test_x.shape[0] + batch_size - 1) // batch_size

    for i in range(n):
        s = i * batch_size
        x = test_x[s:s + batch_size]
        y = test_y[s:s + batch_size]
        logits = model(x)
        loss = mx.sum(nn.losses.cross_entropy(logits, y))
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size

    return (total_loss / total_tokens) / np.log(2)


def run_scale_test(variant_names):
    print("=" * 70)
    print("  PHASE 4 — Scale Test (Character-Level Shakespeare)")
    print(f"  Variants: {variant_names}")
    print("=" * 70)

    text = download_shakespeare()
    seq_len = 256
    train_seqs, test_seqs, vocab_size, chars = prepare_data(text, seq_len)

    print(f"  Vocab: {vocab_size} chars")
    print(f"  Train: {train_seqs.shape[0]} sequences")
    print(f"  Test: {test_seqs.shape[0]} sequences")

    train_x = mx.array(train_seqs[:, :-1])
    train_y = mx.array(train_seqs[:, 1:])
    test_x = mx.array(test_seqs[:, :-1])
    test_y = mx.array(test_seqs[:, 1:])

    dim = 128
    num_heads = 8
    num_layers = 2
    epochs = 50
    batch_size = 32
    lr = 3e-4

    results = []

    for vn in variant_names:
        attn_cls = ATTENTION_VARIANTS[vn]

        print(f"\n  {'=' * 50}")
        print(f"  {vn}")
        print(f"  {'=' * 50}")

        mx.random.seed(42)
        model = MultiLayerModel(vocab_size, dim, num_heads, num_layers,
                                seq_len - 1, attn_cls)
        mx.eval(model.parameters())
        n_params = count_params(model.parameters())
        print(f"    Params: {n_params:,}")

        optimizer = optim.Adam(learning_rate=lr)
        grad_fn = nn.value_and_grad(model, loss_fn)

        losses = []
        t0 = time.time()
        for epoch in range(epochs):
            avg = train_epoch(model, optimizer, grad_fn, train_x, train_y, batch_size)
            losses.append(avg)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(f"    Epoch {epoch + 1:>3}/{epochs}  loss: {avg:.4f}  ({elapsed:.0f}s)")

        train_time = time.time() - t0
        bpc = compute_bpc(model, test_x, test_y, batch_size)
        print(f"\n    BPC: {bpc:.4f}")
        print(f"    Time: {train_time:.0f}s")

        results.append({
            "variant": vn,
            "bpc": bpc,
            "losses": losses,
            "train_time": train_time,
            "n_params": n_params,
        })

    # Comparison
    print(f"\n{'=' * 70}")
    print(f"  SCALE TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  {'Variant':<22} {'BPC':>8} {'Params':>10} {'Time':>8}")
    print(f"  {'-' * 50}")

    for r in results:
        print(f"  {r['variant']:<22} {r['bpc']:>8.4f} {r['n_params']:>10,} {r['train_time']:>7.0f}s")

    if len(results) > 1:
        base_bpc = results[0]["bpc"]
        print(f"\n  Delta vs {results[0]['variant']}:")
        for r in results[1:]:
            d = r["bpc"] - base_bpc
            direction = "better" if d < 0 else "worse"
            print(f"    {r['variant']:<22} {d:>+.4f} BPC ({direction})")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        variants = sys.argv[1:]
    else:
        variants = ["standard", "fold_differential", "differential", "rossler"]
    run_scale_test(variants)
