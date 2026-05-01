"""Six attention variants for ablation testing.

All share the same interface:
  __call__(self, x) -> output tensor
  get_diagnostics(self, x) -> dict of intermediate values
"""

import mlx.core as mx
import mlx.nn as nn


def make_causal_mask(T):
    rows = mx.arange(T).reshape(T, 1)
    cols = mx.arange(T).reshape(1, T)
    return mx.where(rows >= cols, mx.array(0.0), mx.array(-1e9))


class _BaseAttention(nn.Module):
    """Shared QKV projection logic."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def _project(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        return Q, K, V

    def _output(self, O, B, T, D):
        return self.o_proj(O.transpose(0, 2, 1, 3).reshape(B, T, D))


# ── 1. Standard Attention ───────────────────────────────────

class StandardAttention(_BaseAttention):

    def __call__(self, x):
        B, T, D = x.shape
        Q, K, V = self._project(x)
        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)
        A = mx.softmax(S, axis=-1)
        return self._output(A @ V, B, T, D)

    def get_diagnostics(self, x):
        B, T, D = x.shape
        Q, K, V = self._project(x)
        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)
        A = mx.softmax(S, axis=-1)
        return {"A": A}


# ── 2. Fold-Only Attention ──────────────────────────────────

class FoldOnlyAttention(_BaseAttention):
    """Standard + score-based fold. No spiral, no reinjection."""

    def __init__(self, dim, num_heads):
        super().__init__(dim, num_heads)
        self.rossler_c = mx.array([0.0] * num_heads)
        self.rossler_lam = mx.array([0.1] * num_heads)

    def _forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        Q, K, V = self._project(x)

        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)

        c = self.rossler_c.reshape(1, H, 1, 1)
        lam = self.rossler_lam.reshape(1, H, 1, 1)

        v_energy = mx.sqrt(mx.sum(V * V, axis=-1, keepdims=True)).transpose(0, 1, 3, 2)
        F = v_energy * nn.relu(S - c)
        A = mx.softmax(S - lam * F, axis=-1)
        O = A @ V
        return O, A, F, S, V, v_energy

    def __call__(self, x):
        B, T, D = x.shape
        O, *_ = self._forward(x)
        return self._output(O, B, T, D)

    def get_diagnostics(self, x):
        O, A, F, S, V, v_energy = self._forward(x)
        return {"A": A, "F": F, "v_energy": v_energy}


# ── 3. Spiral-Only Attention ────────────────────────────────

class SpiralOnlyAttention(_BaseAttention):
    """Standard + key self-feedback. No fold, no reinjection."""

    def __init__(self, dim, num_heads):
        super().__init__(dim, num_heads)
        d = dim // num_heads
        self.rossler_a = mx.array([0.1] * num_heads)
        self.w_self = mx.random.normal(shape=(num_heads, d, d)) * (2.0 / (d + d)) ** 0.5

    def __call__(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q, K, V = self._project(x)

        a = self.rossler_a.reshape(1, H, 1, 1)
        K_spiral = K + a * (K @ self.w_self.reshape(1, H, d, d))

        S = (Q @ K_spiral.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)
        A = mx.softmax(S, axis=-1)
        return self._output(A @ V, B, T, D)

    def get_diagnostics(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q, K, V = self._project(x)
        a = self.rossler_a.reshape(1, H, 1, 1)
        K_spiral = K + a * (K @ self.w_self.reshape(1, H, d, d))
        S = (Q @ K_spiral.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)
        A = mx.softmax(S, axis=-1)
        return {"A": A}


# ── 4. Differential Attention ───────────────────────────────

class DifferentialAttention(nn.Module):
    """Two independent attention heads, output is their difference.

    Formalizes the pattern the network kept discovering (negative b in Rossler).
    O = A1 @ V1 - alpha * A2 @ V2, with alpha learnable per-head.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q1_proj = nn.Linear(dim, dim)
        self.k1_proj = nn.Linear(dim, dim)
        self.v1_proj = nn.Linear(dim, dim)

        self.q2_proj = nn.Linear(dim, dim)
        self.k2_proj = nn.Linear(dim, dim)
        self.v2_proj = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)
        self.alpha = mx.array([0.1] * num_heads)

    def _attend(self, q_proj, k_proj, v_proj, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q = q_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        K = k_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = v_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)
        A = mx.softmax(S, axis=-1)
        return A, V

    def __call__(self, x):
        B, T, D = x.shape
        H = self.num_heads
        A1, V1 = self._attend(self.q1_proj, self.k1_proj, self.v1_proj, x)
        A2, V2 = self._attend(self.q2_proj, self.k2_proj, self.v2_proj, x)
        alpha = self.alpha.reshape(1, H, 1, 1)
        O = (A1 @ V1) - alpha * (A2 @ V2)
        return self.o_proj(O.transpose(0, 2, 1, 3).reshape(B, T, D))

    def get_diagnostics(self, x):
        H = self.num_heads
        A1, V1 = self._attend(self.q1_proj, self.k1_proj, self.v1_proj, x)
        A2, V2 = self._attend(self.q2_proj, self.k2_proj, self.v2_proj, x)
        return {"A": A1, "A2": A2, "alpha": self.alpha}


# ── 5. Fold + Differential Attention ────────────────────────

class FoldDifferentialAttention(_BaseAttention):
    """Fold-modified attention minus scaled standard attention.

    O = softmax(S - lam*F) @ V  -  alpha * softmax(S) @ V
    Combines fold suppression with differential contrast.
    """

    def __init__(self, dim, num_heads):
        super().__init__(dim, num_heads)
        self.rossler_c = mx.array([0.0] * num_heads)
        self.rossler_lam = mx.array([0.1] * num_heads)
        self.alpha = mx.array([0.1] * num_heads)

    def _forward(self, x):
        B, T, D = x.shape
        H = self.num_heads
        Q, K, V = self._project(x)

        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)

        c = self.rossler_c.reshape(1, H, 1, 1)
        lam = self.rossler_lam.reshape(1, H, 1, 1)
        alpha = self.alpha.reshape(1, H, 1, 1)

        v_energy = mx.sqrt(mx.sum(V * V, axis=-1, keepdims=True)).transpose(0, 1, 3, 2)
        F = v_energy * nn.relu(S - c)

        A_fold = mx.softmax(S - lam * F, axis=-1)
        A_std = mx.softmax(S, axis=-1)

        O = (A_fold @ V) - alpha * (A_std @ V)
        return O, A_fold, A_std, F, V, v_energy

    def __call__(self, x):
        B, T, D = x.shape
        O, *_ = self._forward(x)
        return self._output(O, B, T, D)

    def get_diagnostics(self, x):
        O, A_fold, A_std, F, V, v_energy = self._forward(x)
        return {"A": A_fold, "A_std": A_std, "F": F, "v_energy": v_energy}


# ── 6. Full Rossler Attention ───────────────────────────────

class RosslerAttention(_BaseAttention):
    """Full Rossler mechanism: fold + spiral + reinjection."""

    def __init__(self, dim, num_heads):
        super().__init__(dim, num_heads)
        d = dim // num_heads
        self.rossler_a = mx.array([0.1] * num_heads)
        self.rossler_b = mx.array([0.01] * num_heads)
        self.rossler_c = mx.array([0.0] * num_heads)
        self.rossler_lam = mx.array([0.1] * num_heads)
        self.w_self = mx.random.normal(shape=(num_heads, d, d)) * (2.0 / (d + d)) ** 0.5

    def _forward(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q, K, V = self._project(x)

        a = self.rossler_a.reshape(1, H, 1, 1)
        b = self.rossler_b.reshape(1, H, 1, 1)
        c = self.rossler_c.reshape(1, H, 1, 1)
        lam = self.rossler_lam.reshape(1, H, 1, 1)

        K_spiral = K + a * (K @ self.w_self.reshape(1, H, d, d))
        S = (Q @ K_spiral.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)

        v_energy = mx.sqrt(mx.sum(V * V, axis=-1, keepdims=True)).transpose(0, 1, 3, 2)
        F = v_energy * nn.relu(S - c)
        A = mx.softmax(S - lam * F, axis=-1)

        spiral_only = mx.softmax(S, axis=-1) @ V
        fold_output = A @ V
        O = fold_output + b * spiral_only

        return O, A, F, S, V, v_energy, spiral_only, fold_output, b

    def __call__(self, x):
        B, T, D = x.shape
        O, *_ = self._forward(x)
        return self._output(O, B, T, D)

    def get_diagnostics(self, x):
        O, A, F, S, V, v_energy, spiral_only, fold_output, b = self._forward(x)
        return {"A": A, "F": F, "v_energy": v_energy,
                "spiral_only": spiral_only, "fold_output": fold_output, "b": b}


# Registry for easy iteration
ATTENTION_VARIANTS = {
    "standard": StandardAttention,
    "fold_only": FoldOnlyAttention,
    "spiral_only": SpiralOnlyAttention,
    "differential": DifferentialAttention,
    "fold_differential": FoldDifferentialAttention,
    "rossler": RosslerAttention,
}

# Which variants have Rossler-specific diagnostics
ROSSLER_VARIANTS = {"fold_only", "fold_differential", "rossler"}
