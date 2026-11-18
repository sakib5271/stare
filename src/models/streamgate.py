"""S3 — Shared Streaming State (the proposed method; see paper/proposed-method.md).

One byte-budget spatial state S is maintained with a learnable EWMA update:
    p = proj(frame)                      # cheap 1x1-conv projection to k channels
    S <- (1-alpha)*S + alpha*p           # O(1) RAM, in place (alpha learned)
    R = |proj(frame) - S|                # streaming residual = temporal smoke-growth signal
The tiny CNN then classifies [current_frame ++ R]. The SAME state also drives the wake gate
(a head on pooled R) and the duty-cycle policy — one buffer, three jobs.

Deployment vs training:
  - Deployed on the MCU, the node keeps S persistent and runs ONE frame per step: peak RAM =
    current frame + S + CNN activations — NO frame stack. That is the honest, favourable
    number reported in `meta['deploy_peak_ram_kb']`.
  - For training/eval we only have short T-frame stacks, so we UNROLL the EWMA over the stack
    (state warmed on the first T-1 frames, residual taken at the last). Same weights.

This first build is the temporal core of S3 (feature path). The shared-state gate integrates
into run_cascade (gate_type="streamstate") as a follow-up.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .backbone import build_for_budget, TFLM_OVERHEAD_KB
from ..simulate import peak_ram_kb


class StreamState(layers.Layer):
    """Unroll a learnable-alpha EWMA over a projected T-stack; return the last-frame residual.

    Input : projected frames (N, T, H, W, k)
    Output: residual R (N, H, W, k) = |p_last - S|, with S warmed over p_0..p_{T-2}.
    alpha = sigmoid(w) keeps the mixing coefficient in (0,1) and is trained end-to-end.
    """

    def __init__(self, alpha_init=0.5, **kw):
        super().__init__(**kw)
        self._a0 = float(np.log(alpha_init / (1 - alpha_init)))   # inverse-sigmoid

    def build(self, input_shape):
        self.alpha_w = self.add_weight(
            name="alpha", shape=(), trainable=True,
            initializer=keras.initializers.Constant(self._a0))

    def call(self, p):
        a = tf.sigmoid(self.alpha_w)
        T = p.shape[1]
        S = p[:, 0]
        for t in range(1, max(1, T - 1)):            # warm the state on the past frames
            S = (1.0 - a) * S + a * p[:, t]
        return tf.abs(p[:, -1] - S)                   # residual at the current frame

    def get_config(self):
        return {**super().get_config(), "alpha_init": float(tf.sigmoid(self._a0))}


def build_streamgate(input_size=64, budget_kb=64, k=2, temporal_frames=3):
    """Return (train_model, meta). train_model: (N,T,H,W,3) -> logit.
    meta carries width/params and the HONEST single-step deployment RAM (frame + state)."""
    H = W = input_size
    T = temporal_frames

    # budget-fit the CNN on the DEPLOYMENT input shape (frame ++ residual = 3+k channels),
    # so flash/RAM reflect what actually ships (a single frame + state, not the T-stack).
    bb, meta = build_for_budget((H, W, 3 + k), budget_kb, name="s3_backbone")

    # --- training graph: unroll EWMA over the stack, then run the shared CNN ---
    stack_in = keras.Input(shape=(T, H, W, 3), name="stack")
    proj = layers.Conv2D(k, 1, padding="same", use_bias=False, name="proj")
    p = layers.TimeDistributed(proj, name="td_proj")(stack_in)     # (N,T,H,W,k)
    R = StreamState(name="ewma")(p)                                # (N,H,W,k)
    last = layers.Lambda(lambda z: z[:, -1], name="last_frame")(stack_in)
    x = layers.Concatenate(name="frame_plus_residual")([last, R])  # (N,H,W,3+k)
    logit = bb(x)
    train_model = keras.Model(stack_in, logit, name="s3_streamgate")

    # honest deployment peak RAM = CNN single-frame activations + persistent state S (H*W*k)
    persistent_state_kb = round(H * W * k / 1024.0, 1)
    proj_params = 3 * k                                            # 1x1 conv, no bias
    meta.update({
        "method": "S3", "k": k, "temporal_frames": T,
        "proj_params": proj_params,
        "persistent_state_kb": persistent_state_kb,
        "deploy_peak_ram_kb": round(peak_ram_kb(bb) + persistent_state_kb, 1),
        "params": meta["params"] + proj_params + 1,               # +alpha
    })
    return train_model, meta
