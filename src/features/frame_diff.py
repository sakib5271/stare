"""Cheap temporal front-end (H1): frame differencing over a short stack.

Smoke grows slowly; a few bytes of temporal signal (abs frame diffs) exposes that
motion without an LSTM/ViT. On an MCU this is a subtraction over the previous frame
held in a ring buffer, i.e. fixed DSP with a few hundred bytes of state.

Input : X_temporal  (N, T, H, W, 3)  uint8 or float in [0,1]
Output: (N, H, W, 3 + 3*(T-1))       last frame  ++  abs diffs between consecutive frames
"""
import numpy as np


def temporal_stack(x_temporal):
    x = np.asarray(x_temporal)
    if x.dtype == np.uint8:
        x = x.astype("float32") / 255.0
    else:
        x = x.astype("float32")
    n, t, h, w, c = x.shape
    if t < 2:
        # No temporal signal available; degrade gracefully to single-frame.
        return x[:, 0]
    last = x[:, -1]                                   # (N,H,W,3) the current frame
    diffs = np.abs(x[:, 1:] - x[:, :-1])              # (N,T-1,H,W,3)
    diffs = diffs.transpose(0, 2, 3, 1, 4).reshape(n, h, w, (t - 1) * c)
    return np.concatenate([last, diffs], axis=-1)     # (N,H,W, 3 + 3*(T-1))


def temporal_channels(t):
    """Channel count produced by temporal_stack for a stack of length t."""
    return 3 if t < 2 else 3 + 3 * (t - 1)
