"""Streaming background subtraction — 'memory in a few bytes' (H1).

Running-mean background model updated per frame:  bg <- (1-a)*bg + a*frame.
The foreground residual |frame - bg| highlights a slowly-appearing smoke plume
against a static scene. On an MCU this is one EWMA buffer (H*W*3 bytes) — no
recurrent network. Provided as an alternative temporal front-end to frame_diff.
"""
import numpy as np


def foreground_residual(x_temporal, alpha=0.5):
    """Return |current - running_mean_background| for the last frame of each stack.

    x_temporal: (N, T, H, W, 3). Returns (N, H, W, 3) foreground residual, plus the
    current frame concatenated => (N, H, W, 6), matching a 'frame + motion' input.
    """
    x = np.asarray(x_temporal)
    if x.dtype == np.uint8:
        x = x.astype("float32") / 255.0
    else:
        x = x.astype("float32")
    n, t, h, w, c = x.shape
    bg = x[:, 0].copy()
    for i in range(1, t):
        bg = (1.0 - alpha) * bg + alpha * x[:, i]
    cur = x[:, -1]
    resid = np.abs(cur - bg)
    return np.concatenate([cur, resid], axis=-1)      # (N,H,W,6)
