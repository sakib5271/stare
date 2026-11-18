"""Handcrafted smoke cues (H3): a few cheap global stats that may transfer across
biomes better than raw CNN features.

All are O(pixels) reductions computable on an MCU in a single pass:
  - per-channel mean & std (smoke desaturates / greys the scene)
  - grayscale mean & std (brightness / haze)
  - mean gradient magnitude (texture; smoke softens edges)
  - "greyness": 1 - mean saturation proxy (max-min over channels)
  - optional motion energy (mean abs frame-diff) if a temporal stack is given

Returns a small fixed-length vector per sample, concatenated with CNN features in
the handcrafted-hybrid model.
"""
import numpy as np

HANDCRAFTED_DIM = 11


def _grad_mag_mean(gray):
    gy = np.abs(np.diff(gray, axis=1))   # (N,H-1,W)
    gx = np.abs(np.diff(gray, axis=2))   # (N,H,W-1)
    return gy.reshape(gy.shape[0], -1).mean(1) + gx.reshape(gx.shape[0], -1).mean(1)


def handcrafted_features(x_single, x_temporal=None):
    """x_single: (N,H,W,3) uint8/float. x_temporal optional (N,T,H,W,3) for motion.
    Returns (N, HANDCRAFTED_DIM) float32."""
    x = np.asarray(x_single)
    x = x.astype("float32") / 255.0 if x.dtype == np.uint8 else x.astype("float32")
    n = x.shape[0]
    flat = x.reshape(n, -1, 3)
    ch_mean = flat.mean(1)                                   # (N,3)
    ch_std = flat.std(1)                                     # (N,3)
    gray = x.mean(-1)                                        # (N,H,W)
    g_mean = gray.reshape(n, -1).mean(1, keepdims=True)      # (N,1)
    g_std = gray.reshape(n, -1).std(1, keepdims=True)        # (N,1)
    grad = _grad_mag_mean(gray).reshape(n, 1)               # (N,1)
    greyness = 1.0 - (x.max(-1) - x.min(-1)).reshape(n, -1).mean(1, keepdims=True)
    if x_temporal is not None:
        xt = np.asarray(x_temporal)
        xt = xt.astype("float32") / 255.0 if xt.dtype == np.uint8 else xt.astype("float32")
        if xt.shape[1] >= 2:
            motion = np.abs(xt[:, -1] - xt[:, -2]).reshape(n, -1).mean(1, keepdims=True)
        else:
            motion = np.zeros((n, 1), "float32")
    else:
        motion = np.zeros((n, 1), "float32")
    feats = np.concatenate([ch_mean, ch_std, g_mean, g_std, grad, greyness, motion], axis=1)
    return feats.astype("float32")                          # (N,11)
