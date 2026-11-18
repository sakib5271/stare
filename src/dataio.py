"""Load processed splits and prepare mechanism-specific model inputs.

Data is stored once at the native 96x96 resolution; `size` downsamples on the fly so
input resolution can be swept (48/64/96) as an ablation axis without regenerating data.
Lower resolution is the main lever for fitting KB-RAM (peak activation shrinks ~size^2).
"""
import os
import numpy as np

from .features.frame_diff import temporal_stack
from .features.color_texture import handcrafted_features

PROCESSED = os.environ.get("WILDFIRE_DATA", "data/processed")


def _resize(x, size):
    """Resize (...,H,W,C) float image(s) to size x size via tf.image (bilinear)."""
    import tensorflow as tf
    orig = x.shape
    flat = x.reshape(-1, orig[-3], orig[-2], orig[-1])          # (M,H,W,C)
    out = tf.image.resize(flat, (size, size), method="bilinear").numpy()
    return out.reshape(*orig[:-3], size, size, orig[-1]).astype("float32")


def load_split(region, split, root=PROCESSED, size=None):
    """Return (X_single float[0,1] (N,S,S,3), X_temporal float[0,1] (N,T,S,S,3), y).
    If `size` is None or equals the native resolution, no resize is done."""
    s = np.load(os.path.join(root, region, f"{split}_single.npz"))
    t = np.load(os.path.join(root, region, f"{split}_temporal.npz"))
    xs = s["X"].astype("float32") / 255.0
    xt = t["X"].astype("float32") / 255.0
    y = s["y"].astype("float32")
    if size is not None and size != xs.shape[1]:
        xs = _resize(xs, size)
        xt = _resize(xt, size)
    return xs, xt, y


def prepare_inputs(kind, xs, xt):
    """Map a mechanism's 'inputs' tag -> the actual array(s) fed to the keras model."""
    if kind == "single":
        return xs
    if kind == "temporal":
        return temporal_stack(xt)
    if kind == "handcrafted":
        return [xs, handcrafted_features(xs, xt)]
    if kind == "stack":
        return xt                       # raw T-frame stack (N,T,H,W,3) for S3 streaming state
    raise ValueError(f"unknown input kind {kind!r}")
