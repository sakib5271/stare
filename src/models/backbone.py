"""Budget-fitting tiny CNN backbone (int8-friendly, MCU-plausible).

Depthwise-separable blocks (CMSIS-NN friendly). A single scalar `width` scales all
channels; we search `width` so the int8 model lands just under a target flash budget.
int8 weights are ~1 byte/param, so params(KB) ~= flash(KB) to first order; simulate.py
reports the exact .tflite size after quantization.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _dw_sep(x, out_ch, stride, name):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False,
                               name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dwbn")(x)
    x = layers.ReLU(6.0, name=f"{name}_dwrelu")(x)
    x = layers.Conv2D(out_ch, 1, padding="same", use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pwbn")(x)
    x = layers.ReLU(6.0, name=f"{name}_pwrelu")(x)
    return x


def build_backbone(input_shape, width, extra_feat_dim=0, name="tinycnn"):
    """Return (model, gap_tensor_model_none). Model outputs a single logit (binary).

    If extra_feat_dim>0 the model takes a second input (handcrafted features) that is
    concatenated with the global-avg-pooled CNN features before the classifier head.
    """
    img_in = keras.Input(shape=input_shape, name="image")
    x = layers.Conv2D(width, 3, strides=2, padding="same", use_bias=False,
                      name="stem")(img_in)                       # /2
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(6.0, name="stem_relu")(x)
    x = _dw_sep(x, width * 2, 2, "b1")                            # /4
    x = _dw_sep(x, width * 4, 2, "b2")                            # /8
    x = _dw_sep(x, width * 4, 2, "b3")                            # /16
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    inputs = [img_in]
    if extra_feat_dim > 0:
        feat_in = keras.Input(shape=(extra_feat_dim,), name="handcrafted")
        f = layers.BatchNormalization(name="feat_bn")(feat_in)
        x = layers.Concatenate(name="concat")([x, f])
        inputs.append(feat_in)
    out = layers.Dense(1, name="logit")(x)                        # linear logit
    return keras.Model(inputs, out, name=name)


def count_params(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


# A tiny TFLM .tflite carries ~30 KB of operator tables + metadata beyond the int8
# weights. The budget is the MODEL (flash) size, so we fit WEIGHTS to budget - overhead.
# run_cell overrides meta['fits_budget'] with the true post-quantization flash size.
TFLM_OVERHEAD_KB = 32


def build_for_budget(input_shape, budget_kb, extra_feat_dim=0, headroom=0.90,
                     width_range=range(2, 80), name="tinycnn"):
    """Pick the largest width whose int8 weights fit (budget - TFLM overhead) AND whose
    peak activation RAM fits the budget. Both constraints matter on an MCU: flash holds
    the weights, SRAM holds the activation arena (the binding constraint at 96x96).
    Returns (model, meta)."""
    from ..simulate import peak_ram_kb    # lazy import (simulate depends only on TF)
    weight_budget_kb = max(2.0, budget_kb - TFLM_OVERHEAD_KB)
    target_params = int(weight_budget_kb * 1024 * headroom)   # 1 byte per int8 weight
    best = None          # largest width satisfying BOTH constraints
    smallest = None      # fallback if nothing fits (honest fits_budget=False)
    for w in width_range:
        m = build_backbone(input_shape, w, extra_feat_dim, name)
        p = count_params(m)
        ram = peak_ram_kb(m)
        if smallest is None:
            smallest = (w, p, m, ram)
        weights_ok = p <= target_params
        ram_ok = ram <= budget_kb
        if weights_ok and ram_ok:
            best = (w, p, m, ram)
        elif p > target_params:
            break        # params grow monotonically; past flash budget -> stop
    w, p, m, ram = best if best is not None else smallest
    est_flash_kb = round(p / 1024.0 + TFLM_OVERHEAD_KB, 1)
    meta = {"width": w, "params": p, "est_flash_kb": est_flash_kb,
            "est_peak_ram_kb": ram, "budget_kb": budget_kb,
            "fits_budget": est_flash_kb <= budget_kb and ram <= budget_kb}
    return m, meta
