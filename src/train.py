"""Config-driven training for one ablation cell / one seed.

Trains the budget-fitted mechanism on the in-region train split (§9), returns the
float32 keras model + validation metrics. run_cell.py handles multi-seed aggregation,
quantization, simulation, energy and cross-region evaluation.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .models import build_mechanism
from .dataio import load_split, prepare_inputs
from .metrics import eval_scores


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_one(cfg, seed, verbose=0):
    set_seed(seed)
    built = build_mechanism(cfg["mechanism"], cfg["budget_kb"],
                            cfg.get("input_size", 96), cfg.get("temporal_frames", 3))
    model, meta, kind = built["model"], built["meta"], built["inputs"]

    size = cfg.get("input_size", 96)
    xs_tr, xt_tr, y_tr = load_split(cfg["train_region"], "train", size=size)
    xs_va, xt_va, y_va = load_split(cfg["train_region"], "val", size=size)
    X_tr = prepare_inputs(kind, xs_tr, xt_tr)
    X_va = prepare_inputs(kind, xs_va, xt_va)

    # class weights (fires are rarer in the real sets; harmless when balanced)
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    cw = {0: 1.0, 1: neg / max(pos, 1.0)}

    model.compile(
        optimizer=keras.optimizers.Adam(cfg.get("lr", 1e-3)),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.AUC(name="pr_auc", curve="PR", from_logits=True)],
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=cfg.get("epochs", 12), batch_size=cfg.get("batch_size", 64),
              class_weight=cw, verbose=verbose)

    val_logits = model.predict(X_va, verbose=0).ravel()
    val_scores = 1.0 / (1.0 + np.exp(-val_logits))
    val_metrics = eval_scores(y_va, val_scores, cfg.get("target_fpr", 0.01))
    return model, meta, kind, val_metrics
