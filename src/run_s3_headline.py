"""Headline S3 experiment: raw vs streaming-background-residual, tiny CNN @64KB, FIgLib
camera-holdout, >=3 seeds (mean +/- std), with int8 PTQ delta + latency/RAM.

  python -m src.run_s3_headline --seeds 3 --alpha 0.001

Loads RAW FIgLib frames, builds a causal online-EWMA background residual (the deployable
mechanism), and reports float & int8 PR-AUC (mean+/-std), the float->int8 delta, and
flash / peak-RAM (incl. the persistent background state) / latency from the analytical
Cortex-M model (Renode not installed here; swap in via src/simulate when available).
"""
import argparse
import os
import re
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from .models.backbone import build_for_budget
from .quantize import to_int8, tflite_predict
from .simulate import simulate
from .config import DEFAULT_MCU
from .metrics import recall_at_fpr

RES = 96
RAW = "data/raw/figlib"
MARGIN = 600
IMG_RE = re.compile(r"(\d{10})_([+-]\d+)\.jpg$")


def _cam(seq):
    p = seq.split("_"); return "_".join(p[2:]) if len(p) > 2 else seq


def load_figlib(alpha, max_seq=200):
    """Return raw, residual, y, cams (clear frames only), residual via causal online EWMA."""
    from PIL import Image
    def load(path):
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB").resize((RES, RES), Image.BILINEAR), "float32") / 255.0
    raws, resids, ys, cams = [], [], [], []
    seqs = [s for s in sorted(os.listdir(RAW)) if os.path.isdir(os.path.join(RAW, s))][:max_seq]
    for seq in seqs:
        d = os.path.join(RAW, seq)
        fr = sorted((int(m.group(2)), os.path.join(d, f))
                    for f in os.listdir(d) for m in [IMG_RE.search(f)] if m)
        if len(fr) < 6:
            continue
        S = None
        for off, path in fr:
            img = load(path)
            if S is None:
                S = img.copy()
            r = np.abs(img - S)
            S = (1 - alpha) * S + alpha * img
            if abs(off) >= MARGIN:
                raws.append(img); resids.append(r)
                ys.append(int(off >= 0)); cams.append(_cam(seq))
    return (np.array(raws, "float32"), np.array(resids, "float32"),
            np.array(ys), np.array(cams))


def camera_holdout(cams):
    u = sorted(set(cams)); rng = np.random.default_rng(1)
    tc = {u[k] for k in rng.permutation(len(u))[:max(1, len(u) // 5)]}
    te = np.array([i for i in range(len(cams)) if cams[i] in tc])
    tr = np.array([i for i in range(len(cams)) if cams[i] not in tc])
    return tr, te


def train_eval(X, y, tr, te, seed, budget, name, target_fpr=0.01):
    tf.keras.utils.set_random_seed(seed)
    model, meta = build_for_budget((RES, RES, X.shape[-1]), budget, name=f"{name}_s{seed}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    pos = y[tr].sum(); neg = len(tr) - pos
    model.fit(X[tr], y[tr], epochs=20, batch_size=64, verbose=0,
              class_weight={0: 1.0, 1: neg / max(pos, 1)})
    f = 1 / (1 + np.exp(-model.predict(X[te], verbose=0).ravel()))
    ap_f = average_precision_score(y[te], f)
    # int8 PTQ
    blob, flash_kb = to_int8(model, X[tr][:200])
    q = tflite_predict(blob, X[te])
    ap_q = average_precision_score(y[te], q)
    rec_q, _ = recall_at_fpr(y[te], q, target_fpr)
    return ap_f, ap_q, rec_q, model, meta, flash_kb


def agg(v):
    return float(np.mean(v)), float(np.std(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=200)
    args = ap.parse_args()

    raws, resids, y, cams = load_figlib(args.alpha, args.max_seq)
    tr, te = camera_holdout(cams)
    print(f"FIgLib clear frames: {len(y)} | cameras {len(set(cams))} | "
          f"train {len(tr)} / test {len(te)} | base {y[te].mean():.3f} | alpha {args.alpha}")
    mcu = dict(DEFAULT_MCU)

    results = {}
    for name, X in [("raw", raws), ("residual", resids)]:
        fs, qs, recs = [], [], []
        last = None
        for seed in range(args.seeds):
            ap_f, ap_q, rec_q, model, meta, flash_kb = train_eval(X, y, tr, te, seed,
                                                                  args.budget, name)
            fs.append(ap_f); qs.append(ap_q); recs.append(rec_q)
            last = (model, meta, flash_kb)
        model, meta, flash_kb = last
        sim = simulate(model, flash_kb, mcu)
        state_kb = round(RES * RES * X.shape[-1] / 1024.0, 1) if name == "residual" else 0.0
        peak_ram = round(sim["peak_ram_kb"] + state_kb, 1)
        results[name] = dict(prauc_f=agg(fs), prauc_q=agg(qs), recall=agg(recs),
                             int8_delta=round(agg(fs)[0] - agg(qs)[0], 4),
                             width=meta["width"], params=meta["params"], flash_kb=flash_kb,
                             peak_ram_kb=peak_ram, state_kb=state_kb,
                             latency_ms=sim["latency_ms"], macs=sim["macs"],
                             latency_src=sim["latency_source"])
    _print(results, args)


def _print(r, args):
    print(f"\n===== S3 HEADLINE (FIgLib camera-holdout, {args.seeds} seeds, "
          f"tiny CNN @{args.budget}KB, online bg alpha={args.alpha}) =====")
    print(f"{'':<10}{'PR-AUC float':>16}{'PR-AUC int8':>16}{'int8 drop':>11}"
          f"{'rec@1%FPR':>11}")
    for k in ("raw", "residual"):
        d = r[k]
        print(f"{k:<10}{d['prauc_f'][0]:>8.3f}±{d['prauc_f'][1]:<6.3f}"
              f"{d['prauc_q'][0]:>8.3f}±{d['prauc_q'][1]:<6.3f}"
              f"{d['int8_delta']:>11.4f}{d['recall'][0]:>9.3f}±{d['recall'][1]:.3f}")
    d = r["residual"]
    print(f"\n  deployed residual model: width={d['width']} params={d['params']} "
          f"flash={d['flash_kb']}KB")
    print(f"  peak RAM={d['peak_ram_kb']}KB (activations + {d['state_kb']}KB persistent "
          f"background state) | latency={d['latency_ms']}ms [{d['latency_src']}] "
          f"| MACs={d['macs']}")


if __name__ == "__main__":
    main()
