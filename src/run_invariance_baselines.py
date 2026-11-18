"""Do the STANDARD cures for background-shortcut learning close the cross-camera gap the way
STARE's residual representation does? This compares, under the SAME camera-holdout and the SAME
MCU-budget backbone:

  raw (no invariance)  |  raw + heavy color/lighting augmentation  |  raw + domain-adversarial
  (DANN, gradient-reversal camera head)  |  STARE (K=3, deployed)

The first three take RAW pixels and try to become camera-invariant by training; STARE removes the
camera background at the input. If augmentation/DANN match STARE, STARE's contribution is
efficiency; if they don't, the residual representation is doing something training cannot.

  python -m src.run_invariance_baselines --seeds 1 --max-seq 150
"""
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from .run_baselines import load, split, make_rep, RES, MARGIN
from .run_adaptive import RATES, stare_rep          # K=3 deployed residuals
from .models.backbone import build_for_budget

L = tf.keras.layers


class ColorJitter(L.Layer):
    """Training-only color/lighting jitter simulating cross-camera variation."""
    def __init__(self, b=0.3, c=0.3, h=0.1, s=0.3, **kw):
        super().__init__(**kw); self.b, self.c, self.h, self.s = b, c, h, s
    def call(self, x, training=None):
        if not training:
            return x
        x = tf.image.random_brightness(x, self.b)
        x = tf.image.random_contrast(x, 1 - self.c, 1 + self.c)
        x = tf.image.random_hue(x, self.h)
        x = tf.image.random_saturation(x, 1 - self.s, 1 + self.s)
        return tf.clip_by_value(x, 0.0, 1.0)


def make_grl(lamb):
    @tf.custom_gradient
    def f(x):
        def grad(dy):
            return -lamb * dy
        return tf.identity(x), grad
    return f


class GRL(L.Layer):
    """Gradient reversal: forward identity, backward multiplies gradient by -lambda."""
    def __init__(self, lamb=1.0, **kw):
        super().__init__(**kw); self._f = make_grl(lamb)
    def call(self, x):
        return self._f(x)


def _predict(model, X):
    return 1 / (1 + np.exp(-model.predict(X, verbose=0).ravel()))


def fit_plain(Xtr, ytr, seed, aug=False):
    tf.keras.utils.set_random_seed(seed)
    base, _ = build_for_budget((RES, RES, Xtr.shape[-1]), 64, name=f"p{seed}")
    if aug:
        inp = tf.keras.Input((RES, RES, Xtr.shape[-1]))
        model = tf.keras.Model(inp, base(ColorJitter()(inp)))
    else:
        model = base
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    pos = ytr.sum(); neg = len(ytr) - pos
    model.fit(Xtr, ytr, epochs=18, batch_size=64, verbose=0,
              class_weight={0: 1.0, 1: neg / max(pos, 1)})
    return model


def fit_dann(Xtr, ytr, cam_tr, seed, lamb=1.0, cam_w=0.5):
    tf.keras.utils.set_random_seed(seed)
    u = sorted(set(cam_tr)); idm = {c: i for i, c in enumerate(u)}; ncam = len(u)
    cam_ids = np.array([idm[c] for c in cam_tr])
    base, _ = build_for_budget((RES, RES, Xtr.shape[-1]), 64, name=f"d{seed}")
    feat = base.get_layer("gap").output
    smoke = L.Dense(1, name="smoke_out")(feat)
    c = GRL(lamb)(feat)
    c = L.Dense(64, activation="relu")(c)
    c = L.Dense(ncam, activation="softmax", name="cam_out")(c)
    model = tf.keras.Model(base.input, [smoke, c])          # outputs in order: [smoke, cam]
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        tf.keras.losses.SparseCategoricalCrossentropy()],
                  loss_weights=[1.0, cam_w])
    pos = ytr.sum(); neg = len(ytr) - pos
    sw = np.where(ytr == 1, neg / max(pos, 1), 1.0).astype("float32")
    model.fit(Xtr, [ytr, cam_ids],
              sample_weight=[sw, np.ones(len(ytr), "float32")],
              epochs=18, batch_size=64, verbose=0)
    return tf.keras.Model(base.input, smoke)     # smoke head only for inference


def fit_stare(Xtr, ytr, seed):
    tf.keras.utils.set_random_seed(seed)
    m, _ = build_for_budget((RES, RES, Xtr.shape[-1]), 64, name=f"s{seed}")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    pos = ytr.sum(); neg = len(ytr) - pos
    m.fit(Xtr, ytr, epochs=18, batch_size=64, verbose=0, class_weight={0: 1.0, 1: neg / max(pos, 1)})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max-seq", type=int, default=150)
    args = ap.parse_args()
    seqs = load(args.max_seq)
    print(f"loaded {len(seqs)} sequences, {len({s['cam'] for s in seqs})} cameras", flush=True)

    Xr, y, cams = make_rep(seqs, "raw")                       # raw pixels, clear frames
    tr, te = split(cams, "camera", seed=1)
    ntr_cam = len(set(cams[tr]))
    print(f"clear frames: {len(y)}  train {len(tr)} / test {len(te)}  "
          f"({ntr_cam} train cams, {len(set(cams[te]))} test cams)", flush=True)

    rows = {}
    def run(name, fn):
        aps = [average_precision_score(y[te], _predict(fn(s), Xr[te])) for s in range(args.seeds)]
        rows[name] = (float(np.mean(aps)), float(np.std(aps)))
        print(f"  {name:<30} PR-AUC {rows[name][0]:.3f} ± {rows[name][1]:.3f}", flush=True)

    print("\n===== invariance baselines on RAW input (cross-camera, camera-holdout) =====", flush=True)
    run("raw (no invariance)",      lambda s: fit_plain(Xr[tr], y[tr], s))
    run("raw + augmentation",       lambda s: fit_plain(Xr[tr], y[tr], s, aug=True))
    run("raw + domain-adversarial", lambda s: fit_dann(Xr[tr], y[tr], cams[tr], s))
    del Xr

    Xs, ys, camss = stare_rep(seqs)                          # K=3 residuals
    trs, tes = split(camss, "camera", seed=1)
    aps = [average_precision_score(ys[tes], _predict(fit_stare(Xs[trs], ys[trs], s), Xs[tes]))
           for s in range(args.seeds)]
    rows["STARE (K=3)"] = (float(np.mean(aps)), float(np.std(aps)))
    print(f"  {'STARE (K=3)':<30} PR-AUC {rows['STARE (K=3)'][0]:.3f} ± {rows['STARE (K=3)'][1]:.3f}", flush=True)

    print("\nSUMMARY (cross-camera PR-AUC):", flush=True)
    for k, (mu, sd) in rows.items():
        print(f"  {k:<30} {mu:.3f} ± {sd:.3f}", flush=True)


if __name__ == "__main__":
    main()
