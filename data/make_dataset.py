"""Dataset builder: produces 96x96, region-tagged data with matched single-frame and
temporal (T-frame) versions, saved as .npz under data/processed/<region>/.

Two paths:
  --synthetic   Generate a small, physically-plausible synthetic set so the WHOLE
                pipeline (train -> int8 -> latency/energy -> cross-region eval) runs
                on a laptop before any real download. Regions differ in background
                palette so AZ->boreal / AZ->SoCal domain shift is real and measurable.
  (real)        Stubs for FLAME / FIgLib / Boreal-UAV (§4 of the spec). These require
                gated downloads (IEEE DataPort etc.); fill in _load_flame() etc. and
                run without --synthetic. Output format is identical, so training code
                never changes.

Layout written:
  data/processed/<region>/{train,val,test}_single.npz    X:(N,96,96,3)uint8  y:(N,)
  data/processed/<region>/{train,val,test}_temporal.npz  X:(N,T,96,96,3)uint8 y:(N,)
  data/processed/manifest.json
"""
import argparse
import json
import os
import re
import numpy as np

SIZE = 96
T = 3
REGIONS = {
    # base RGB background + smoke tint; chosen so a model can overfit AZ and then
    # partially fail on boreal/socal (the robustness stress test, §9).
    "arizona_pine":     dict(bg=(120, 110, 70),  jitter=18, smoke=(200, 200, 200)),
    "socal_chaparral":  dict(bg=(150, 135, 95),  jitter=16, smoke=(205, 200, 190)),
    "boreal":           dict(bg=(60, 90, 60),    jitter=22, smoke=(190, 195, 200)),
}


def _bg(rng, cfg):
    base = np.array(cfg["bg"], "float32")
    img = np.ones((SIZE, SIZE, 3), "float32") * base
    # low-freq lighting + high-freq grain (texture the CNN/handcrafted cues can use)
    low = rng.normal(0, cfg["jitter"], (6, 6, 3)).astype("float32")
    low = np.array([[low[int(i / SIZE * 6), int(j / SIZE * 6)] for j in range(SIZE)]
                    for i in range(SIZE)])
    img += low
    img += rng.normal(0, 6, (SIZE, SIZE, 3))
    return img


def _smoke_layer(rng, cfg, cx, cy, radius, opacity):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = np.exp(-d2 / (2 * radius ** 2))[..., None]        # soft gaussian plume
    smoke = np.array(cfg["smoke"], "float32")
    return mask * opacity, smoke


def _sequence(rng, cfg, positive):
    """Return (T,H,W,3) uint8 sequence. Positive: a plume that GROWS across frames."""
    frames = []
    cx, cy = rng.uniform(20, SIZE - 20), rng.uniform(20, SIZE - 20)
    r0 = rng.uniform(6, 10)
    op0 = rng.uniform(0.35, 0.6)
    for t in range(T):
        img = _bg(rng, cfg)
        if positive:
            radius = r0 * (1.0 + 0.5 * t)                    # slow growth = temporal cue
            opacity = min(0.9, op0 + 0.12 * t)
            drift = 2.0 * t
            m, smoke = _smoke_layer(rng, cfg, cx + drift, cy - drift, radius, opacity)
            img = img * (1 - m) + smoke * m
        else:
            # hard negatives: static haze/cloud patch (no growth) so temporal helps
            if rng.random() < 0.5:
                m, smoke = _smoke_layer(rng, cfg, cx, cy, rng.uniform(10, 16),
                                        rng.uniform(0.15, 0.3))
                img = img * (1 - m) + smoke * m
        frames.append(np.clip(img, 0, 255).astype("uint8"))
    return np.stack(frames)


def gen_region(region, n_per_class, seed):
    rng = np.random.default_rng(seed)
    cfg = REGIONS[region]
    X, y = [], []
    for _ in range(n_per_class):
        X.append(_sequence(rng, cfg, True));  y.append(1)
        X.append(_sequence(rng, cfg, False)); y.append(0)
    X = np.stack(X); y = np.array(y, "int64")
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _save(out, region, split, X_temporal, y):
    d = os.path.join(out, region)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{split}_temporal.npz"), X=X_temporal, y=y)
    np.savez_compressed(os.path.join(d, f"{split}_single.npz"), X=X_temporal[:, -1], y=y)


def build_synthetic(out, n_per_class, seed):
    manifest = {"size": SIZE, "temporal_frames": T, "regions": {}, "synthetic": True}
    for r_i, region in enumerate(REGIONS):
        # in-region region (arizona) gets train/val/test; others get test-only + small val
        splits = {"train": n_per_class, "val": max(30, n_per_class // 5),
                  "test": max(50, n_per_class // 3)}
        manifest["regions"][region] = {}
        for s_i, (split, n) in enumerate(splits.items()):
            X, y = gen_region(region, n, seed + 1000 * r_i + s_i)
            _save(out, region, split, X, y)
            manifest["regions"][region][split] = {"n": int(len(y)),
                                                   "pos": int(y.sum())}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[make_dataset] synthetic set written to {out}")
    print(json.dumps(manifest["regions"], indent=2))


# --- real-dataset ingest (FLAME primary; generic folder loader for Boreal/FIgLib) ---
#
# DOWNLOAD (manual, gated — Claude cannot do this for you):
#   FLAME: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
#     Grab the *classification* images ("Training" and "Test", each with Fire/ and
#     No_Fire/ subfolders of 254x254 JPEGs). Unzip so you have, e.g.:
#         data/raw/flame/Training/Fire/*.jpg
#         data/raw/flame/Training/No_Fire/*.jpg
#         data/raw/flame/Test/Fire/*.jpg
#         data/raw/flame/Test/No_Fire/*.jpg
#   Then:  python data/make_dataset.py --dataset flame --raw data/raw/flame
#
# Boreal-UAV / FIgLib (cross-region): arrange as <root>/<positive>/ and <root>/<negative>/
# image folders, then:
#   python data/make_dataset.py --dataset folder --raw <root> --region boreal
#   python data/make_dataset.py --dataset folder --raw <root> --region socal_chaparral
#
# Temporal note: FLAME classification frames come from video; sorting each class folder
# in natural filename order recovers near-adjacent frames, so T-stacks over consecutive
# sorted frames approximate the native cadence. Use --temporal-mode replicate if a
# source's frames are NOT sequential (temporal then degrades to single-frame — logged).

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
POS_KEYS = ("fire", "smoke", "flame")           # folder-name -> label 1
NEG_HINTS = ("no", "non", "empty", "background", "clear")   # forces label 0


def _natural_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _label_for(folder_name):
    n = folder_name.lower()
    if any(h in n for h in NEG_HINTS):
        return 0
    if any(k in n for k in POS_KEYS):
        return 1
    return None                                  # unrecognized -> skip


def _class_dirs(root):
    """Yield (dir_path, label) for immediate subfolders recognized as pos/neg."""
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            lab = _label_for(name)
            if lab is not None:
                yield p, lab


def _load_resize(path, size):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype="uint8")


def _ingest_split_dir(root, size, temporal_mode, max_per_class):
    """Ingest a directory whose subfolders are pos/neg classes.
    Returns X_temporal (N,T,size,size,3) uint8, y (N,)."""
    Xt, ys = [], []
    for cdir, lab in _class_dirs(root):
        files = [os.path.join(cdir, f) for f in sorted(os.listdir(cdir), key=_natural_key)
                 if f.lower().endswith(IMG_EXTS)]
        if max_per_class:
            files = files[:max_per_class]
        frames = [_load_resize(f, size) for f in files]
        if len(frames) == 0:
            continue
        arr = np.stack(frames)                              # (M,size,size,3)
        if temporal_mode == "order" and len(arr) >= T:
            # non-overlapping consecutive T-stacks preserve temporal adjacency
            m = (len(arr) // T) * T
            stacks = arr[:m].reshape(-1, T, size, size, 3)
        else:                                               # replicate single frame
            stacks = np.repeat(arr[:, None], T, axis=1)
        Xt.append(stacks)
        ys.append(np.full(len(stacks), lab, "int64"))
        print(f"  [{os.path.basename(cdir)}] label={lab}: {len(files)} imgs "
              f"-> {len(stacks)} samples")
    if not Xt:
        raise SystemExit(f"No recognized class folders under {root}. Expected subdirs "
                         f"whose names imply fire/smoke (pos) or no_fire/background (neg).")
    X = np.concatenate(Xt); y = np.concatenate(ys)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _split_train_val(X, y, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_frac)
    return (X[idx[n_val:]], y[idx[n_val:]]), (X[idx[:n_val]], y[idx[:n_val]])


def build_flame(raw, out, size, temporal_mode, max_per_class, val_frac=0.15):
    """FLAME classification: raw/Training/{Fire,No_Fire} + raw/Test/{Fire,No_Fire}.
    Region tag = arizona_pine. Training -> train/val, Test -> test."""
    region = "arizona_pine"
    train_root = _find_subdir(raw, ("Training", "train"))
    test_root = _find_subdir(raw, ("Test", "test"))
    if train_root is None:
        # single-level layout: raw/{Fire,No_Fire} -> split into train/val/test
        print("[flame] no Training/Test dirs; treating raw as one pool")
        X, y = _ingest_split_dir(raw, size, temporal_mode, max_per_class)
        (Xtr, ytr), (Xte, yte) = _split_train_val(X, y, 0.2, 1)
        (Xtr, ytr), (Xva, yva) = _split_train_val(Xtr, ytr, val_frac, 2)
    else:
        print(f"[flame] Training <- {train_root}")
        Xall, yall = _ingest_split_dir(train_root, size, temporal_mode, max_per_class)
        (Xtr, ytr), (Xva, yva) = _split_train_val(Xall, yall, val_frac, 2)
        print(f"[flame] Test <- {test_root}")
        Xte, yte = _ingest_split_dir(test_root, size, temporal_mode, max_per_class)
    for split, (Xs, yv) in {"train": (Xtr, ytr), "val": (Xva, yva),
                            "test": (Xte, yte)}.items():
        _save(out, region, split, Xs, yv)
    _write_manifest(out, "flame", size, {region: {"train": len(ytr), "val": len(yva),
                                                   "test": len(yte)}}, temporal_mode)
    print(f"[flame] wrote region '{region}' to {out}  "
          f"(train {len(ytr)} / val {len(yva)} / test {len(yte)})")


def build_folder(raw, out, region, size, temporal_mode, max_per_class):
    """Generic single-region ingest (Boreal-UAV, FIgLib, extra negatives). Splits the
    pooled images into train/val/test and tags them with `region`."""
    X, y = _ingest_split_dir(raw, size, temporal_mode, max_per_class)
    (Xtr, ytr), (Xte, yte) = _split_train_val(X, y, 0.3, 1)     # 70/30 then val off train
    (Xtr, ytr), (Xva, yva) = _split_train_val(Xtr, ytr, 0.15, 2)
    for split, (Xs, yv) in {"train": (Xtr, ytr), "val": (Xva, yva),
                            "test": (Xte, yte)}.items():
        _save(out, region, split, Xs, yv)
    _write_manifest(out, f"folder:{region}", size,
                    {region: {"train": len(ytr), "val": len(yva), "test": len(yte)}},
                    temporal_mode, merge=True)
    print(f"[folder] wrote region '{region}' (train {len(ytr)} / val {len(yva)} "
          f"/ test {len(yte)})")


FIGLIB_IMG_RE = re.compile(r"(\d{10})_([+-]\d+)\.jpg$")


def _figlib_camera(seq):
    """Camera id = everything after 'YYYYMMDD_FIRE_' (e.g. 20160604_FIRE_rm-n-mobo-c -> rm-n-mobo-c)."""
    parts = seq.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else seq


def _figlib_frames(seq_dir):
    """Return frames sorted by timestamp: list of (unixtime, offset_seconds, path)."""
    items = []
    for f in os.listdir(seq_dir):
        m = FIGLIB_IMG_RE.search(f)
        if m:
            items.append((int(m.group(1)), int(m.group(2)), os.path.join(seq_dir, f)))
    items.sort()
    return items


def build_figlib(raw, out, size, val_frac=0.15, test_frac=0.2):
    """FIgLib (SoCal, static camera). Per-frame label from filename offset (>=0 => smoke).
    Temporal stacks from consecutive ~60 s frames within a sequence; label = last frame.
    Split by CAMERA (not sequence) to avoid camera leakage (spec §9). region=socal_chaparral."""
    import re as _re  # (module already imported at top)
    region = "socal_chaparral"
    seqs = sorted(d for d in os.listdir(raw) if os.path.isdir(os.path.join(raw, d)))
    cameras = sorted({_figlib_camera(s) for s in seqs})
    rng = np.random.default_rng(0)
    order = rng.permutation(len(cameras))
    n_test = max(1, int(len(cameras) * test_frac))
    n_val = max(1, int(len(cameras) * val_frac))
    test_cams = {cameras[i] for i in order[:n_test]}
    val_cams = {cameras[i] for i in order[n_test:n_test + n_val]}

    def split_of(seq):
        cam = _figlib_camera(seq)
        return "test" if cam in test_cams else "val" if cam in val_cams else "train"

    bins = {"train": [], "val": [], "test": []}
    for seq in seqs:
        frames = _figlib_frames(os.path.join(raw, seq))
        if len(frames) < T:
            continue
        imgs = np.stack([_load_resize(p, size) for _, _, p in frames])
        offs = np.array([o for _, o, _ in frames])
        m = (len(imgs) // T) * T
        stacks = imgs[:m].reshape(-1, T, size, size, 3)
        y = (offs[:m].reshape(-1, T)[:, -1] >= 0).astype("int64")   # label = current frame
        bins[split_of(seq)].append((stacks, y))

    counts = {}
    for split, items in bins.items():
        if not items:
            continue
        X = np.concatenate([s for s, _ in items])
        Y = np.concatenate([y for _, y in items])
        idx = np.random.default_rng(1).permutation(len(Y))
        _save(out, region, split, X[idx], Y[idx])
        counts[split] = {"n": int(len(Y)), "pos": int(Y.sum())}
    _write_manifest(out, "figlib", size, {region: counts}, "order", merge=True)
    print(f"[figlib] cameras: {len(cameras)} ({len(test_cams)} test / {len(val_cams)} val) "
          f"| region '{region}' -> {counts}")


def _find_subdir(root, names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.isdir(p):
            return p
    return None


def _write_manifest(out, dataset, size, regions, temporal_mode, merge=False):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "manifest.json")
    man = {"size": size, "temporal_frames": T, "synthetic": False,
           "dataset": dataset, "temporal_mode": temporal_mode, "regions": {}}
    if merge and os.path.exists(path):
        with open(path) as f:
            man = json.load(f)
        man.setdefault("regions", {})
    man["regions"].update({r: v for r, v in regions.items()})
    with open(path, "w") as f:
        json.dump(man, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "flame", "folder", "figlib"],
                    default="synthetic")
    ap.add_argument("--synthetic", action="store_true",
                    help="alias for --dataset synthetic")
    ap.add_argument("--raw", help="raw dataset root (for flame/folder)")
    ap.add_argument("--region", default="boreal",
                    help="region tag for --dataset folder (e.g. boreal, socal_chaparral)")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=SIZE, help="native store resolution")
    ap.add_argument("--temporal-mode", choices=["order", "replicate"], default="order")
    ap.add_argument("--max-per-class", type=int, default=3000,
                    help="cap images per class per split (0 = all; guards laptop RAM)")
    ap.add_argument("--n-per-class", type=int, default=250, help="synthetic only")
    ap.add_argument("--seed", type=int, default=0, help="synthetic only")
    args = ap.parse_args()

    ds = "synthetic" if args.synthetic else args.dataset
    cap = args.max_per_class or None
    if ds == "synthetic":
        build_synthetic(args.out, args.n_per_class, args.seed)
    elif ds == "flame":
        if not args.raw:
            raise SystemExit("--raw is required for --dataset flame (see header of this file)")
        build_flame(args.raw, args.out, args.size, args.temporal_mode, cap)
    elif ds == "folder":
        if not args.raw:
            raise SystemExit("--raw is required for --dataset folder")
        build_folder(args.raw, args.out, args.region, args.size,
                     args.temporal_mode, cap)
    elif ds == "figlib":
        if not args.raw:
            raise SystemExit("--raw is required for --dataset figlib (default data/raw/figlib)")
        build_figlib(args.raw, args.out, args.size)


if __name__ == "__main__":
    main()
