"""Download a subset of the HPWREN FIgLib dataset (open HTTP, no login).

FIgLib = fixed-camera SoCal wildfire sequences, ~81 frames each spaced ~60 s, spanning
40 min before -> 40 min after ignition. The per-frame label is encoded in the filename:
    <unixtime>_<offset_seconds>.jpg   offset < 0 => pre-ignition (NO smoke), >= 0 => smoke.

This is the static-camera, slow-cadence dataset the S3 method needs (and it supplies its own
negatives). We grab an evenly-spread subset of whole sequences (so camera-holdout splits are
possible later) into data/raw/figlib/<SEQUENCE>/.

    python data/download_figlib.py --sequences 40 --out data/raw/figlib
"""
import argparse
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data"
UA = {"User-Agent": "Mozilla/5.0 (research; tinyml-wildfire)"}
IMG_RE = re.compile(r"[0-9]{10}_[+-][0-9]+\.jpg")
SEQ_RE = re.compile(r"20[0-9]{6}_[A-Za-z0-9_-]+")


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_sequences():
    html = _get(f"{BASE}/index.html").decode("utf-8", "ignore")
    seqs = sorted(set(SEQ_RE.findall(html)))
    return [s for s in seqs if not s.endswith(".jpg")]


def list_images(seq):
    html = _get(f"{BASE}/{seq}/index.html").decode("utf-8", "ignore")
    return sorted(set(IMG_RE.findall(html)))


def _download_one(seq, fname, out):
    dst = os.path.join(out, seq, fname)                 # saved with literal '+'/'-'
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url_name = fname.replace("+", "%2B")                # server encodes post-ignition '+' as %2B
    for attempt in range(3):
        try:
            data = _get(f"{BASE}/{seq}/{url_name}")
            with open(dst, "wb") as f:
                f.write(data)
            return len(data)
        except Exception:
            if attempt == 2:
                return -1
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", type=int, default=40, help="how many sequences to fetch")
    ap.add_argument("--out", default="data/raw/figlib")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    all_seqs = list_sequences()
    print(f"[figlib] {len(all_seqs)} sequences available")
    # evenly spread across the archive for camera/site/year diversity
    n = min(args.sequences, len(all_seqs))
    step = max(1, len(all_seqs) // n)
    chosen = all_seqs[::step][:n]
    print(f"[figlib] downloading {len(chosen)} sequences -> {args.out}")

    total_bytes = 0
    total_imgs = 0
    for i, seq in enumerate(chosen, 1):
        try:
            imgs = list_images(seq)
        except Exception as e:
            print(f"  [{i}/{len(chosen)}] {seq}: index FAILED ({e})")
            continue
        got = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_download_one, seq, f, args.out): f for f in imgs}
            for fut in as_completed(futs):
                b = fut.result()
                if b > 0:
                    total_bytes += b; got += 1
        total_imgs += got
        print(f"  [{i}/{len(chosen)}] {seq}: {got}/{len(imgs)} imgs "
              f"(cumulative {total_imgs} imgs, {total_bytes/1e6:.0f} MB)")
    print(f"[figlib] done: {total_imgs} images, {total_bytes/1e9:.2f} GB in {args.out}")


if __name__ == "__main__":
    main()
