"""
sim2real.py - use synthetic Isaac Sim images (labels known exactly) to set and check the local rules, and train a
small position classifier on SYNTHETIC crops only; then test both on the REAL hand-labelled camera frames.

  python sim2real.py --synth synth.zip --real-dir <folder with real frames> --labels test_labels.py
"""
import argparse
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image

import vial_core as vc

CLASSES = ["pass", "missing_cap", "crooked_cap", "no_vial"]


def crop_vec(crop, r):
    """Lighting-normalised radial colour profile of the target hole (same idea as the removed v5 feature, but it
    is trained here on SYNTHETIC images only)."""
    a = np.array(crop.convert("RGB").resize((64, 64)))
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(float)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(float)
    k = 64 / crop.width
    R = r * k
    yy, xx = np.mgrid[0:64, 0:64]
    d = np.hypot(xx - 32, yy - 32)
    plate = np.percentile(hsv[..., 2][(d > 1.15 * R) & (d < 1.5 * R)], 75) + 1
    out = []
    for lo, hi in ((0, 0.35), (0.35, 0.7), (0.7, 1.0), (1.0, 1.3)):
        m = (d >= lo * R) & (d < hi * R)
        H, S, V = hsv[..., 0][m], hsv[..., 1][m], hsv[..., 2][m] / plate
        yel = ((H >= 18) & (H <= 40) & (S >= 90) & (hsv[..., 2][m] >= 110)).mean()
        out += [V.mean(), V.std(), np.percentile(V, 90), np.percentile(V, 10), S.mean() / 255, yel]
    e = cv2.Canny(g.astype(np.uint8), 50, 120)
    out += [e[d < 0.7 * R].mean() / 255, e[(d >= 0.7 * R) & (d < 1.1 * R)].mean() / 255]
    return np.array(out)


def positions(img):
    c = vc.all_centers(vc.DEFAULT_CALIB, img.size)
    r = vc.hole_radius(img.size)
    return c, r


def load_synth(path):
    z = zipfile.ZipFile(path)
    labels = json.loads(z.read("labels.json"))
    rows = []
    for name, info in labels.items():
        img = Image.open(io.BytesIO(z.read(name))).convert("RGB")
        c, r = positions(img)
        for p, lab in info["labels"].items():
            crop = vc.crop_square(img, *c[p], 56)
            rows.append(dict(frame=name, pid=p, label=lab, feats=vc.local_features(crop, r), vec=crop_vec(crop, r),
                             placed=info["placed_G"], gap=info.get("notes", {}).get(p) == "missed_placement"))
    return rows, labels


def load_real(real_dir, label_sets):
    rows = []
    for setname, labs in label_sets.items():
        for img_name, lab in labs.items():
            img = Image.open(f"{real_dir}/{img_name}").convert("RGB")
            c, r = positions(img)
            for p, t in lab.items():
                if p[0] != "G":
                    continue
                crop = vc.crop_square(img, *c[p], 56)
                rows.append(dict(set=setname, frame=img_name, pid=p, label=t, feats=vc.local_features(crop, r),
                                 vec=crop_vec(crop, r)))
    return rows


def describe(rows, title):
    print(f"\n== {title}: local measurements by true label ==")
    by = defaultdict(list)
    for x in rows:
        by[x["label"]].append(x["feats"])
    for lab in CLASSES:
        fs = by.get(lab, [])
        if not fs:
            continue
        cov = np.array([f.center_cov for f in fs]); yel = np.array([f.yellow for f in fs])
        p90 = np.array([f.p90 for f in fs]); std = np.array([f.std for f in fs])
        print(f"  {lab:12s} n={len(fs):4d} | cap-over-centre p5/p50/p95 {np.percentile(cov,5):.2f}/{np.median(cov):.2f}/{np.percentile(cov,95):.2f}"
              f" | yellow p5/p50 {np.percentile(yel,5):.2f}/{np.median(yel):.2f} | p90 med {np.median(p90):.0f} | std med {np.median(std):.0f}")


def rule_scores(rows, title):
    """How the current local rules behave on labelled crops (no model)."""
    c = Counter()
    for x in rows:
        f, t = x["feats"], x["label"]
        g = vc.local_guess(f)
        c[(t, g)] += 1
    print(f"\n== {title}: local guess (cap / empty / uncertain) by true label ==")
    for t in CLASSES:
        tot = sum(v for (a, _), v in c.items() if a == t)
        if tot:
            print(f"  {t:12s} " + "  ".join(f"{g}={c[(t, g)]}" for g in ("cap", "empty", "uncertain")) + f"  (n={tot})")
    pass_low = [x for x in rows if x["label"] == "pass" and x["feats"].center_cov < vc.CENTER_COV_MIN]
    open_cap = [x for x in rows if x["label"] == "missing_cap" and x["feats"].center_cov >= vc.CENTER_COV_MIN]
    empty_as_cap = [x for x in rows if x["label"] == "no_vial" and vc.local_guess(x["feats"]) == "cap"]
    print(f"  good caps with cap-over-centre < {vc.CENTER_COV_MIN} (would go to REVIEW): {len(pass_low)}")
    print(f"  missing caps that look covered (cap-over-centre >= {vc.CENTER_COV_MIN}, could pass): {len(open_cap)}")
    print(f"  empty holes that look capped: {len(empty_as_cap)}")


def train_eval(train, test_sets):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.array([x["vec"] for x in train]); y = np.array([x["label"] for x in train])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")).fit(X, y)
    print(f"\n== Classifier trained on {len(train)} SYNTHETIC crops ({dict(Counter(y))}) ==")
    out = {}
    for name, rows in test_sets.items():
        Xt = np.array([x["vec"] for x in rows]); yt = np.array([x["label"] for x in rows])
        pr = m.predict(Xt)
        cm = Counter(zip(yt, pr))
        defects = [i for i, t in enumerate(yt) if t in ("missing_cap", "crooked_cap")]
        caught = sum(1 for i in defects if pr[i] != "pass" and pr[i] != "no_vial")
        acc = float((pr == yt).mean())
        print(f"  {name}: exact {acc:.1%} ({(pr == yt).sum()}/{len(yt)}) | defects flagged {caught}/{len(defects)}")
        for t in CLASSES:
            row = {p: cm[(t, p)] for p in CLASSES if cm[(t, p)]}
            if row:
                print(f"     {t:12s} -> {row}")
        out[name] = dict(acc=acc, n=len(yt), caught=caught, defects=len(defects))
    return m, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(__import__("pathlib").Path(a.labels).parent))
    import test_labels as TL
    syn, _ = load_synth(a.synth)
    print("synthetic positions:", len(syn), dict(Counter(x["label"] for x in syn)))
    describe(syn, "SYNTHETIC")
    rule_scores(syn, "SYNTHETIC")
    real = load_real(a.real_dir, {"dev": TL.TEST, "test": TL.TEST2})
    describe(real, "REAL (dev+test)")
    rule_scores(real, "REAL (dev+test)")
    train_eval(syn, {"real dev": [x for x in real if x["set"] == "dev"], "real test": [x for x in real if x["set"] == "test"]})
