"""
learn_loop.py - "learn from its mistakes" on the digital twin.

Each round:
  1. train a per-position model on ALL synthetic (Isaac Sim) positions so far - mistakes from the last round count extra
  2. test it on a FIXED held-out synthetic test set (exact labels) -> score + list of mistakes
  3. group the mistakes by scenario (tilt angle, lean angle, light, camera shift, fill) and write the next
     Isaac Sim batch settings (focus_next.txt) that render MORE of what it got wrong
  4. look (never train) at the real hand-labelled frames, to see whether it carries over to the real camera

Lab rule: real line images are only looked at, never trained on. Everything trained here is synthetic.

  python learn_loop.py --pools synth_a.zip synth_b.zip synth_c.zip --test-from synth_c.zip --real-dir <frames> --labels test_labels.py
"""
import argparse
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vial_core as vc  # noqa: E402

CLASSES = ["good", "missing_cap", "crooked_cap", "no_vial"]
DEFECTS = ("missing_cap", "crooked_cap")


# ------------------------------------------------------------------ features
def crop_features(crop: Image.Image, r: float) -> np.ndarray:
    a = np.array(crop.convert("RGB").resize((64, 64)))
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(float)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    R = r * 64 / crop.width
    yy, xx = np.mgrid[0:64, 0:64]
    d = np.hypot(xx - 32, yy - 32)
    grey = hsv[..., 1] < 60
    plate = np.percentile(hsv[..., 2][grey], 75) + 1 if grey.sum() > 50 else 255.0
    vf = min(vc.YELLOW_V_MIN, max(vc.YELLOW_V_FLOOR_DIM, 1.1 * plate))
    yel = (hsv[..., 0] >= 18) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 0.55 * vf)
    out = []
    for lo, hi in ((0, 0.35), (0.35, 0.7), (0.7, 1.0), (1.0, 1.4), (1.4, 2.0)):
        m = (d >= lo * R) & (d < hi * R)
        V = hsv[..., 2][m] / plate
        out += [V.mean(), V.std(), np.percentile(V, 90), np.percentile(V, 10), hsv[..., 1][m].mean() / 255, yel[m].mean()]
    # where the yellow sits relative to the hole (direction-free): radial profile + spread
    ys, xs = np.nonzero(yel)
    if len(xs) > 10:
        dist = np.hypot(xs - 32, ys - 32) / R
        out += [np.percentile(dist, 10), np.percentile(dist, 50), np.percentile(dist, 90), len(xs) / (math.pi * R * R)]
    else:
        out += [9, 9, 9, 0]
    e = cv2.Canny(g, 50, 120)
    out += [e[d < 0.7 * R].mean() / 255, e[(d >= 0.7 * R) & (d < 1.1 * R)].mean() / 255]
    f = vc.local_features(crop, r)
    out += [f.yellow, f.center_cov, f.p90 / plate, f.std / plate, min(f.cap_area, 3), min(f.cap_off, 9)]
    return np.array(out, dtype=np.float32)


# ------------------------------------------------------------------ augmentation (camera + sensor realism)
def augment(img: Image.Image, rng) -> Image.Image:
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-0.6, 0.6), rng.uniform(0.98, 1.02))
    M[:, 2] += rng.uniform(-5, 5, 2)                         # camera knocked a few pixels
    a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    a = a * rng.uniform(0.75, 1.3) * np.array([rng.uniform(1.0, 1.15), 1.0, rng.uniform(0.8, 1.0)])   # warmer light
    a = 255 * (np.clip(a, 0, 255) / 255) ** rng.uniform(0.8, 1.25)
    a = np.clip(a + rng.normal(0, rng.uniform(1, 7), a.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 1.0))
    ok, b = cv2.imencode(".jpg", a[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 95))])
    return Image.fromarray(cv2.imdecode(b, 1)[..., ::-1])


# ------------------------------------------------------------------ data
def kind_of(label, note):
    if label == "pass":
        return "lean" if note.startswith("lean") else "good"
    if note == "missed_placement":
        return "missed_placement"
    return label


def target(label):
    return "good" if label == "pass" else label


def scenario(info, p, label, note):
    s = {"kind": kind_of(label, note), "light": min(info["lights"]), "fill": info["placed_G"]}
    m = re.findall(r"[\d.]+", note)
    if note.startswith("tilt"):
        s["tilt"] = float(m[0])
    if note.startswith("lean"):
        s["lean"] = float(m[0])
    cam = info.get("camera_mm_deg")
    s["cam_deg"] = cam[3] if cam else 0.0
    return s


def load_pool(path, frames=None, n_aug=0, seed=0):
    z = zipfile.ZipFile(path)
    labels = json.loads(z.read("labels.json"))
    rng = np.random.default_rng(seed)
    rows = []
    for name, info in labels.items():
        if frames is not None and name not in frames:
            continue
        img0 = Image.open(io.BytesIO(z.read(name))).convert("RGB")
        for k in range(1 + n_aug):
            img = img0 if k == 0 else augment(img0, rng)
            cen = vc.all_centers(vc.DEFAULT_CALIB, img.size)
            r = vc.hole_radius(img.size)
            for p, lab in info["labels"].items():
                note = info["notes"].get(p, "")
                rows.append(dict(src=f"{Path(path).name}:{name}", pid=p, y=target(lab), aug=k,
                                 x=crop_features(vc.crop_square(img, *cen[p], 56), r), sc=scenario(info, p, lab, note)))
    return rows


def load_real(real_dir, label_sets):
    rows = []
    for setname, labs in label_sets.items():
        for name, lab in labs.items():
            img = Image.open(Path(real_dir) / name).convert("RGB")
            cen = vc.all_centers(vc.DEFAULT_CALIB, img.size)
            r = vc.hole_radius(img.size)
            for p, t in lab.items():
                if p[0] != "G":
                    continue
                rows.append(dict(set=setname, frame=name, pid=p, y=target(t),
                                 x=crop_features(vc.crop_square(img, *cen[p], 56), r)))
    return rows


# ------------------------------------------------------------------ model
def train(rows, weights=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.array([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    cw = {c: len(y) / (len(CLASSES) * max(1, (y == c).sum())) for c in CLASSES}
    w = np.array([cw[t] for t in y]) * (weights if weights is not None else 1.0)
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                       random_state=0)
    m.fit(X, y, sample_weight=w)
    return m


def score(m, rows):
    pr = m.predict(np.array([r["x"] for r in rows]))
    out = Counter()
    for r, p in zip(rows, pr):
        k = r.get("sc", {}).get("kind", r["y"])
        out[(k, "ok" if p == r["y"] or (r["y"] in DEFECTS and p in DEFECTS) else "wrong")] += 1
        if r["y"] in DEFECTS:
            out["defects"] += 1
            out["defects_caught"] += p in DEFECTS
        if r["y"] == "good":
            out["good"] += 1
            out["good_flagged"] += p != "good"
        if r["y"] == "no_vial":
            out["empty"] += 1
            out["empty_ok"] += p == "no_vial"
    return out, pr


def mistakes_profile(rows, pr):
    """Which scenarios does the model get wrong? -> settings for the next Isaac Sim batch."""
    bad = defaultdict(list)
    for r, p in zip(rows, pr):
        wrong = not (p == r["y"] or (r["y"] in DEFECTS and p in DEFECTS))
        sc = r["sc"]
        bad[sc["kind"]].append((wrong, sc))
    prof = {}
    for k, v in bad.items():
        prof[k] = dict(n=len(v), wrong=sum(w for w, _ in v))
        for key in ("tilt", "lean"):
            vals = [(s[key], w) for w, s in v if key in s]
            if vals:
                edges = (0, 8, 12, 16, 20, 25, 30, 41) if key == "tilt" else (0, 5, 8, 11, 16)
                prof[k][key] = {f"{lo}-{hi}": f"{sum(w for x, w in vals if lo <= x < hi)}/{sum(1 for x, _ in vals if lo <= x < hi)}"
                                for lo, hi in zip(edges, edges[1:]) if any(lo <= x < hi for x, _ in vals)}
        dim = [w for w, s in v if s["light"] < 0.8]
        prof[k]["dim_light"] = f"{sum(dim)}/{len(dim)}"
    return prof


def next_focus(prof):
    """Turn the mistakes into the next render settings (more of the hard cases)."""
    args = {}
    c = prof.get("crooked_cap", {})
    if c.get("tilt"):
        worst = max(c["tilt"].items(), key=lambda kv: int(kv[1].split("/")[0]) / max(1, int(kv[1].split("/")[1])))
        lo, hi = map(int, worst[0].split("-"))
        args["--tilt-min"], args["--tilt-max"] = max(5, lo - 2), hi + 2
    l = prof.get("lean", {})
    if l.get("n") and l["wrong"] / l["n"] > 0.05:
        args["--p-lean"] = 0.45
    c_rate = c.get("wrong", 0) / max(1, c.get("n", 1))
    args["--p-crooked"] = 0.25 if c_rate > 0.05 else 0.12
    return args


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", required=True, help="round spec: 'name=pool1.zip,pool2.zip' (cumulative)")
    ap.add_argument("--test-from", required=True)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--aug", type=int, default=2)
    ap.add_argument("--real-dir")
    ap.add_argument("--labels")
    ap.add_argument("--out", default="learning_curve.json")
    a = ap.parse_args()

    zt = zipfile.ZipFile(a.test_from)
    tl = sorted(json.loads(zt.read("labels.json")))
    test_frames = set(tl[::int(round(1 / a.test_frac))])
    test = load_pool(a.test_from, frames=test_frames)
    print(f"fixed synthetic test set: {len(test_frames)} frames, {len(test)} positions (never trained on)")
    real = []
    if a.real_dir and a.labels:
        sys.path.insert(0, str(Path(a.labels).parent))
        import test_labels as TL
        real = load_real(a.real_dir, {"dev": TL.TEST, "test": TL.TEST2, "extra": TL.TEST3})
    cache, curve, hard = {}, [], {}
    for spec in a.rounds:
        name, pools = spec.split("=")
        train_rows = []
        for pz in pools.split(","):
            if pz not in cache:
                frames = None
                if Path(pz).name == Path(a.test_from).name:
                    zz = zipfile.ZipFile(pz)
                    frames = set(json.loads(zz.read("labels.json"))) - test_frames
                cache[pz] = load_pool(pz, frames=frames, n_aug=a.aug, seed=len(cache))
            train_rows += cache[pz]
        # learn from mistakes: positions the previous model got wrong count 3x
        w = np.array([3.0 if (r["src"], r["pid"]) in hard else 1.0 for r in train_rows])
        m = train(train_rows, w)
        sc, pr = score(m, test)
        # mine the training set for the next round's hard examples
        trp = m.predict(np.array([r["x"] for r in train_rows]))
        hard = {(r["src"], r["pid"]) for r, p in zip(train_rows, trp) if p != r["y"]}
        prof = mistakes_profile(test, pr)
        res = dict(round=name, train_positions=len(train_rows), hard_examples=len(hard),
                   twin_defects=f'{sc["defects_caught"]}/{sc["defects"]}', twin_good_flagged=f'{sc["good_flagged"]}/{sc["good"]}',
                   twin_lean_wrong=f'{sc[("lean", "wrong")]}/{sc[("lean", "ok")] + sc[("lean", "wrong")]}',
                   twin_empty_ok=f'{sc["empty_ok"]}/{sc["empty"]}', mistakes=prof, next_render=next_focus(prof))
        if real:
            for setname in ("dev", "test", "extra"):
                rr = [x for x in real if x["set"] == setname]
                rs, rp = score(m, rr)
                res[f"real_{setname}"] = dict(defects=f'{rs["defects_caught"]}/{rs["defects"]}',
                                             good_flagged=f'{rs["good_flagged"]}/{rs["good"]}', empty_ok=f'{rs["empty_ok"]}/{rs["empty"]}',
                                             wrong=[(x["frame"][18:24], x["pid"], x["y"], p) for x, p in zip(rr, rp)
                                                    if not (p == x["y"] or (x["y"] in DEFECTS and p in DEFECTS))][:12])
        curve.append(res)
        print(json.dumps({k: v for k, v in res.items() if k not in ("mistakes",)}, indent=1))
    Path(a.out).write_text(json.dumps(curve, indent=1))
    focus = curve[-1]["next_render"]
    Path("focus_next.txt").write_text(" ".join(f"{k} {v}" for k, v in focus.items()))
    print("next Isaac Sim batch:", focus)
