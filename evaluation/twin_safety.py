"""twin_safety.py - run the app's local checks + fill-order rules on synthetic digital-twin frames (synth.zip from
isaac/run_synth.bat) and count, per true label, how each position ended up.   python evaluation/twin_safety.py synth.zip"""
import sys, json, io, zipfile, copy
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
from collections import Counter
from PIL import Image
import vial_core as vc
z = zipfile.ZipFile(sys.argv[1]); lab = json.loads(z.read("labels.json"))
modes = {"robot count given": Counter(), "fill estimated from image": Counter(), "no count": Counter()}
for name, info in lab.items():
    img = Image.open(io.BytesIO(z.read(name))).convert("RGB")
    cen = vc.all_centers(vc.DEFAULT_CALIB, img.size); r = vc.hole_radius(img.size)
    base = {p: vc.decide(p, vc.local_features(vc.crop_square(img, *cen[p], 56), r), None) for p in cen if p[0] == "G"}
    for m, cnt in modes.items():
        res = copy.deepcopy(base)
        if m == "robot count given":
            vc.apply_placement_count(res, {"G": info["placed_G"]}); vc.apply_fill_pattern(res, {"G": info["placed_G"]})
        elif m == "fill estimated from image":
            vc.apply_estimated_count(res, {"G": vc.estimate_placed(res)["G"]}); vc.apply_fill_pattern(res, {})
        else:
            vc.apply_fill_pattern(res, {})
        for p, t in info["labels"].items():
            kind = "missed_placement" if info["notes"].get(p) == "missed_placement" else t
            cnt[(kind, res[p].status)] += 1
for m, cnt in modes.items():
    print(f"\n== {m} ==")
    for t in ("pass", "missing_cap", "crooked_cap", "missed_placement", "no_vial"):
        d = {s: v for (a, s), v in cnt.items() if a == t}
        esc = sum(v for s, v in d.items() if s in ("EMPTY",)) if t in ("missing_cap", "crooked_cap", "missed_placement") else None
        print(f"  {t:16s} {d}" + (f"   <- slipped through as EMPTY: {esc}" if esc is not None else ""))
