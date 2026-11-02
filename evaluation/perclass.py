"""Per-class results on the HELD-OUT real sets only (never used for tuning). Scores only - changes nothing.
Usage: python evaluation/perclass.py <folder with test_labels.py and ds/images> <vlm_cache.json>
(test_labels.py holds the same labels as the *_labels.csv files here; ds/images = Dataset 1 frames)."""
import os
import sys, json, hashlib; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); S = sys.argv[1]; sys.path.insert(0, S)
from PIL import Image; from collections import Counter, defaultdict
import vial_core as vc, test_labels as TL
cache = json.load(open(sys.argv[2]))
PH = hashlib.md5(vc.PROMPT.encode()).hexdigest()[:8]; VH = hashlib.md5(vc.VERIFY_CAP_PROMPT.encode()).hexdigest()[:8]
def get(nm, p, tag):
    return cache.get(f"{nm}|{p}|{vc.DEFAULT_MODEL}|{tag}|56.0") or cache.get(f"{nm}|{p}|{vc.DEFAULT_MODEL}|{tag}|56")
def run(nm, l, use_model, use_count):
    img = Image.open(f"{S}/ds/images/{nm}").convert("RGB"); cen = vc.all_centers(vc.DEFAULT_CALIB, img.size); r = vc.hole_radius(img.size)
    hidden = set(vc.HIDDEN_POSITIONS) | set(vc.blocked_positions(img, cen)); res = {}; answers = 0
    for p in cen:
        f = vc.local_features(vc.crop_square(img, *cen[p], 56), r)
        if p in hidden: res[p] = vc.hidden_result(p, f); continue
        v = get(nm, p, PH) if use_model else None; sec = get(nm, p, "verify-" + VH) if use_model else None
        answers += bool(v)
        res[p] = vc.decide(p, f, v, second=sec) if (v and vc.local_guess(f) != "empty") else vc.decide(p, f, None)
    filled = [int(p[1:]) for p, t in l.items() if p[0] == "G" and t != "no_vial"]
    if use_count:
        pl = {"G": max(filled) if filled else 0, "S": 0}; vc.apply_placement_count(res, pl); vc.apply_fill_pattern(res, pl)
    else:
        vc.apply_estimated_count(res, vc.estimate_placed(res)); vc.apply_fill_pattern(res, {})
    return res, answers
FLAG = ("REJECT", "REVIEW", "GAP")
out = {}
for setname, labs, use_model in (("A", TL.TEST2, True), ("A", TL.TEST2, False), ("B", TL.TEST3, False), ("C", TL.TEST4, False)):
    for use_count in (False, True):
        k = f"{setname}|{'model' if use_model else 'local'}|{'count' if use_count else 'estimate'}"
        cls = defaultdict(Counter); misses = []; ans = 0
        for nm, l in labs.items():
            res, a = run(nm, l, use_model, use_count); ans += a
            for p, t in l.items():
                s = res[p].status
                if s == "NO VIEW": cls["no_view"]["n"] += 1; continue
                if (nm, p) in TL.EXPECT_GAP:
                    cls["missed_placement"]["n"] += 1; cls["missed_placement"]["caught"] += s in FLAG
                    if s not in FLAG: misses.append([nm[9:24], p, "missed_placement", s])
                    continue
                c = cls[t]; c["n"] += 1
                if t in ("missing_cap", "crooked_cap", "damaged_vial"):
                    c["caught"] += s in FLAG; c["rejected"] += s == "REJECT"
                    if s not in FLAG: misses.append([nm[9:24], p, t, s])
                elif t == "pass":
                    c["flagged"] += s not in ("PASS", "CAP"); c["rejected"] += s == "REJECT"
                else:
                    c["flagged"] += s not in ("EMPTY",); c["rejected"] += s == "REJECT"
        out[k] = {"frames": len(labs), "model_answers": ans, "classes": {c: dict(v) for c, v in cls.items()}, "misses": misses}
        print(k, out[k]["frames"], "answers", ans, {c: dict(v) for c, v in cls.items()}, misses)
json.dump(out, open(f"{S}/perclass.json", "w"), indent=1)
