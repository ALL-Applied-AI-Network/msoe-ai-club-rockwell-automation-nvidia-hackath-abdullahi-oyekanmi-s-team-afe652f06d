"""autofit_test.py - take real labelled frames, transform them like a different camera would (higher resolution, 16:9,
rotated, zoomed out/in, moved), auto-fit them, and compare every decision with the decision on the original frame.
   python autofit_test.py <folder containing ds/images and test_labels.py>"""
import sys; sys.path.insert(0, "/home/claude/work"); S = sys.argv[1]; sys.path.insert(0, S)
import numpy as np, cv2
from PIL import Image
import vial_core as vc, test_labels as TL
def M_of(scale, rot, cw, ch, cx=320, cy=240, tx=0, ty=0):
    M = cv2.getRotationMatrix2D((cx, cy), rot, scale); M[:, 2] += [cw / 2 - cx + tx, ch / 2 - cy + ty]; return M
frames = list(TL.TEST2) + list(TL.TEST)[::2] + list(TL.TEST4)[-3:] + list(TL.DEV2)[::4]
exec("T = " + open(f"{S}/fit_test.py").read().split("T = ")[1].split("rng = ")[0])
labels = {}; [labels.update(d) for d in (TL.TEST, TL.TEST2, TL.TEST4, TL.DEV2)]
def decide_all(img, cal):
    cen = vc.all_centers(cal, img.size); r = vc.hole_radius(img.size)
    hidden = set(vc.HIDDEN_POSITIONS) | set(vc.blocked_positions(img, cen))
    res = {}
    for p in cen:
        f = vc.local_features(vc.crop_square(img, *cen[p], 56), r)
        res[p] = vc.hidden_result(p, f) if p in hidden else vc.decide(p, f, None)
    vc.apply_estimated_count(res, vc.estimate_placed(res)); vc.apply_fill_pattern(res, {})
    return res
rng = np.random.default_rng(1)
for name, mk in T.items():
    M, cw, ch = mk()
    same = tot = 0; defects_kept = defects = base_kept = 0; worst = []; lost = []
    for fr in frames:
        img = Image.open(f"{S}/ds/images/{fr}").convert("RGB")
        base = decide_all(img, vc.DEFAULT_CALIB)
        bg = np.clip(rng.normal(95, 25, (ch, cw, 3)), 0, 255).astype(np.uint8)
        im = Image.fromarray(cv2.warpAffine(np.array(img), M, (cw, ch), borderMode=cv2.BORDER_TRANSPARENT, dst=bg.copy()))
        canon, cal, info = vc.auto_fit(im)
        fit = decide_all(canon, cal)
        lab = labels[fr]
        visible = [p for p in base if p in lab and (lambda q: 0 <= q[0] < cw and 0 <= q[1] < ch)(M @ np.array([*vc.all_centers(vc.DEFAULT_CALIB, (640, 480))[p], 1.0]))]
        s_ = sum(base[p].status == fit[p].status for p in visible); same += s_; tot += len(visible)
        for p, t in lab.items():
            if t in ("missing_cap", "crooked_cap") and p in visible:
                defects += 1; defects_kept += fit[p].status in ("REJECT", "REVIEW", "GAP"); base_kept += base[p].status in ("REJECT", "REVIEW", "GAP")
                if base[p].status in ("REJECT", "REVIEW", "GAP") and fit[p].status not in ("REJECT", "REVIEW", "GAP"): lost.append((fr[18:24], p, t, fit[p].status))
        if s_ < len(visible): worst.append((fr[18:24], len(visible) - s_))
    print(f"{name:36s} same decision as the original photo: {same}/{tot} ({same / tot:.1%}) | defects flagged: original {base_kept}/{defects}, auto-fit {defects_kept}/{defects}, lost {lost} | frames with differences {worst[:6]}", flush=True)
