"""
vial_core.py - everything except the Streamlit UI (so it can be tested without a browser).

Pipeline per fixture position:
  1. Geometry : 4 calibrated corner holes per fixture -> 30 centers (6 cols x 5 rows)
  2. Crop     : square crop around each center (context = neighbouring holes)
  3. Local CV : yellow-cap fraction + dark-hole fraction (cheap, deterministic)
  4. VLM      : NVIDIA NIM vision model classifies the crop into the 5 official labels
  5. Decision : combine VLM + local CV + fill-pattern rule -> PASS / REJECT / REVIEW / EMPTY / GAP
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
BASE_W, BASE_H = 640, 480          # calibration was done on a 640x480 frame
HOLE_RADIUS_BASE = 10.5            # approx radius (px) of a large fixture hole at 640x480

LABELS = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"]
DEFECT_LABELS = ["missing_cap", "crooked_cap", "damaged_vial"]

# Corner holes (center of the TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT large hole
# of the 6x5 grid) in 640x480 pixel space. G = good fixture (left), S = scrap fixture (right).
# The sixth (rightmost) scrap column is partly cut off by the image edge; its corner
# points are extrapolated from the column spacing.
DEFAULT_CALIB = {
    "G": {"TL": (228, 166), "TR": (378, 163), "BR": (382, 305), "BL": (232, 304)},
    "S": {"TL": (421, 259), "TR": (556, 252), "BR": (556, 380), "BL": (424, 392)},
}

# Positions the camera cannot see properly: the rightmost scrap column is half hidden at the plate edge.
HIDDEN_POSITIONS = ["S01", "S02", "S03", "S04", "S05"]

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"

PROMPT = (
    "This is a cropped overhead image from an industrial vial-inspection camera looking down at a "
    "metal fixture plate. The large holes hold glass vials; the small holes in between are just "
    "part of the plate and must be ignored. The target position is the large hole at the exact "
    "CENTER of this image, marked by a thin cyan ring. Neighbouring positions are context only: a yellow "
    "cap that belongs to a NEIGHBOURING position and overlaps the edge of the ring does NOT count as "
    "the target's cap.\n\n"
    "IMPORTANT about camera perspective: the vials are tall and the camera looks down at an angle, so a "
    "correctly seated cap often appears shifted toward one side of its hole, with a dark crescent of the hole "
    "visible beside it. That shift is NORMAL and is NOT a crooked cap. Only call a cap crooked if it is clearly "
    "tilted (you can see its side wall, or it looks like an oval rather than a circle) or it is resting off the "
    "vial so the open mouth is exposed.\n\n"
    "First answer two checks about the target position only:\n"
    "- opening_visible: true if you can see the inside or open mouth of a vial at the target because "
    "no cap sits on it.\n"
    "- cap_tilted: true if a cap on the target looks tilted, offset to one side, or not seated flat.\n\n"
    "Then classify the target as exactly one of:\n"
    "- pass: a vial is present with a cap that covers its opening and sits level.\n"
    "- missing_cap: a vial is present but its mouth is open, no cap (a transparent vial body or "
    "glass rim is NOT an empty hole).\n"
    "- crooked_cap: the cap is clearly tilted or sitting off the vial (not just shifted by perspective).\n"
    "- damaged_vial: the vial body is visibly cracked or broken.\n"
    "- no_vial: the hole is empty (just a dark hole).\n"
    "When unsure between pass and a defect, choose the defect.\n"
    "Respond with ONLY a JSON object, no other text: "
    '{"opening_visible": <true|false>, "cap_tilted": <true|false>, "label": "<one of the five labels>", '
    '"confidence": <0 to 1>, "reason": "<one short sentence>"}'
)


# Second look, asked only when the model said "missing cap" although a yellow cap physically covers the
# vial opening (a contradiction the model fell into repeatedly on the development frames).
VERIFY_CAP_PROMPT = (
    "This is a cropped overhead image from an industrial vial-inspection camera. The target is the vial at "
    "the exact CENTER of the image, marked by a thin cyan ring. A yellow cap IS sitting on top of this vial. "
    "Because the camera looks down at an angle, a correctly seated cap often looks shifted to one side of the "
    "hole - that is normal. Decide only whether this cap is seated properly or crooked (clearly tilted, you "
    "can see its side wall, or it rests off the vial so part of the open mouth is exposed).\n"
    'Respond with ONLY JSON: {"label": "pass" or "crooked_cap", "confidence": <0 to 1>, '
    '"reason": "<one short sentence>"}'
)


# Second look #2, asked only when the model said "pass" but no yellow cap covers the hole centre. On the
# development frames the model said "pass" for bright EMPTY holes - and also for real open vials.
VERIFY_VIAL_PROMPT = (
    "This is a cropped overhead image from an industrial vial-inspection camera looking down at a metal "
    "fixture plate. The target is the large hole at the exact CENTER of the image, marked by a thin cyan ring. "
    "There is NO cap on the target. Ignore caps and vials in neighbouring holes. Decide only: is there a glass "
    "vial standing in the target hole (you can see a glass rim, a glass wall, or the open mouth of a vial), or "
    "is the hole EMPTY (you see only the inside of the round hole and the floor beneath it)?\n"
    'Respond with ONLY JSON: {"label": "missing_cap" if a glass vial is in the hole, or "no_vial" if the hole '
    'is empty, "confidence": <0 to 1>, "reason": "<one short sentence>"}'
)


def needs_second_look(feats: "LocalFeatures", vlm: Optional[dict]) -> bool:
    # Official: missing_cap = "a vial is present, but its MOUTH IS OPEN". A yellow cap over the opening contradicts
    # that, so the model is asked again whenever a cap covers at least CENTER_COV_MIN of the hole centre (was 90%:
    # real G19 - a leaning vial with 86% coverage - was rejected as "missing cap" because of that gap).
    return bool(vlm) and vlm.get("label") == "missing_cap" and feats.center_cov >= 0.45 and feats.yellow >= 0.20


def needs_vial_check(feats: "LocalFeatures", vlm: Optional[dict]) -> bool:
    return bool(vlm) and vlm.get("label") == "pass" and feats.center_cov < 0.05


def second_look_prompt(feats: "LocalFeatures", vlm: Optional[dict]) -> Optional[str]:
    if needs_second_look(feats, vlm):
        return VERIFY_CAP_PROMPT
    if USE_VIAL_CHECK and needs_vial_check(feats, vlm):
        return VERIFY_VIAL_PROMPT
    return None


# Tried on the DEV frames (01:45, Nov 1): the vial-or-empty question said "glass vial present" for 3/3 real
# open vials but ALSO for 7/8 empty holes, so it cannot separate them. It is switched off: those positions
# stay REVIEW (safe), and the limitation is reported. Kept here so the experiment is reproducible.
USE_VIAL_CHECK = False
TRUST_EMPTY_CHECK = False


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def position_id(prefix: str, col: int, row: int) -> str:
    """col 0 = leftmost ... col 5 = rightmost, row 0 = top.
    Robot fill order: rightmost column top->bottom first (01-05), then one column left."""
    n = (5 - col) * 5 + row + 1
    return f"{prefix}{n:02d}"


def fixture_centers(corners: Dict[str, Tuple[float, float]], prefix: str,
                    img_size: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
    """Bilinear interpolation between the 4 corner holes -> 30 centers in ORIGINAL image pixels."""
    w, h = img_size
    sx, sy = w / BASE_W, h / BASE_H
    TL, TR, BR, BL = (np.array(corners[k], dtype=float) for k in ("TL", "TR", "BR", "BL"))
    out = {}
    for c in range(6):
        for r in range(5):
            u, v = c / 5.0, r / 4.0
            top = TL * (1 - u) + TR * u
            bot = BL * (1 - u) + BR * u
            p = top * (1 - v) + bot * v
            out[position_id(prefix, c, r)] = (float(p[0] * sx), float(p[1] * sy))
    return out


def all_centers(calib: dict, img_size: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
    centers = {}
    for prefix in ("G", "S"):
        centers.update(fixture_centers(calib[prefix], prefix, img_size))
    return dict(sorted(centers.items()))


def detect_holes(img: Image.Image) -> np.ndarray:
    """Round holes/caps of about the fixture-hole size, as an (N, 2) array of centres in image pixels."""
    k = img.width / BASE_W
    g = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 1.2)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=14 * k, param1=90, param2=14,
                         minRadius=int(8 * k), maxRadius=int(14 * k) + 1)
    return np.zeros((0, 2)) if c is None else c[0][:, :2].astype(float)


def _grid_from_corners(corners, img_size):
    return np.array(list(fixture_centers(corners, "X", img_size).values()))


def _grid_and_border(corners, img_size):
    """30 grid centres plus the ring of 'phantom' positions one pitch OUTSIDE the 6x5 grid (no holes there)."""
    w, h = img_size
    sx, sy = w / BASE_W, h / BASE_H
    TL, TR, BR, BL = (np.array(corners[k], dtype=float) for k in ("TL", "TR", "BR", "BL"))
    def at(c, r):
        u, v = c / 5.0, r / 4.0
        top = TL * (1 - u) + TR * u
        bot = BL * (1 - u) + BR * u
        p = top * (1 - v) + bot * v
        return (p[0] * sx, p[1] * sy)
    grid = np.array([at(c, r) for c in range(6) for r in range(5)])
    border = np.array([at(c, r) for c in range(-1, 7) for r in range(-1, 6) if c in (-1, 6) or r in (-1, 5)])
    return grid, border


def auto_align(img: Image.Image, calib: dict, search_px: int = 45) -> Tuple[dict, dict]:
    """Re-fit each fixture's 6x5 grid to the round holes actually visible in the image.
    For small rotations/scales around the current calibration, detected holes vote for the best shift.
    A candidate scores +1 for each grid position that lands on a hole and -1 for each hole found one pitch
    OUTSIDE the grid (the plate edge is there), which stops the grid sliding by a whole hole spacing.
    Only translation/rotation/scale are changed. Returns (new_calib in 640x480 space, matched holes per fixture)."""
    k = img.width / BASE_W
    pts = detect_holes(img)
    new, quality = {}, {}
    tol = 4.5 * k
    for fx in ("G", "S"):
        base, border = _grid_and_border(calib[fx], img.size)
        if fx == "S":
            # S01-S05 (rightmost column, index 0-4 in base) are half hidden at the plate edge: no full round hole
            # is visible there, so treat them like the plate edge rather than as holes to match.
            hidden_idx = [i for i in range(30) if i // 5 == 5]
            border = np.vstack([border, base[hidden_idx]])
            match = base[[i for i in range(30) if i // 5 != 5]]
        else:
            match = base
        c0 = base.mean(axis=0)
        near = pts[(np.abs(pts - c0) < (np.ptp(base, axis=0) / 2 + (search_px + 45) * k)).all(axis=1)]
        best = None
        if len(near):
            def score_of(R, t):
                g = (match - c0) @ R.T + c0 + t
                b = (border - c0) @ R.T + c0 + t
                dg = np.min(np.linalg.norm(g[:, None, :] - near[None, :, :], axis=2), axis=1)
                db = np.min(np.linalg.norm(b[:, None, :] - near[None, :, :], axis=2), axis=1)
                inl = int((dg < tol).sum())
                ang = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                scl = abs(np.hypot(R[0, 0], R[1, 0]) - 1)
                prior = 0.01 * float(np.hypot(*t)) / k + 0.15 * ang + 8 * scl     # prefer small corrections
                return inl - 1.5 * int((db < tol).sum()) - 0.05 * float(dg[dg < tol].mean() if inl else 0) - prior, inl
            for rot in np.radians(np.arange(-4, 4.01, 1.0)):
                for sc in np.arange(0.94, 1.0601, 0.02):
                    R = sc * np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                    pred = (match - c0) @ R.T + c0
                    votes = (near[None, :, :] - pred[:, None, :]).reshape(-1, 2)
                    votes = votes[(np.abs(votes) <= search_px * k).all(axis=1)]
                    if not len(votes):
                        continue
                    b = 2.0 * k
                    keys, cnt = np.unique(np.round(votes / b).astype(int), axis=0, return_counts=True)
                    for idx in np.argsort(-cnt)[:6]:
                        t = keys[idx] * b
                        sc_, inl = score_of(R, t)
                        if best is None or sc_ > best[0]:
                            best = (sc_, inl, R, t)
            # also score "no change" so a good calibration is never made worse
            sc0, inl0 = score_of(np.eye(2), np.zeros(2))
            if best is None or sc0 >= best[0] - 1.5:
                best = (sc0, inl0, np.eye(2), np.zeros(2))
        if best is None or best[1] < 8:
            new[fx], quality[fx] = dict(calib[fx]), 0 if best is None else best[1]
            continue
        _, inl, R, t = best
        corners = {}
        for name, (x, y) in calib[fx].items():
            q = R @ (np.array([x * k, y * k]) - c0) + c0 + t
            corners[name] = (round(float(q[0] / k), 1), round(float(q[1] / k), 1))
        new[fx], quality[fx] = corners, inl
    return new, quality


def alignment_quality(img: Image.Image, calib: dict) -> dict:
    """How many of the 30 grid positions per fixture sit on a detected round hole/cap (within ~4 px)."""
    k = img.width / BASE_W
    pts = detect_holes(img)
    out = {}
    for fx in ("G", "S"):
        g = _grid_from_corners(calib[fx], img.size)
        if not len(pts):
            out[fx] = 0
            continue
        d = np.min(np.linalg.norm(g[:, None, :] - pts[None, :, :], axis=2), axis=1)
        out[fx] = int((d < 4.5 * k).sum())
    return out


def hole_radius(img_size: Tuple[int, int]) -> float:
    return HOLE_RADIUS_BASE * img_size[0] / BASE_W


def crop_square(img: Image.Image, cx: float, cy: float, side: float) -> Image.Image:
    """Square crop centered on (cx, cy). Out-of-image areas are black, so it never raises."""
    half = side / 2.0
    box = (int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half)))
    return img.crop(box)


def upscale(img: Image.Image, min_side: int = 320) -> Image.Image:
    if img.width >= min_side:
        return img
    return img.resize((min_side, min_side), Image.LANCZOS)


# --------------------------------------------------------------------------------------
# Local computer-vision features (cheap, deterministic, no API)
# --------------------------------------------------------------------------------------
@dataclass
class LocalFeatures:
    yellow: float      # fraction of yellow pixels where a cap would sit
    center_cov: float  # fraction of the hole CENTER covered by yellow (a seated cap covers the opening)
    p90: float         # 90th-percentile brightness (V) inside the hole: empty dark hole is low
    std: float         # brightness spread inside the hole: empty hole is uniform (low)
    cap_area: float = 0.0   # nearest whole-cap-sized yellow blob, in hole areas (one cap ~0.9-1.0; two merged ~1.4+)
    cap_off: float = 9.0    # its distance from the hole centre, in hole radii (a neighbour's cap is ~2.9 away)


def cap_blob(crop: Image.Image, r: float) -> Tuple[float, float]:
    """The yellow cap blob nearest the hole: (area in hole areas, distance from the hole centre in hole radii).
    One whole cap ~0.9-1.0; two touching caps (e.g. a neighbour's cap spilling over an open vial) ~1.4+."""
    a = np.array(crop.convert("RGB")); h = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    grey = h[..., 1] < 60
    plate = float(np.percentile(h[..., 2][grey], 75)) if grey.sum() > 50 else 255.0
    vf = min(YELLOW_V_MIN, max(YELLOW_V_FLOOR_DIM, 1.1 * plate))
    yl = (h[..., 0] >= 18) & (h[..., 0] <= 40) & (h[..., 1] >= 90) & (h[..., 2] >= 0.55 * vf)
    n, _, st, ce = cv2.connectedComponentsWithStats(yl.astype(np.uint8))
    c = a.shape[0] / 2.0
    big = [k for k in range(1, n) if st[k][4] >= 0.3 * np.pi * r * r]
    if not big:
        return 0.0, 9.0
    k = min(big, key=lambda k: np.hypot(ce[k][0] - c, ce[k][1] - c))
    return float(st[k][4] / (np.pi * r * r)), float(np.hypot(ce[k][0] - c, ce[k][1] - c) / r)


def local_features(crop: Image.Image, r: float) -> LocalFeatures:
    """crop is square with the target hole at its center; r = hole radius in crop pixels."""
    arr = np.array(crop.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0

    # Caps sit slightly up/left of the hole in this camera view (perspective), so look a bit wider.
    cap_mask = (xx - cx) ** 2 + (yy - (cy - 0.3 * r)) ** 2 <= (1.25 * r) ** 2
    # Cap yellow is bright and lemon-yellow (hue ~24-31). Glass seen down an uncapped vial under warm
    # light is orange-brown (hue ~18-21) and dimmer, so it must NOT count as a cap.
    # Brightness floor for "yellow" follows the light: normally 170, lower only when the whole scene is dim
    # (the digital-twin lighting sweep showed caps dropping below 170 when the station light falls ~40 %).
    # The plate (low-saturation metal) tells us how bright the light is; on the real camera it is ~190-240,
    # so the floor stays at 170 there and nothing changes.
    grey = hsv[..., 1] < 60
    plate = float(np.percentile(hsv[..., 2][grey], 75)) if grey.sum() > 50 else 255.0
    v_floor = min(YELLOW_V_MIN, max(YELLOW_V_FLOOR_DIM, 1.1 * plate))
    yellow = (hsv[..., 0] >= 22) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= v_floor)
    yellow_frac = float(yellow[cap_mask].mean()) if cap_mask.any() else 0.0

    center = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.3 * r) ** 2
    center_cov = float(yellow[center].mean()) if center.any() else 0.0

    inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.6 * r) ** 2
    v = hsv[..., 2][inner].astype(float)
    area, off = cap_blob(crop, r)
    return LocalFeatures(yellow_frac, center_cov, float(np.percentile(v, 90)), float(v.std()), area, off)


# Thresholds were read off ONE example frame (see README) - they are triage heuristics, not the
# final judge. Anything not clearly 'cap' or clearly 'empty' goes to the vision model.
YELLOW_PRESENT = 0.20     # >= this: a yellow cap is present
YELLOW_ABSENT = 0.05      # <= this: no yellow cap
EMPTY_P90 = 115           # clearly empty hole: dark ...
YELLOW_V_MIN = 170         # cap yellow must be at least this bright under normal light
YELLOW_V_FLOOR_DIM = 120   # ...and never below this, however dim the scene
CENTER_COV_MIN = 0.45     # a seated cap covers >=~66% of the hole centre on the example frame; displaced caps ~0-25%
EMPTY_STD = 30            # ... and uniform


EMPTY_P90_LOOSE = 130     # looser darkness limit when nothing yellow covers the hole centre (neighbour bleed)


# The official label definitions (CSI Dataset 1). Shown on every flagged card so each decision is tied to them.
OFFICIAL_LABELS = {
    "pass": "The visible cap and vial body appear acceptable. It does not confirm properties hidden from the camera.",
    "missing_cap": "A vial is present, but its mouth is open. Do not confuse the transparent vial body with an empty "
                   "fixture hole.",
    "crooked_cap": "The cap is visibly tilted or not seated. A leaning vial by itself is not enough.",
    "damaged_vial": "The target vial body is visibly damaged. Overlapping transparent bodies may require the full "
                    "scene for context.",
    "no_vial": "The target aperture is empty. Whether that is an error depends on fill progress.",
}


LEAN_NOTE = ("From straight above, a leaning vial with a seated cap and a tilted cap look alike: in the digital "
             "twin, 55 of 128 tilted caps still looked like neat round caps. So a person checks - a quick look from "
             "the side settles it (leaning vial = OK, cap tipped on the vial = crooked).")


def local_guess(f: LocalFeatures) -> str:
    """Rule-based (no training on real data, per the lab rules).
    'empty' = clearly dark, uniform empty hole (skips the model); 'cap' = lemon-yellow cap present;
    'uncertain' = everything else -> the vision model decides."""
    if f.center_cov < 0.05 and f.p90 < EMPTY_P90_LOOSE and f.std < EMPTY_STD:
        return "empty"
    if f.yellow >= YELLOW_PRESENT:
        return "cap"
    if f.yellow <= YELLOW_ABSENT and f.p90 < EMPTY_P90 and f.std < EMPTY_STD:
        return "empty"
    return "uncertain"


# --------------------------------------------------------------------------------------
# NVIDIA VLM
# --------------------------------------------------------------------------------------
def _jpeg_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def parse_vlm_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON object returned: {text[:120]!r}")
    obj = json.loads(m.group(0))
    label = str(obj.get("label", "")).strip().lower().replace(" ", "_")
    if label not in LABELS:
        raise ValueError(f"Label {label!r} is not one of {LABELS}")
    conf = obj.get("confidence", None)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = None
    def _b(v):
        return v if isinstance(v, bool) else str(v).strip().lower() == "true"
    return {"label": label, "confidence": conf, "reason": str(obj.get("reason", ""))[:300],
            "opening_visible": _b(obj.get("opening_visible", False)),
            "cap_tilted": _b(obj.get("cap_tilted", False))}


def call_vlm(crop: Image.Image, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 90,
             retries: int = 3, url: str = NVIDIA_URL, prompt: Optional[str] = None) -> dict:
    """Returns {'label','confidence','reason','latency_ms'} or {'error': str,'latency_ms'}.
    Never raises: refusals / timeouts / bad JSON all come back as an error (-> REVIEW)."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(crop)}"}},
            ],
        }],
        "max_tokens": 160,
        "temperature": 0.1,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    last_err = "unknown error"
    t0 = time.time()
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {r.text[:200]}",
                        "latency_ms": (time.time() - t0) * 1000}
            text = r.json()["choices"][0]["message"]["content"]
            out = parse_vlm_json(text)
            out["latency_ms"] = (time.time() - t0) * 1000
            return out
        except Exception as e:  # noqa: BLE001 - we want to retry on anything
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"error": last_err, "latency_ms": (time.time() - t0) * 1000}


# --------------------------------------------------------------------------------------
# Decision logic
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    pid: str
    label: str = ""               # one of LABELS, or "" if unknown
    status: str = "REVIEW"        # PASS / REJECT / REVIEW / EMPTY / GAP (local-only mode: CAP)
    reason: str = ""
    source: str = ""              # 'vlm' | 'local' | 'conflict' | 'error'
    confidence: Optional[float] = None
    yellow: float = 0.0
    center_cov: float = 0.0
    p90: float = 0.0
    latency_ms: Optional[float] = None
    cap_area: float = 0.0
    cap_off: float = 9.0


CAP_CLEARLY_SEATED = 0.90   # cap covers >=90% of the hole centre


def decide(pid: str, feats: LocalFeatures, vlm: Optional[dict], require_cap_for_pass: bool = True,
           strict: bool = False, second: Optional[dict] = None) -> Result:
    """Combine local CV and (optional) VLM output. Priority: never let a defect escape,
    so any error / refusal / disagreement becomes REVIEW, never PASS."""
    res = Result(pid=pid, yellow=feats.yellow, center_cov=feats.center_cov, p90=feats.p90,
                 cap_area=feats.cap_area, cap_off=feats.cap_off)
    guess = local_guess(feats)

    # ---- local only mode ----
    if vlm is None:
        res.source = "local"
        if guess == "cap" and feats.center_cov < CENTER_COV_MIN:
            res.label, res.status = "crooked_cap", "REVIEW"
            res.reason = (f"Cap is off the vial opening (cap over hole {feats.center_cov:.0%}): leaning vial or "
                          f"tilted cap? {LEAN_NOTE}")
        elif guess == "cap":
            res.label, res.status, res.reason = "pass", "CAP", "Yellow cap present. Local check only - crooked caps and damage are NOT assessed"
        elif guess == "empty":
            res.label, res.status, res.reason = "no_vial", "EMPTY", "Dark, uniform empty hole"
        elif feats.yellow <= YELLOW_ABSENT:
            res.label, res.status = "missing_cap", "REVIEW"
            res.reason = "Something is in the hole but no yellow cap - possible uncapped vial (needs the vision model)"
        else:
            res.status, res.reason = "REVIEW", "Not clearly capped or empty - needs the vision model"
        return res

    # ---- VLM mode ----
    if "error" in vlm:
        res.source, res.status = "error", "REVIEW"
        res.reason = (f"No model answer (the call failed: {vlm['error'][:110]}). The camera checks alone cannot "
                      f"clear it, so a person looks. Press Inspect again to retry.")
        res.latency_ms = vlm.get("latency_ms")
        return res

    res.label, res.confidence, res.latency_ms = vlm["label"], vlm.get("confidence"), vlm.get("latency_ms")
    res.reason, res.source = vlm.get("reason", ""), "vlm"

    if vlm["label"] == "no_vial":
        if guess == "cap":
            res.status, res.source = "REVIEW", "conflict"
            res.reason = f"VLM says empty but a yellow cap is visible (yellow={feats.yellow:.0%}). " + res.reason
        else:
            res.status = "EMPTY"
    elif vlm["label"] == "pass" and USE_VIAL_CHECK and needs_vial_check(feats, vlm) and second and "label" in second:
        if second["label"] == "missing_cap":
            res.label, res.status, res.source = "missing_cap", "REJECT", "second look"
            res.reason = f"Open vial: no cap over the opening, and a second look sees a glass vial. {second.get('reason', '')}"
        elif second["label"] == "no_vial" and TRUST_EMPTY_CHECK and guess != "cap":
            res.label, res.status, res.source = "no_vial", "EMPTY", "second look"
            res.reason = f"Empty hole (first answer 'pass' had no cap behind it; second look: empty). {second.get('reason', '')}"
        else:
            res.status, res.source = "REVIEW", "conflict"
            res.reason = (f"Model says pass but no cap covers the opening; second look said "
                          f"'{second['label']}' - check by eye. " + res.reason)
    elif vlm["label"] == "pass":
        if require_cap_for_pass and feats.yellow <= YELLOW_ABSENT:
            res.status, res.source = "REVIEW", "conflict"
            res.reason = f"VLM says pass but no yellow cap was found (yellow={feats.yellow:.0%}) - check for an open vial. " + res.reason
        elif require_cap_for_pass and feats.center_cov < CENTER_COV_MIN:
            # Hard safety rule: a PASS needs a yellow cap over the vial opening.
            res.status, res.source = "REVIEW", "conflict"
            res.reason = (f"VLM says pass but the cap does not cover the vial opening "
                          f"(coverage {feats.center_cov:.0%}): leaning vial or tilted cap? {LEAN_NOTE} " + res.reason)
        elif (vlm.get("opening_visible") or vlm.get("cap_tilted")) and (strict or feats.center_cov < CAP_CLEARLY_SEATED):
            res.status, res.source = "REVIEW", "conflict"
            what = "the vial opening is visible" if vlm.get("opening_visible") else "the cap is tilted"
            res.reason = f"VLM says pass but also reports {what}. " + res.reason
        else:
            res.status = "PASS"
    elif vlm["label"] == "missing_cap" and feats.center_cov >= CENTER_COV_MIN and feats.yellow >= YELLOW_PRESENT:
        # Physically contradictory: a lemon-yellow cap covers the vial opening, so the cap is not missing.
        # On the development frames the model made this mistake 8 times out of 8 such cases.
        # Second look says pass and the camera confirms a cap over the opening (the same 45 % rule a first-look pass
        # needs) -> PASS. (v26 asked for 90 % here, which sent seated caps on a new camera to review although the
        # model's second look and the camera agreed.)
        if second and second.get("label") == "pass" and feats.center_cov >= CENTER_COV_MIN:
            res.label, res.status, res.source = "pass", "PASS", "second look"
            res.reason = (f"First answer 'missing cap' contradicted the yellow cap covering the opening "
                          f"({feats.center_cov:.0%}); a second look says the cap is seated. {second.get('reason', '')}")
        elif second and second.get("label") == "crooked_cap" and feats.center_cov >= CAP_CLEARLY_SEATED:
            res.label, res.status, res.source = "crooked_cap", "REJECT", "second look"
            res.reason = f"Cap present but crooked (second look). {second.get('reason', '')}"
        else:
            # A cap is over the opening, so "missing cap" is physically wrong; whether it is seated, tipped or the
            # vial just leans cannot be settled from straight above -> a person decides (never PASS, never blind REJECT).
            res.label, res.status, res.source = "crooked_cap", "REVIEW", "conflict"
            sec = f" Second look: {second.get('label')}." if second and "label" in second else ""
            res.reason = (f"Model said 'missing cap', but a yellow cap covers {feats.center_cov:.0%} of the vial "
                          f"opening, so the mouth is not open.{sec} Leaning vial or tilted cap? {LEAN_NOTE}")
    elif vlm["label"] == "crooked_cap" and not strict and CENTER_COV_MIN <= feats.center_cov < CAP_CLEARLY_SEATED:
        # Cap mostly over the opening but off-centre: exactly what a leaning vial looks like from above.
        res.status, res.source = "REVIEW", "conflict"
        res.reason = (f"Model says crooked cap; the cap covers {feats.center_cov:.0%} of the opening. Official rule: "
                      f"a leaning vial by itself is not a defect. {LEAN_NOTE} " + res.reason)
    elif vlm["label"] in ("crooked_cap", "missing_cap") and not strict and feats.center_cov < CENTER_COV_MIN \
            and own_whole_cap(res):
        # This vial's own whole cap sits next to the hole: leaning vial (OK) or tipped/fallen cap (defect).
        res.label, res.status, res.source = "crooked_cap", "REVIEW", "conflict"
        res.reason = (f"Model says {vlm['label'].replace('_', ' ')}; a whole cap sits next to the hole "
                      f"({feats.center_cov:.0%} over the opening): leaning vial or tipped cap? {LEAN_NOTE}")
    elif (vlm["label"] == "crooked_cap" and not strict and feats.center_cov >= CAP_CLEARLY_SEATED):
        # The model says crooked, but the local check sees a round cap fully covering the vial opening.
        # On this camera that is usually perspective, so ask a human instead of auto-rejecting.
        res.status, res.source = "REVIEW", "conflict"
        res.reason = (f"Model says crooked cap, but the cap fully covers the vial opening "
                      f"(coverage {feats.center_cov:.0%}) - may be camera perspective, check by eye. " + res.reason)
    else:
        res.status = "REJECT"
    return res


def decide_multi(pid: str, feats: LocalFeatures, vlms: List[Optional[dict]], tags: Optional[List[str]] = None,
                 require_cap_for_pass: bool = True, strict: bool = False) -> Result:
    """Combine several independent model opinions (e.g. wide crop + tight crop).
    A PASS must survive every opinion; any disagreement is REVIEW (never auto-accepted)."""
    rs = [decide(pid, feats, v, require_cap_for_pass, strict) for v in vlms]
    if len(rs) == 1:
        return rs[0]
    tags = tags or [f"pass{i + 1}" for i in range(len(rs))]
    sts = [r.status for r in rs]
    lat = [r.latency_ms for r in rs if r.latency_ms]
    if len(set(sts)) == 1:                      # every opinion agrees on the status
        out = rs[0]
    elif "REJECT" in sts and "PASS" not in sts:
        out = next(r for r in rs if r.status == "REJECT")
    else:
        out = rs[0]
        out.status, out.source = "REVIEW", "conflict"
        out.label = next((r.label for r in rs if r.label and r.status in ("REJECT", "REVIEW")), out.label)
        out.reason = "Opinions disagree - " + " || ".join(
            f"[{t}] {r.status} {r.label or '-'}: {r.reason[:110]}" for t, r in zip(tags, rs))
    out.latency_ms = sum(lat) if lat else None
    return out


ARM_MIN_AREA_FRAC = 4000 / (640 * 480)   # robot arm blob ~17,500 px; biggest cluster of caps seen ~1,100 px


def blocked_positions(img: Image.Image, centers: Dict[str, Tuple[float, float]]) -> List[str]:
    """Positions hidden under the yellow robot arm (a saturated-yellow blob far bigger than any cap)."""
    a = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 100)).astype(np.uint8)
    xs = [x for x, _ in centers.values()]
    ys = [y for _, y in centers.values()]
    pad = 2.5 * hole_radius(img.size)
    roi = np.zeros_like(m)
    roi[max(0, int(min(ys) - pad)):int(max(ys) + pad), max(0, int(min(xs) - pad)):int(max(xs) + pad)] = 1
    m *= roi
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    min_area = ARM_MIN_AREA_FRAC * img.width * img.height
    arm = np.zeros_like(m)
    for i in range(1, n):
        if st[i][4] >= min_area:
            arm[lab == i] = 1
    if not arm.any():
        return []
    k = max(3, int(2 * hole_radius(img.size)) | 1)
    arm = cv2.dilate(arm, np.ones((k, k), np.uint8))
    return [p for p, (x, y) in centers.items()
            if 0 <= int(y) < arm.shape[0] and 0 <= int(x) < arm.shape[1] and arm[int(y), int(x)]]


def blocked_result(pid: str, feats: LocalFeatures) -> Result:
    return Result(pid=pid, status="NO VIEW", source="geometry", yellow=feats.yellow, center_cov=feats.center_cov,
                  p90=feats.p90, reason="Covered by the robot arm in this frame - not judged. Use a frame "
                                        "where the arm is out of the way.")


def hidden_result(pid: str, feats: LocalFeatures) -> Result:
    return Result(pid=pid, status="NO VIEW", source="geometry", yellow=feats.yellow, center_cov=feats.center_cov,
                  p90=feats.p90, reason="Half hidden at the plate edge in this camera view - cannot be judged "
                                        "reliably. Inspect by eye or adjust the camera.")


# Measured results (see evaluation/RESULTS.md) - shown in the app's Evidence tab. Only measured numbers.
MEASURED = {
    "test": {"n": 321, "defects": 6, "good": 69, "base_caught": 1, "sys_caught": 6, "base_fa": 9, "sys_fa": 0},
    "empty": {"n": 247, "no_count": 13, "estimated": 4, "with_count": 0},
    "change": {"frames": 1097, "all": 16446, "aware": 248, "saved": "98.5%"},
    "align": {"cases": 54, "worst": 7.4, "median": 4.4, "seconds": 0.15},
    # Auto-fit (v26): 16 real labelled frames x 6 camera changes (2x resolution, 16:9 1920x1080, rotated 7 deg,
    # zoomed out, zoomed in, camera moved) -> straightened -> same decision as on the original frame.
    "autofit": {"variants": 6, "frames": 16, "same_min": 96.4, "same_max": 98.0, "standard_checked": 394,
                "standard_fitted": 0},
    # Round 2: 28 more real frames, labelled by eye first; 15 practice (look at mistakes) + 13 LOCKED (score only).
    # Local CV only (no API), image-estimated fill.
    "round2": {"frames": 28, "locked_frames": 13, "locked_good": 59, "locked_good_flagged": 0, "locked_defects": 8,
               "locked_defects_flagged": 6, "locked_lean": 2, "locked_empty": 642, "locked_empty_checks": 22,
               "locked_empty_checks_count": 0, "practice_good": 25, "practice_good_flagged": 0},
    # Per class, HELD-OUT real frames only (scored, never tuned on): eval_scripts/perclass.py.
    # model = NVIDIA model + rules on set A (6 frames, saved answers); local = camera checks only on sets A+B+C
    # (22 frames). "caught" = REJECT / REVIEW / GAP (a person or the line acts on it). Strict: labels as written.
    "perclass": {
        "model": {"frames": 6, "missing_cap": (3, 3), "crooked_cap": (3, 3), "pass_flagged": (0, 69),
                  "empty_flagged_est": (4, 247), "empty_flagged_count": (0, 247)},
        "local": {"frames": 22, "missing_cap": (7, 7), "crooked_cap": (4, 7), "missed_placement": (1, 1),
                  "pass_flagged": (1, 134), "empty_flagged_est": (32, 1046), "empty_flagged_count": (0, 1046)},
        "misses": "local: G19 (one leaning vial, labelled crooked_cap by eye) in 3 frames; model mode flags it",
    },
    # NVIDIA Isaac Sim + the CSI digital twin (isaac/gen_synth.py): 60 synthetic CapSealCam frames, 1,800 G positions,
    # labels exact by construction. Local checks before/after the lighting fix that the twin exposed.
    "twin": {"frames": 60, "positions": 1800, "defects": 286, "missing": 141, "crooked": 145, "missed_placements": 59,
             "good": 950, "good_before": 907, "good_after": 950, "dim": 378, "dim_before": 338, "dim_after": 378,
             "est_before": 42, "est_after": 55, "esc_before": 9, "esc_after": 1, "esc_count": 0,
             "real_checked": 17820, "real_changed": 0, "grid_px": 2.7,
             "crooked_round": 55, "crooked_measured": 128,
             # second set: 150 frames with leaning vials (3-12 deg, cap seated = good)
             "lean_frames": 150, "lean_n": 506, "lean_review": 221, "lean_rej_before": 66, "lean_rej_after": 20,
             "lean_quick": 162, "lean_crooked_n": 355, "shape_cleared": 110, "shape_slipped": 3, "shape_crooked": 152,
             "loop": [
                 # learn_loop.py: twin-only model, fixed held-out twin test set (47 frames); real frames looked at only
                 {"round": "1 basic", "twin_defects": "208/215", "twin_good": "150/753", "twin_lean": "147/188",
                  "real_good": "123/141", "real_defects": "15/15"},
                 {"round": "2 + leaning vials", "twin_defects": "200/215", "twin_good": "28/753", "twin_lean": "24/188",
                  "real_good": "101/141", "real_defects": "7/15"},
                 {"round": "3 + camera shift", "twin_defects": "202/215", "twin_good": "9/753", "twin_lean": "8/188",
                  "real_good": "101/141", "real_defects": "8/15"},
                 {"round": "4 + learn from mistakes", "twin_defects": "201/215", "twin_good": "13/753", "twin_lean": "12/188",
                  "real_good": "106/141", "real_defects": "11/15"}],
             "clf_real_defects": 15, "clf_real_caught": 15, "clf_real_good": 141, "clf_real_good_flagged": 141},
}


# --------------------------------------------------------------------------------------
# Change-aware inspection: only re-inspect positions whose picture changed since they were last inspected
# --------------------------------------------------------------------------------------
CHANGE_T = 0.15   # lighting-normalised mean abs difference. Labelled frame pairs: real changes >= 0.35,
                  # unchanged positions <= 0.23 -> 0.15 leaves a wide margin on the safe side.


def position_signatures(img: Image.Image, centers: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    """Small, lighting-normalised grey patch per position (24x24), for change detection between frames."""
    g = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(float)
    h = int(1.6 * hole_radius(img.size))
    out = {}
    for p, (x, y) in centers.items():
        a = g[max(0, int(y) - h):int(y) + h, max(0, int(x) - h):int(x) + h]
        if a.size == 0:
            continue
        a = cv2.resize(a, (24, 24))
        out[p] = (a - a.mean()) / (a.std() + 8)
    return out


def changed_positions(prev: Dict[str, np.ndarray], cur: Dict[str, np.ndarray], t: float = CHANGE_T) -> List[str]:
    return [p for p in cur if p not in prev or float(np.abs(prev[p] - cur[p]).mean()) > t]


# --------------------------------------------------------------------------------------
# Closing the loop: turn results into line actions + a PLC/MQTT-style message
# --------------------------------------------------------------------------------------
def line_actions(results: Dict[str, "Result"], placed: Optional[Dict[str, Optional[int]]] = None,
                 frame: str = "", station: str = "CSI_FinalVialInspection") -> dict:
    """What the cell should do next. Good-fixture defects must leave the good fixture (DIVERT_TO_SCRAP); a missed
    placement needs a robot re-place; anything uncertain needs an operator. The line HOLDs until good-fixture
    defects and gaps are resolved."""
    actions, mask = [], {"G": {"reject": 0, "review": 0, "gap": 0}, "S": {"reject": 0, "review": 0, "gap": 0}}
    for pid in sorted(results):
        r = results[pid]
        fx, n = pid[0], int(pid[1:])
        bit = 1 << (n - 1)
        if r.status == "REJECT":
            mask[fx]["reject"] |= bit
            actions.append({"position": pid, "action": "DIVERT_TO_SCRAP" if fx == "G" else "LOG_SCRAP_DEFECT",
                            "defect": r.label, "why": r.reason[:160]})
        elif r.status == "GAP":
            mask[fx]["gap"] |= bit
            actions.append({"position": pid, "action": "REPLACE_MISSED_VIAL", "defect": "missed_placement",
                            "why": r.reason[:160]})
        elif r.status == "REVIEW":
            mask[fx]["review"] |= bit
            actions.append({"position": pid, "action": "OPERATOR_CHECK", "defect": r.label or "uncertain",
                            "why": r.reason[:160]})
    blocked = [p for p, r in results.items() if r.status == "NO VIEW" and "robot arm" in r.reason]
    if blocked:
        actions.append({"position": ",".join(sorted(blocked)), "action": "RECAPTURE_FRAME",
                        "defect": "occluded", "why": "Robot arm in view - take the next frame."})
    hold = any(a["action"] in ("DIVERT_TO_SCRAP", "REPLACE_MISSED_VIAL") for a in actions)
    review = any(a["action"] == "OPERATOR_CHECK" for a in actions)
    good_ok = sum(1 for p, r in results.items() if p[0] == "G" and r.status == "PASS")
    complete = bool(placed and placed.get("G") == 30 and good_ok == 30)
    return {
        "station": station, "frame": frame,
        "line_state": "HOLD" if hold else ("CHECK" if review else "CONTINUE"),
        "order_complete": complete,
        "good_fixture": {"passed": good_ok, "placed": (placed or {}).get("G")},
        "plc_tags": {f"{station}.{fx}_{k}_mask": v for fx in ("G", "S") for k, v in mask[fx].items()},
        "actions": actions,
    }


def apply_placement_count(results: Dict[str, "Result"], placed: Optional[Dict[str, Optional[int]]]) -> Dict[str, "Result"]:
    """Use the robot's placement count (from the PLC/robot, or typed in by the operator) to settle the case the
    vision model cannot: a hole with NO cap could be empty or hold an uncapped glass vial.
      - position number > count: the robot has not placed a vial there yet -> a no-cap hole is EMPTY (waiting).
        A visible yellow cap there is unexpected -> REVIEW.
      - position number <= count: a vial must be there -> a no-cap hole is a defect (missing cap or missed
        placement) -> REJECT.
    count None = unknown (nothing changes)."""
    if not placed:
        return results
    for prefix in ("G", "S"):
        n = placed.get(prefix)
        if n is None:
            continue
        for i in range(1, 31):
            pid = f"{prefix}{i:02d}"
            r = results.get(pid)
            if r is None or r.status == "NO VIEW":
                continue
            # Judge "no cap" by the hole CENTRE only: a neighbour's cap can leak yellow into the crop's edge, but a
            # seated cap always covers the centre (good vials on the dev frames: >= 62 %; open vials: 0 %).
            no_cap = r.center_cov < 0.05
            if i > n:
                if no_cap and r.status in ("REVIEW", "REJECT") and r.label in ("", "pass", "missing_cap", "no_vial"):
                    r.label, r.status, r.source = "no_vial", "EMPTY", "placement count"
                    r.reason = (f"Beyond the placement count ({n}): no vial placed yet, and no cap is visible. "
                                f"(Model's first answer ignored - it cannot tell empty holes from glass.)")
                elif r.yellow >= YELLOW_PRESENT and r.status in ("PASS", "CAP"):
                    r.status, r.source = "REVIEW", "placement count"
                    r.reason = f"A capped vial is visible beyond the placement count ({n}) - unexpected vial? " + r.reason
            else:
                if no_cap and r.status in ("REVIEW", "REJECT") and r.label in ("", "pass", "missing_cap") and own_whole_cap(r):
                    # Its own whole cap is right next to the hole: a vial leaning so far that the cap is off the opening,
                    # or a cap that fell beside it. Not a blind REJECT - a one-click check (digital-twin finding).
                    r.label, r.status, r.source = "crooked_cap", "REVIEW", "placement count"
                    r.reason = f"A whole cap is next to the hole but not over it: leaning vial or tipped cap? {LEAN_NOTE}"
                elif no_cap and r.status == "REVIEW" and r.label in ("", "pass", "missing_cap"):
                    r.label, r.status, r.source = "missing_cap", "REJECT", "placement count"
                    r.reason = (f"The robot has placed {n} vial(s), so {pid} should hold a capped vial, but no cap covers "
                                f"the opening - missing cap or missed placement.")
    return results


def evidence_markdown(m: Optional[dict] = None) -> str:
    """The measured results as Markdown, for the executive summary / report. Only numbers stored in MEASURED."""
    m = m or MEASURED
    t, e, pc, tw, r2 = m["test"], m["empty"], m["perclass"], m["twin"], m["round2"]
    f = lambda x: f"{x[0]}/{x[1]}" if x else "-"
    md, lo = pc["model"], pc["local"]
    lines = [
        "# Final vial inspection - measured results", "",
        "All real-frame numbers: CSI overhead camera frames (Dataset 1), labelled by eye before any system output. "
        "Held-out frames were only scored, never tuned on. \"Caught\" = REJECT, REVIEW or GAP.", "",
        "## Against a baseline (held-out, 6 frames)", "",
        "| | Model alone (baseline) | This system |", "|---|---|---|",
        f"| Defects caught | {t['base_caught']}/{t['defects']} | {t['sys_caught']}/{t['defects']} |",
        f"| Good vials flagged | {t['base_fa']}/{t['good']} | {t['sys_fa']}/{t['good']} |",
        f"| Empty holes flagged (no count / image fill / robot count) | - | {e['no_count']} / {e['estimated']} / "
        f"{e['with_count']} of {e['n']} |", "",
        "## Per class (held-out only)", "",
        f"| Class | NVIDIA model + rules ({md['frames']} frames) | Camera checks only ({lo['frames']} frames) |",
        "|---|---|---|",
        f"| missing_cap caught | {f(md.get('missing_cap'))} | {f(lo.get('missing_cap'))} |",
        f"| crooked_cap caught | {f(md.get('crooked_cap'))} | {f(lo.get('crooked_cap'))} |",
        f"| missed placement caught (GAP) | {f(md.get('missed_placement'))} | {f(lo.get('missed_placement'))} |",
        "| damaged_vial | no real example | no real example |",
        f"| good vials flagged (false alarms) | {f(md.get('pass_flagged'))} | {f(lo.get('pass_flagged'))} |",
        f"| empty holes flagged, robot count | {f(md.get('empty_flagged_count'))} | {f(lo.get('empty_flagged_count'))} |",
        f"| empty holes flagged, fill from image | {f(md.get('empty_flagged_est'))} | {f(lo.get('empty_flagged_est'))} |",
        "", f"Missed defects (false negatives): {pc['misses']}.", "",
        f"Locked round 2 ({r2['locked_frames']} frames, camera checks only): good flagged "
        f"{r2['locked_good_flagged']}/{r2['locked_good']}, defects flagged {r2['locked_defects_flagged']}/"
        f"{r2['locked_defects']} (the 2 unflagged are the leaning G19).", "",
        "## Digital twin (NVIDIA Isaac Sim)", "",
        f"- {tw['frames']} synthetic frames, {tw['positions']} positions, {tw['defects']} defects "
        f"({tw['missing']} missing, {tw['crooked']} crooked), {tw['missed_placements']} missed placements; labels exact.",
        f"- Escaped defects on twin frames after the fixes the twin exposed: {tw['esc_before']} -> {tw['esc_after']} "
        f"(image fill), {tw['esc_count']} with the robot count; {tw['real_changed']} of {tw['real_checked']:,} real "
        "positions changed decision with those fixes.",
        f"- Leaning vials ({tw['lean_frames']} frames, {tw['lean_n']} leaning vials): rejected as missing cap "
        f"{tw['lean_rej_before']} -> {tw['lean_rej_after']}.",
        f"- A classifier trained on twin images only catches {tw['clf_real_caught']}/{tw['clf_real_defects']} real "
        f"defects but flags {tw['clf_real_good_flagged']}/{tw['clf_real_good']} good vials at that threshold, so it "
        "is not used in the decision.", "",
        "## Operations", "",
        f"- Change-aware: {m['change']['saved']} fewer model calls over {m['change']['frames']} real frames "
        f"({m['change']['all']:,} -> {m['change']['aware']:,}).",
        f"- Auto-align: worst error {m['align']['worst']} px over {m['align']['cases']} shifted/rotated cases "
        f"({m['align']['seconds']} s per image).",
        f"- Auto-fit to other cameras: {m['autofit']['same_min']:.0f}-{m['autofit']['same_max']:.0f}% same decisions "
        f"over {m['autofit']['frames']} frames x {m['autofit']['variants']} camera changes.", "",
        "## Limits", "",
        "- Small real test set (14 held-out defects; 3 crooked caps are one leaning vial in 3 frames); no damaged_vial.",
        "- Cap check tuned to the lemon-yellow cap; a new camera can place caps off the hole centre (flagged, not passed).",
        "- Model mode needs internet; a failed call becomes REVIEW, never PASS.",
    ]
    return "\n".join(lines) + "\n"


def estimate_placed(results: Dict[str, "Result"]) -> Dict[str, int]:
    """Estimate how far the robot has filled each fixture from the image alone: the last position (in fill order)
    that holds a clearly seated lemon-yellow cap (cap over the hole centre)."""
    est = {}
    for fx in ("G", "S"):
        # A position counts as filled if a cap sits over its opening OR its own whole cap sits right beside it
        # (a leaning vial still has its cap, so the robot has been there - found when auto-fit centred the grid).
        capped = [i for i in range(1, 31) if (r := results.get(f"{fx}{i:02d}")) is not None and r.status != "NO VIEW"
                  and r.yellow >= YELLOW_PRESENT and (r.center_cov >= CENTER_COV_MIN or own_whole_cap(r))]
        est[fx] = max(capped) if capped else 0
    return est


AUTO_MARGIN = 2   # positions right after the last capped vial stay "check by eye": the newest vial may be uncapped
# The image-only fill estimate ("last seated cap") is used for the GOOD fixture only. Scrap vials are often
# uncapped (that is why they are scrap), so "no cap beyond the last cap" says nothing there: on a new camera
# (Cap_seal2) it turned a scrap plate full of clear glass vials into "EMPTY". Scrap positions keep their own
# camera/model verdict instead (a typed robot count still applies to both fixtures).
ESTIMATE_FIXTURES = ("G",)


def apply_estimated_count(results: Dict[str, "Result"], est: Dict[str, int], margin: int = AUTO_MARGIN) -> Dict[str, "Result"]:
    """Like apply_placement_count, but with a count estimated from the image. Positions up to the last capped vial
    must hold a vial (no cap there -> REJECT); positions more than `margin` past it are waiting (no cap -> EMPTY);
    the `margin` positions in between are left as they are (usually REVIEW), because the newest vial could be an
    uncapped one."""
    for fx, fr in est.items():                 # only the fixtures given (a typed robot count takes precedence)
        if fx not in ESTIMATE_FIXTURES:
            continue
        for i in range(1, 31):
            pid = f"{fx}{i:02d}"
            r = results.get(pid)
            if r is None or r.status == "NO VIEW" or r.center_cov >= 0.05:
                continue
            if i <= fr and r.status == "REVIEW" and r.label in ("", "pass", "missing_cap") and own_whole_cap(r):
                r.label, r.status, r.source = "crooked_cap", "REVIEW", "fill pattern"
                r.reason = f"A whole cap is next to the hole but not over it: leaning vial or tipped cap? {LEAN_NOTE}"
            elif i <= fr and r.status == "REVIEW" and r.label in ("", "pass", "missing_cap"):
                r.label, r.status, r.source = "missing_cap", "REJECT", "fill pattern"
                r.reason = (f"Vials up to {fx}{fr:02d} are already placed (last seated cap), so {pid} should hold a capped "
                            f"vial, but no cap covers its opening - missing cap or missed placement.")
            elif fr < i <= fr + margin and r.status in ("REVIEW", "REJECT") and r.label in ("", "pass", "missing_cap"):
                # Right after the last seated cap: this is either the robot's newest (possibly uncapped) vial or a hole
                # still waiting. Nothing here can tell those apart, so a human looks - never PASS, never a blind REJECT.
                r.label, r.status, r.source = "missing_cap", "REVIEW", "fill pattern"
                r.reason = (f"Next in fill order after the last seated cap ({fx}{fr:02d}) and no cap visible: either the "
                            f"newest vial without a cap, or a hole still waiting - check by eye (or enter the placement count).")
            elif i > fr + margin and r.status in ("REVIEW", "REJECT") and r.label in ("", "pass", "missing_cap", "no_vial"):
                r.label, r.status, r.source = "no_vial", "EMPTY", "fill pattern"
                r.reason = (f"Beyond the last seated cap ({fx}{fr:02d}) in fill order and no cap visible - waiting for "
                            f"a vial. (Model's first answer ignored - it cannot tell empty holes from glass.)")
    return results


def apply_fill_pattern(results: Dict[str, Result], placed: Optional[Dict[str, int]] = None) -> Dict[str, Result]:
    """Robot fills 01..30 in order (rightmost column top-to-bottom, then one column left).
    If the placement count for a fixture is known (placed={"G": n}), positions 01..n must hold a vial:
    an empty one there is a missed placement (GAP), and later empties are just waiting.
    Without a count, fall back to: empty BEFORE the last occupied position = possible missed placement."""
    placed = placed or {}
    for prefix in ("G", "S"):
        ids = [f"{prefix}{n:02d}" for n in range(1, 31)]
        n_placed = placed.get(prefix) or 0
        if n_placed > 0:
            last, why = min(n_placed, 30), f"{n_placed} placement(s) reported for this fixture"
            cutoff = last                      # positions with index < cutoff should be filled
        else:
            # a position counts as occupied only if we actually know a vial is there
            # A position counts as filled only with real evidence of a vial: a pass/reject/cap verdict, or a
            # REVIEW where a cap is visible or a defect was named. A model "pass" on a hole with no cap at all
            # is NOT evidence (the model sometimes calls empty holes "pass").
            def filled(r):
                if r.status in ("PASS", "REJECT", "CAP"):
                    return True
                return r.status == "REVIEW" and (r.label in DEFECT_LABELS or r.yellow >= YELLOW_PRESENT)
            occupied = [i for i, pid in enumerate(ids) if pid in results and filled(results[pid])]
            if not occupied:
                continue
            cutoff = max(occupied)
            why = f"{ids[cutoff]} is already filled"
        for i, pid in enumerate(ids):
            r = results.get(pid)
            if r and r.status == "EMPTY":
                if i < cutoff:
                    r.status = "GAP"
                    r.reason = (r.reason + " | " if r.reason else "") + f"Empty but {why} - possible missed placement"
                else:
                    r.reason = (r.reason + " | " if r.reason else "") + "Empty at end of fill pattern (waiting for later vials)"
    return results


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------
STATUS_COLOR = {
    "PASS": (46, 204, 113), "REJECT": (231, 76, 60), "REVIEW": (243, 156, 18),
    "EMPTY": (149, 165, 166), "GAP": (155, 89, 182), "CAP": (52, 152, 219), "NO VIEW": (100, 116, 139), None: (0, 220, 255),
}
STATUS_ICON = {"PASS": "ok", "REJECT": "x", "REVIEW": "?", "GAP": "!", "EMPTY": None, "CAP": None, "NO VIEW": None, None: None}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def draw_overlay(img: Image.Image, centers: Dict[str, Tuple[float, float]],
                 results: Optional[Dict[str, Result]] = None, scale: float = 2.0,
                 highlight: Optional[List[str]] = None) -> Image.Image:
    """Circles + ID tags + status icons on the full scene, upscaled for crisp text."""
    base = img.convert("RGB").resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(base)
    R = hole_radius(img.size) * scale * 1.05
    font = _font(max(10, int(8 * scale)))
    for pid, (x, y) in centers.items():
        x, y = x * scale, y * scale
        res = results.get(pid) if results else None
        status = res.status if res else None
        col = STATUS_COLOR.get(status, STATUS_COLOR[None])
        w = 3 if (highlight and pid in highlight) else 2
        d.ellipse([x - R, y - R, x + R, y + R], outline=col, width=w)

        # ID tag below the circle
        tw = d.textlength(pid, font=font)
        tx, ty = x - tw / 2, y + R + 1
        d.rectangle([tx - 2, ty - 1, tx + tw + 2, ty + font.size + 1], fill=(10, 20, 35))
        d.text((tx, ty), pid, fill=col if status else (255, 255, 255), font=font)

        # status icon top-right
        icon = STATUS_ICON.get(status)
        if icon:
            ix, iy, ir = x + R * 0.8, y - R * 0.8, R * 0.45
            d.ellipse([ix - ir, iy - ir, ix + ir, iy + ir], fill=col)
            s = ir * 0.55
            if icon == "ok":
                d.line([(ix - s, iy), (ix - s * 0.2, iy + s * 0.8), (ix + s, iy - s * 0.7)], fill="white", width=max(2, int(scale)))
            elif icon == "x":
                d.line([(ix - s, iy - s), (ix + s, iy + s)], fill="white", width=max(2, int(scale)))
                d.line([(ix - s, iy + s), (ix + s, iy - s)], fill="white", width=max(2, int(scale)))
            else:
                d.text((ix - ir * 0.3, iy - ir * 0.75), icon, fill="white", font=_font(max(8, int(ir * 1.5))))
    return base


def mask_neighbors(crop: Image.Image, r: float, radius_factor: float = 2.2) -> Image.Image:
    """Gray out everything farther than radius_factor*r from the target so neighbouring caps can't mislead."""
    c = crop.width / 2.0
    R = radius_factor * r
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).ellipse([c - R, c - R, c + R, c + R], fill=255)
    return Image.composite(crop.convert("RGB"), Image.new("RGB", crop.size, (128, 128, 128)), mask)


def crop_with_ring(crop: Image.Image, r: float, color=(0, 220, 255)) -> Image.Image:
    """Upscaled crop with a thin ring on the target hole (so you can see which vial was judged)."""
    big = upscale(crop, 320)
    k = big.width / crop.width
    d = ImageDraw.Draw(big)
    c = big.width / 2
    R = r * k * 1.05
    d.ellipse([c - R, c - R, c + R, c + R], outline=color, width=2)
    return big


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate(pred: Dict[str, str], truth: Dict[str, str]) -> dict:
    """pred/truth: position_id -> label. Positions missing from either are skipped.
    Returns confusion matrix, per-class precision/recall, and defect-level FN / FP lists."""
    ids = [p for p in truth if p in pred and truth[p] in LABELS]
    cm = {t: {p: 0 for p in LABELS + ["REVIEW/none"]} for t in LABELS}
    for pid in ids:
        p = pred[pid] if pred[pid] in LABELS else "REVIEW/none"
        cm[truth[pid]][p] += 1
    per_class = {}
    for lab in LABELS:
        tp = cm[lab][lab]
        fn = sum(cm[lab].values()) - tp
        fp = sum(cm[t][lab] for t in LABELS if t != lab)
        per_class[lab] = {
            "support": tp + fn,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
        }
    # defect detection: positive = any defect label. A REVIEW counts as 'flagged' (caught).
    missed, false_alarm, caught, good_ok = [], [], 0, 0
    for pid in ids:
        t_def = truth[pid] in DEFECT_LABELS
        p = pred[pid]
        flagged = (p in DEFECT_LABELS) or (p not in LABELS)  # defect label or REVIEW/none
        if t_def and flagged:
            caught += 1
        elif t_def and not flagged:
            missed.append(pid)
        elif truth[pid] == "pass" and flagged:
            false_alarm.append(pid)
        elif truth[pid] == "pass":
            good_ok += 1
    n_def = caught + len(missed)
    n_good = good_ok + len(false_alarm)
    return {
        "n": len(ids), "confusion": cm, "per_class": per_class,
        "defects_total": n_def, "defects_caught": caught, "defects_missed": missed,
        "defect_recall": caught / n_def if n_def else None,
        "good_total": n_good, "false_alarms": false_alarm,
        "false_alarm_rate": len(false_alarm) / n_good if n_good else None,
    }


LEAN_AREA = (0.80, 1.20)   # one whole cap (digital twin leaning vials: median 0.99)
LEAN_MAX_OFF = 1.8         # hole radii; the nearest neighbour hole is ~2.9 radii away


def own_whole_cap(res: "Result") -> bool:
    """One whole cap (not two merged) close enough to belong to this vial, even if it leans off the hole."""
    return LEAN_AREA[0] <= res.cap_area <= LEAN_AREA[1] and res.cap_off <= LEAN_MAX_OFF


def lean_candidate(res: "Result") -> bool:
    """A REVIEW where this vial's own whole cap is present but sits off the opening: a leaning vial (OK by the
    official rule) or a tipped cap (defect). From straight above these look alike, so an operator confirms with one
    click. Two merged caps (an open vial next to a neighbour's cap) are NOT offered as a quick check."""
    return res.status == "REVIEW" and res.label in ("crooked_cap", "pass", "") and own_whole_cap(res)


def next_in_line_candidate(res: "Result") -> bool:
    """A REVIEW right after the last seated cap with no cap visible: an empty hole still waiting, or the newest vial
    without its cap. Kept as a safety margin (real frame 20:12:06: G20 is an uncapped vial exactly here), but it is a
    one-click operator check - or 0 such checks when the robot's placement count is given."""
    return res.status == "REVIEW" and res.source == "fill pattern" and res.label == "missing_cap" and res.center_cov < 0.05


def apply_operator(results: Dict[str, "Result"], decisions: Dict[str, str]) -> Dict[str, "Result"]:
    """Operator quick-check decisions: 'lean' -> PASS (leaning vial, cap seated), 'tipped' -> REJECT crooked_cap,
    'empty' -> EMPTY (hole waiting), 'uncapped' -> REJECT missing_cap."""
    for pid, d in decisions.items():
        r = results.get(pid)
        if r is None:
            continue
        if d == "lean":
            r.label, r.status, r.source = "pass", "PASS", "operator"
            r.reason = ("Operator confirmed: leaning vial, cap seated. Official rule: a leaning vial by itself is not a "
                        "defect.")
        elif d == "empty":
            r.label, r.status, r.source = "no_vial", "EMPTY", "operator"
            r.reason = "Operator confirmed: empty hole, waiting for a vial."
        elif d == "uncapped":
            r.label, r.status, r.source = "missing_cap", "REJECT", "operator"
            r.reason = "Operator confirmed: a vial without its cap."
        elif d == "tipped":
            r.label, r.status, r.source = "crooked_cap", "REJECT", "operator"
            r.reason = "Operator confirmed: the cap is tipped on the vial (crooked cap)."
        elif d.startswith("label:"):              # reviewer set the class in the review workstation
            lab = d.split(":", 1)[1]
            note = "Reviewer set the class"
            if lab == "pass":
                r.label, r.status = "pass", "PASS"
            elif lab == "no_vial":
                r.label, r.status = "no_vial", "EMPTY"
            elif lab in DEFECT_LABELS:
                r.label, r.status = lab, "REJECT"
            else:
                continue
            r.source = "operator"
            r.reason = f"{note}: {lab}. (System said: {r.reason[:160]})"
    return results


# ---- review workstation ---------------------------------------------------------------------------------------
REVIEW_CLASSES = ["pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial", "needs_review"]
CLASS_COLOR = {"pass": (46, 204, 113), "missing_cap": (245, 158, 11), "crooked_cap": (168, 85, 247),
               "damaged_vial": (239, 68, 68), "no_vial": (59, 130, 246), "needs_review": (148, 163, 184),
               "no_view": (71, 85, 105)}


def system_class(res: "Result") -> str:
    """The system's verdict as one of the official classes (or needs_review)."""
    if res.status in ("PASS", "CAP"):
        return "pass"
    if res.status in ("EMPTY", "GAP"):
        return "no_vial"            # GAP = an empty hole that should hold a vial (a missed placement)
    if res.status == "REJECT" and res.label in DEFECT_LABELS:
        return res.label
    if res.status == "NO VIEW":
        return "no_view"
    return "needs_review"


def draw_review_overlay(img: Image.Image, centers: Dict[str, Tuple[float, float]], classes: Dict[str, str],
                        reviewed: set, selected: Optional[str] = None, scale: float = 2.0) -> Image.Image:
    """Review-workstation view: ring colour = class, solid ring + green tick = reviewed, dashed ring = awaiting review,
    white halo = the position being looked at."""
    base = img.convert("RGB").resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(base)
    R = hole_radius(img.size) * scale * 1.05
    font = _font(max(10, int(8 * scale)))
    for pid, (x, y) in centers.items():
        x, y = x * scale, y * scale
        col = CLASS_COLOR.get(classes.get(pid, "needs_review"), (148, 163, 184))
        box = [x - R, y - R, x + R, y + R]
        if pid == selected:
            d.ellipse([x - R - 5, y - R - 5, x + R + 5, y + R + 5], outline=(255, 255, 255), width=3)
        if pid in reviewed:
            d.ellipse(box, outline=col, width=3)
        else:
            for a in range(0, 360, 30):           # dashed ring = awaiting review
                d.arc(box, a, a + 18, fill=col, width=3)
        tw = d.textlength(pid, font=font)
        d.rectangle([x - tw / 2 - 2, y + R + 1, x + tw / 2 + 2, y + R + font.size + 2], fill=(10, 20, 35))
        d.text((x - tw / 2, y + R + 1), pid, fill=col, font=font)
        if pid in reviewed:
            ix, iy, ir = x + R * 0.8, y - R * 0.8, R * 0.42
            d.ellipse([ix - ir, iy - ir, ix + ir, iy + ir], fill=(34, 197, 94))
            s = ir * 0.55
            d.line([(ix - s, iy), (ix - s * 0.2, iy + s * 0.8), (ix + s, iy - s * 0.7)], fill="white", width=max(2, int(scale)))
    return base


def explain(res: "Result") -> List[str]:
    """Plain-language reasons for a verdict, built only from what was measured and decided (for the 'Any vial' view)."""
    cov, yel = res.center_cov, res.yellow
    lines = []
    src = {"vlm": "the NVIDIA model + camera checks", "local": "the camera checks (local, no API)",
           "conflict": "a disagreement between the model and the camera checks", "second look": "a second question to the model",
           "placement count": "the robot's placement count", "fill pattern": "the robot's fill order",
           "operator": "the operator", "geometry": "the camera geometry", "error": "a model error"}.get(res.source, res.source or "-")
    lines.append(f"Decided by {src}.")
    if res.status == "PASS":
        lines.append(f"A yellow cap covers {cov:.0%} of the vial opening (a pass needs at least {CENTER_COV_MIN:.0%}), "
                     f"so the camera confirms a cap is on the vial.")
        if res.source in ("vlm", "second look"):
            conf = f" ({res.confidence:.0%} sure)" if res.confidence is not None else ""
            lines.append(f"The model also says pass{conf}, and nothing contradicts it - so no person needs to look.")
    elif res.status == "CAP":
        lines.append(f"A yellow cap covers {cov:.0%} of the opening, so a cap is present. Local mode cannot judge whether it "
                     f"is seated or tilted - in model mode the NVIDIA model checks that.")
    elif res.status == "EMPTY":
        if res.source in ("placement count", "fill pattern"):
            lines.append("No cap over the opening, and the robot has not reached this position yet in its fill order - "
                         "the hole is simply waiting for a vial.")
        else:
            lines.append(f"The hole is dark and even (brightest part {res.p90:.0f}/255, no yellow) - an empty hole.")
    elif res.status == "GAP":
        lines.append("Empty, but later positions in the robot's fill order already hold vials - a vial was probably "
                     "skipped here (missed placement).")
    elif res.status == "NO VIEW":
        lines.append("The camera cannot see this position properly (plate edge or robot arm), so no verdict is given "
                     "instead of guessing.")
    else:
        if cov >= CENTER_COV_MIN:
            lines.append(f"A yellow cap covers {cov:.0%} of the opening.")
        elif yel >= YELLOW_PRESENT:
            lines.append(f"Yellow is nearby but covers only {cov:.0%} of the opening - the cap is off the vial mouth "
                         f"(tipped, fallen, or the vial leans).")
        else:
            lines.append("No yellow cap over the opening.")
        if res.status == "REVIEW":
            lines.append("Not passed automatically: something is uncertain or contradictory, so a person checks it "
                         "(a missed defect is worse than a false alarm).")
        else:
            lines.append("Rejected: the evidence points to a defect.")
    if res.reason:
        lines.append("Details: " + res.reason)
    return lines



# ======================================================================================================================
# Auto-fit: find both fixtures in a photo of ANY size / framing and straighten it into the standard 640x480 view
# ======================================================================================================================
# Why rectify instead of re-scaling every threshold: every downstream step (cap colour/coverage rules, crop sizes,
# the model's crops, the twin-tested limits) was designed and validated in the standard camera view. Mapping a new
# photo INTO that view keeps all of it valid, instead of re-deriving each number for each camera.

def _hole_likeness(hsv: np.ndarray, p, r: float) -> float:
    """0..~1: a fixture position here looks like a hole (darker centre than the plate around it) or a cap (yellow)."""
    h, w = hsv.shape[:2]
    x, y = int(round(p[0])), int(round(p[1]))
    R = int(max(3, 1.7 * r))
    if x - R < 0 or y - R < 0 or x + R >= w or y + R >= h:
        return 0.0
    sub = hsv[y - R:y + R + 1, x - R:x + R + 1]
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.hypot(xx, yy)
    V = sub[..., 2].astype(float)
    centre, ring = V[d <= 0.55 * r], V[(d >= 1.25 * r) & (d <= 1.65 * r)]
    yel = ((sub[..., 0] >= 18) & (sub[..., 0] <= 40) & (sub[..., 1] >= 90))[d <= 0.8 * r].mean()
    return float(max(0.0, (np.median(ring) - centre.mean()) / 255.0) * 2 + yel)


def _lattice_points(gray: np.ndarray, hsv: Optional[np.ndarray] = None):
    """Round things at the scale where they form the most lattice-like pattern. Returns (pts, r, pitch-graph edges).
    Only HOLE-LIKE circles count (darker than the plate around them, or a yellow cap): round shapes in machinery or
    texture are not fixture holes."""
    W = gray.shape[1]
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    best = None
    for rr in W * np.geomspace(0.006, 0.045, 14):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=1.7 * rr, param1=90, param2=max(9.0, 1.3 * rr),
                             minRadius=max(3, int(0.75 * rr)), maxRadius=int(1.3 * rr) + 1)
        if c is None or len(c[0]) < 8:
            continue
        pts, rad = c[0][:, :2].astype(float), c[0][:, 2].astype(float)
        if hsv is not None:
            keep = np.array([_hole_likeness(hsv, p, q) > 0.12 for p, q in zip(pts, rad)])
            pts, rad = pts[keep], rad[keep]
            if len(pts) < 8:
                continue
        D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        rm = (rad[:, None] + rad[None]) / 2
        same = np.minimum(rad[:, None], rad[None]) / np.maximum(rad[:, None], rad[None]) > 0.72   # same-size holes
        dv = pts[None] - pts[:, None]
        ang = np.degrees(np.arctan2(np.abs(dv[..., 1]), np.abs(dv[..., 0])))
        axis = (ang < 25) | (ang > 65)       # neighbours lie along a row or a column, not on the diagonal
        A = (D >= 2.4 * rm) & (D <= 3.8 * rm) & same & axis
        np.fill_diagonal(A, False)
        score = int((A.sum(1) >= 2).sum())
        if best is None or score > best[0]:
            best = (score, pts, float(np.median(rad[A.sum(1) >= 2])) if score else rr, A)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _clusters(A: np.ndarray, min_size: int = 8):
    n = len(A)
    seen, out = np.zeros(n, bool), []
    for s in range(n):
        if seen[s] or not A[s].any():
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(A[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) >= min_size:
            out.append(np.array(comp))
    return sorted(out, key=len, reverse=True)


def _fit_lattice(pts: np.ndarray, A: np.ndarray, idx: np.ndarray, r: float):
    """Affine lattice p = O + i*U + j*V through one cluster. U = column step (left->right), V = row step (down)."""
    P = pts[idx]
    sub = A[np.ix_(idx, idx)]
    vec = np.array([P[b] - P[a] for a, b in zip(*np.nonzero(np.triu(sub)))])
    if not len(vec):
        return None
    horiz = np.abs(vec[:, 0]) >= np.abs(vec[:, 1])
    hv = vec[horiz] * np.sign(vec[horiz][:, :1] + 1e-9)
    vv = vec[~horiz] * np.sign(vec[~horiz][:, 1:2] + 1e-9)
    U = np.median(hv, axis=0) if len(hv) else None
    V = np.median(vv, axis=0) if len(vv) else None
    if U is None and V is None:
        return None
    if U is None:
        U = np.array([V[1], -V[0]]) / 1.15
    if V is None:
        V = np.array([-U[1], U[0]]) * 1.15
    seed = P[np.argmin(np.linalg.norm(P - P.mean(0), axis=1))]
    O, keep = seed, np.ones(len(P), bool)
    for _ in range(4):
        B = np.column_stack([U, V])
        ij = np.round(np.linalg.solve(B, (P - O).T).T)
        X = np.column_stack([np.ones(len(P)), ij])
        coef, *_ = np.linalg.lstsq(X[keep], P[keep], rcond=None)
        O, U, V = coef[0], coef[1], coef[2]
        res = np.linalg.norm(X @ coef - P, axis=1)
        keep = res < 0.45 * np.linalg.norm(U)
        if keep.sum() < 6:
            return None
    ij = np.round(np.linalg.solve(np.column_stack([U, V]), (P - O).T).T).astype(int)
    return dict(O=O, U=U, V=V, ij=ij[keep], pts=P[keep])


def _window_candidates(lat: dict, hsv: np.ndarray, r: float, prior_tl=None, hidden_right_col: bool = False, top: int = 6):
    """Score every 6x5 block of the lattice as 'the fixture': detected holes inside, positions that look like holes or
    caps, and NO hole-like spots just outside the block (there the plate ends) - so the block cannot slide by a
    column. For the scrap plate, also how close it is to where it sits relative to the good plate."""
    O, U, V, ij = lat["O"], lat["U"], lat["V"], lat["ij"]
    present = {tuple(x) for x in ij}
    pitch = np.linalg.norm(U)
    like = {}

    def L(i, j):
        if (i, j) not in like:
            like[(i, j)] = _hole_likeness(hsv, O + i * U + j * V, r)
        return like[(i, j)]
    out = []
    for i0 in range(ij[:, 0].min() - 3, ij[:, 0].max() + 2):
        for j0 in range(ij[:, 1].min() - 3, ij[:, 1].max() + 2):
            cols = range(i0, i0 + (5 if hidden_right_col else 6))
            inside = [(i, j) for i in cols for j in range(j0, j0 + 5)]
            ring = [(i, j) for i in (i0 - 1, i0 + 6) for j in range(j0, j0 + 5)] + \
                   [(i, j) for j in (j0 - 1, j0 + 5) for i in range(i0, i0 + 6)]
            if hidden_right_col:
                ring = [q for q in ring if q[0] != i0 + 5 and q[0] != i0 + 6]
            cnt = sum(q in present for q in inside)
            score = cnt + 0.6 * sum(L(*q) for q in inside) - 1.0 * sum(q in present for q in ring) \
                - 0.6 * sum(L(*q) for q in ring)
            if prior_tl is not None:
                score -= 2.0 * np.linalg.norm(O + i0 * U + j0 * V - prior_tl) / pitch
            c = lambda i, j: O + i * U + j * V  # noqa: E731
            out.append((score, {"TL": c(i0, j0), "TR": c(i0 + 5, j0), "BR": c(i0 + 5, j0 + 4), "BL": c(i0, j0 + 4)}, cnt))
    out.sort(key=lambda x: -x[0])
    return out[:top]


def _rel_S_from_G():
    """Where the scrap plate's top-left hole sits in the good plate's own lattice units (from the CSI calibration)."""
    g = {k: np.array(v, float) for k, v in DEFAULT_CALIB["G"].items()}
    U, V = (g["TR"] - g["TL"]) / 5, (g["BL"] - g["TL"]) / 4
    ab = np.linalg.solve(np.column_stack([U, V]), np.array(DEFAULT_CALIB["S"]["TL"], float) - g["TL"])
    return ab


def _border_hits(img: Image.Image, calib: dict) -> int:
    """Round holes found one pitch OUTSIDE the grids, where the plates end - a grid that slid by a row/column has some."""
    pts = detect_holes(img)
    if not len(pts):
        return 0
    k = img.width / BASE_W
    hits = 0
    for fx in ("G", "S"):
        _, border = _grid_and_border(calib[fx], img.size)
        d = np.min(np.linalg.norm(border[:, None, :] - pts[None, :, :], axis=2), axis=1)
        hits += int((d < 4.5 * k).sum())
    return hits


def _place_from(found: dict, fx_from: str, fx_to: str) -> dict:
    """Corners of one plate predicted from the other plate's detected corners, using the fixed layout of the two
    plates (affine fit of the CSI calibration)."""
    src = np.array([DEFAULT_CALIB[fx_from][k] for k in ("TL", "TR", "BR", "BL")], float)
    dst = np.array([found[k] for k in ("TL", "TR", "BR", "BL")], float)
    A, *_ = np.linalg.lstsq(np.column_stack([src, np.ones(4)]), dst, rcond=None)
    tgt = np.array([DEFAULT_CALIB[fx_to][k] for k in ("TL", "TR", "BR", "BL")], float)
    out = np.column_stack([tgt, np.ones(4)]) @ A
    return dict(zip(("TL", "TR", "BR", "BL"), out))


def auto_fit(img: Image.Image) -> Tuple[Image.Image, dict, dict]:
    """Find both fixtures in a photo of any size, aspect or framing, and warp it into the standard 640x480 view.
    Returns (standard-view image, calibration for it, info). info['ok'] is False if no fixture grid was found
    (then the image is just resized and the default grid is used - flagged to the user)."""
    rgb = np.array(img.convert("RGB"))
    H0, W0 = rgb.shape[:2]
    f = min(1.0, 1000.0 / max(W0, H0))
    work = cv2.resize(rgb, (int(W0 * f), int(H0 * f)), interpolation=cv2.INTER_AREA) if f < 1 else rgb
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    info = {"ok": False, "source_size": (W0, H0), "fixtures": []}
    found = _lattice_points(gray, hsv)
    corners = {}
    if found is not None:
        pts, r, A = found
        info["hole_radius_px"] = round(r / f, 1)
        lats = [lt for lt in (_fit_lattice(pts, A, c, r) for c in _clusters(A)[:3]) if lt is not None]
        lats = [lt for lt in lats if len(lt["ij"]) >= 8][:2]
        lats.sort(key=lambda lt: lt["pts"][:, 0].mean())         # good plate on the left, scrap on the right
        if len(lats) == 2 and lats[1]["pts"][:, 1].mean() < lats[0]["pts"][:, 1].mean() - 3 * r:
            lats = lats[::-1]                                     # scrap plate sits lower than the good one
        ab = _rel_S_from_G()
        if lats:
            cands = []          # (G corners, S corners or None) options, checked against the image below
            best = None
            for sg, g, cg in _window_candidates(lats[0], hsv, r):          # choose both plates TOGETHER
                total, s_pick = sg, None
                if len(lats) > 1:
                    gU, gV = (g["TR"] - g["TL"]) / 5, (g["BL"] - g["TL"]) / 4
                    prior = g["TL"] + ab[0] * gU + ab[1] * gV
                    ss, s_, cs = _window_candidates(lats[1], hsv, r, prior_tl=prior, hidden_right_col=True, top=1)[0]
                    if cs >= 6:
                        total, s_pick = sg + ss, (s_, cs)
                cands.append((total, g, cg, s_pick))
            cands.sort(key=lambda x: -x[0])
            info["_cands"] = cands[:4]
            # The scrap plate on its own (often all-empty, so its holes are easy to find), and the good plate placed
            # from it using the fixed layout of the two plates - for photos where the good plate is mostly capped.
            alt = []
            for lt in lats[::-1] if len(lats) > 1 else lats:
                for ss, s_, cs in _window_candidates(lt, hsv, r, hidden_right_col=True, top=2):
                    if cs >= 10:
                        alt.append({"S": s_})
                        alt.append({"G": _place_from(s_, "S", "G"), "S": s_})
            info["_alt"] = alt
            _, g, cg, s_pick = cands[0]
            corners["G"] = g
            info["fixtures"].append(("G", cg))
            if s_pick:
                corners["S"] = s_pick[0]
                info["fixtures"].append(("S", s_pick[1]))
    if "G" in corners or info.get("_alt"):
        # Self-check: try the straightening from both plates and from the good plate alone, for the best few block
        # choices; keep the one that puts the most grid circles on real holes in the standard view.
        options = []
        sets_all = []
        default_c = [(0, corners["G"], 0, (corners["S"], 0) if "S" in corners else None)] if "G" in corners else []
        for _, g, _, s_pick in info.pop("_cands", default_c):
            if s_pick:
                sets_all.append({"G": g, "S": s_pick[0]})
            sets_all.append({"G": g})
            sets_all.append({"G": g, "S": _place_from(g, "G", "S")})
        sets_all += info.pop("_alt", [])
        if True:
            for cs in sets_all:
                src = np.array([cs[fx][k] for fx in cs for k in ("TL", "TR", "BR", "BL")], float) / f
                dst = np.array([DEFAULT_CALIB[fx][k] for fx in cs for k in ("TL", "TR", "BR", "BL")], float)
                Hm, _ = cv2.findHomography(src, dst, 0)
                if Hm is None or not np.isfinite(Hm).all() or abs(np.linalg.det(Hm[:2, :2])) < 1e-6:
                    continue
                options.append((Hm, len(cs)))
        # plain resize of the whole photo as one more option: a wrong fit can never beat it
        S_ = np.array([[BASE_W / W0, 0, 0], [0, BASE_H / H0, 0], [0, 0, 1.0]])
        options.append((S_, 0))
        best = None
        for Hm, nfx in options:
            canon = cv2.warpPerspective(rgb, Hm, (BASE_W, BASE_H),
                                        flags=cv2.INTER_AREA if W0 > BASE_W else cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            out = Image.fromarray(canon)
            cal, _ = auto_align(out, DEFAULT_CALIB)               # fine-tune both grids in the standard view
            q = alignment_quality(out, cal)
            key = q["G"] + q["S"] + 0.5 * (nfx == 2) - 1.0 * _border_hits(out, cal)
            if best is None or key > best[0]:
                best = (key, out, cal, Hm, q)
        if best is not None:
            _, out, cal, Hm, q = best
            ok = (q["G"] + q["S"]) >= 20 and not np.allclose(Hm, S_)
            info.update(ok=ok, H=Hm.tolist(), quality=q)
            return out, cal, info
    out = img.convert("RGB").resize((BASE_W, BASE_H), Image.LANCZOS)
    return out, auto_align(out, DEFAULT_CALIB)[0], info


def needs_auto_fit(img: Image.Image, calib: dict = None) -> bool:
    """Photos that are not a 4:3 frame near 640x480, or where the standard grid finds few holes, get auto-fitted."""
    if abs(img.width / img.height - BASE_W / BASE_H) > 0.02 or not (560 <= img.width <= 720):
        return True
    q = alignment_quality(img, calib or DEFAULT_CALIB)
    return q["G"] < 0.4 * 30 and q["S"] < 0.4 * 25
