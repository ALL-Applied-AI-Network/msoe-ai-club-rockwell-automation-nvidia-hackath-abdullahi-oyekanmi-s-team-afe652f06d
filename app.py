"""
Vial Inspector - CSI Dataset 1 (final vial inspection) dashboard.

Run:  python -m streamlit run app.py
Needs vial_core.py and ui.py in the same folder. API key: set NVIDIA_API_KEY or paste it in the sidebar.
"""
import dataclasses
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import helpbot as hb
import ui
import vial_core as vc

st.set_page_config(page_title="Vial Inspector · CSI Dataset 1", page_icon="🧪", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


def html(s: str, where=st):
    where.markdown(s, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
helper_slot = st.sidebar.container()        # Vial Helper is drawn here at the end of the run (it needs the results)
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Mode", ["Local CV + NVIDIA vision model", "Local CV only (instant, no API)"], index=0)
use_vlm = mode.startswith("Local CV +")

api_key, model = "", vc.DEFAULT_MODEL
workers, timeout, skip_empty, strict = 3, 90, True, False
if use_vlm:
    api_key = st.sidebar.text_input("NVIDIA API key", value=os.environ.get("NVIDIA_API_KEY", ""), type="password")
    model = st.sidebar.text_input("Model", value=vc.DEFAULT_MODEL,
                                  help="Faster option to try: meta/llama-3.2-11b-vision-instruct. "
                                       "Check build.nvidia.com for the exact model IDs.")
    workers = st.sidebar.slider("Parallel requests", 1, 5, 3)
    timeout = st.sidebar.slider("Timeout per request (s)", 30, 180, 90)
    skip_empty = st.sidebar.checkbox("Skip the API for clearly empty holes", value=True,
                                     help="Dark, uniform empty holes are decided locally. Saves time and credits.")
    strict = st.sidebar.checkbox("Strict mode (flag more)", value=False,
                                 help="Asks the model twice per position (wide crop + tight crop) and trusts its "
                                      "self-check answers even when the cap looks seated. Catches more, but raises "
                                      "false alarms and doubles API calls. Measure both modes in 'Check accuracy'.")
change_aware = st.sidebar.checkbox("Change-aware (only re-inspect what changed)", value=True,
                                   help="With several frames of the same fixture, positions that look the same as in the "
                                        "previous frame (in time order) reuse its answers; only new or changed positions go "
                                        "to the model. Measured: 98.5% fewer model calls over 1,097 frames of one shift.")
auto_fill = st.sidebar.checkbox("Estimate the fill from the image when no count is typed", value=True,
                                help="The robot fills in a fixed order, so the last clearly capped vial shows how far it has "
                                     "got. Holes well past it are waiting (EMPTY); holes before it must hold a capped vial; "
                                     "the 2 positions right after it stay 'check by eye'. A typed robot count overrides it.")
require_cap = st.sidebar.checkbox("A 'pass' needs a yellow cap", value=True,
                                  help="If the model says pass but no yellow cap is visible, mark REVIEW. "
                                       "Turn off only if other cap colors are expected.")
hide_edge = st.sidebar.checkbox("S01–S05 hidden at the plate edge → NO VIEW", value=True,
                                help="The rightmost scrap column is half hidden in this camera view, so any verdict there "
                                     "is a guess. Report it as a blind spot instead of a fake PASS / REJECT.")
crop_side = st.sidebar.slider("Crop size around each position (px at 640x480)", 36, 90, 56)
auto_fit_on = st.sidebar.checkbox("🧭 Auto-fit photos of any size / framing", value=True,
                                  help="Finds both fixtures anywhere in the photo (any resolution, aspect ratio, zoom or "
                                       "small rotation) and straightens it into the standard camera view before "
                                       "inspecting. Standard 640x480 frames that already line up are left as they are.")


@st.cache_data(show_spinner="🧭 Finding the fixtures in this photo...", max_entries=64)
def _auto_fit_cached(raw_bytes):
    im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    out, cal, info = vc.auto_fit(im)
    return np.array(out), cal, {k: v for k, v in info.items() if k != "H"}


def fitted_image(raw_bytes):
    """The image the inspection actually uses: the photo itself, or its auto-fitted standard view."""
    im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if auto_fit_on and vc.needs_auto_fit(im):
        arr, cal, info = _auto_fit_cached(raw_bytes)
        return Image.fromarray(arr), cal, info
    return im, None, None

for _fx, _cs in vc.DEFAULT_CALIB.items():
    for _corner, (_x, _y) in _cs.items():
        st.session_state.setdefault(f"{_fx}{_corner}x", float(_x))
        st.session_state.setdefault(f"{_fx}{_corner}y", float(_y))


def _current_calib():
    return {fx: {c: (float(st.session_state[f"{fx}{c}x"]), float(st.session_state[f"{fx}{c}y"]))
                 for c in ("TL", "TR", "BR", "BL")} for fx in ("G", "S")}


def _set_calib(cal):
    for fx, cs in cal.items():
        for c, (x, y) in cs.items():
            st.session_state[f"{fx}{c}x"] = float(x)
            st.session_state[f"{fx}{c}y"] = float(y)


def do_auto_align(raw_bytes, name=None):
    im, fit_cal, fit_info = fitted_image(raw_bytes)
    start = st.session_state.setdefault("calib_of", {}).pop(name, None) or fit_cal or _current_calib()
    before = vc.alignment_quality(im, start)
    new, _ = vc.auto_align(im, start)
    if fit_info is None and min(vc.alignment_quality(im, new).values()) < 10 and auto_fit_on:
        # the small search did not find the fixtures: fall back to the full auto-fit (fixtures anywhere)
        st.session_state.setdefault("force_fit", set()).add(name)
    if fit_info is not None:
        st.session_state["calib_of"][name] = new            # a fitted photo keeps its own grid
    else:
        _set_calib(new)                  # becomes the grid for this image and for images not inspected yet
    st.session_state["align_msg"] = (before, vc.alignment_quality(im, new))


def do_reset_align(name=None):
    st.session_state.setdefault("calib_of", {}).pop(name, None)
    st.session_state.setdefault("force_fit", set()).discard(name)
    _set_calib(vc.DEFAULT_CALIB)
    st.session_state["align_msg"] = None


def remember_grid(prep_):
    """An inspected image keeps the grid it was inspected with, so flipping to another image (and re-aligning
    there) never wipes its results."""
    st.session_state.setdefault("calib_of", {})[prep_["name"]] = prep_["calib"]


with st.sidebar.expander("Calibration (advanced)"):
    st.caption("Centers of the 4 corner holes of each 6x5 grid, in 640x480 pixel space. "
               "Use Auto-align on the main page; edit here only for fine adjustments.")
    for fx, name in (("G", "Good fixture (left)"), ("S", "Scrap fixture (right)")):
        st.markdown(f"**{name}**")
        for corner in ("TL", "TR", "BR", "BL"):
            c1, c2 = st.columns(2)
            c1.number_input(f"{fx} {corner} x", step=1.0, key=f"{fx}{corner}x")
            c2.number_input(f"{fx} {corner} y", step=1.0, key=f"{fx}{corner}y")
calib = _current_calib()

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
MODEL_TAG = model.split("/")[-1] if use_vlm else ""
MODE_TAG = "strict" if strict else ("balanced" if use_vlm else "")
top_slot = st.empty()
html(ui.topbar(MODEL_TAG, live=use_vlm, mode=MODE_TAG), top_slot)
html(ui.hero())
pipe_slot = st.empty()


# --------------------------------------------------------------------------------------
# Digital twin (NVIDIA Isaac Sim): synthetic frames with exact labels, checked live by the same rules
# --------------------------------------------------------------------------------------
TWIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digital_twin", "samples")
TRUE_STATUS = {"pass": "PASS", "missing_cap": "REJECT", "crooked_cap": "REJECT", "no_vial": "EMPTY"}


def twin_section(key: str, real_img=None):
    st.markdown("#### Digital twin: synthetic test frames from NVIDIA Isaac Sim")
    st.caption("Per the lab rules, the real line images are only looked at, never trained on. These frames were rendered "
               "from the CSI digital twin's own CapSealCam (isaac/gen_synth.py): random fill level, missing caps, crooked "
               "caps, missed placements and lighting, with labels known exactly. We use them to stress-test our rules.")
    html(ui.twin_cards(vc.MEASURED["twin"]))
    lab_path = os.path.join(TWIN_DIR, "labels.json")
    if not os.path.exists(lab_path):
        st.info("Sample twin frames not found (digital_twin/samples).")
        return
    tl = json.load(open(lab_path))
    pick = st.selectbox("Synthetic frame", list(tl), key=f"twin_pick_{key}",
                        format_func=lambda n: f"{n}  ·  {tl[n]['placed_G']} vials placed  ·  light x{min(tl[n]['lights']):.2f}"
                                              f"  ·  {sum(v in ('missing_cap', 'crooked_cap') for v in tl[n]['labels'].values())} defects")
    info = tl[pick]
    img = Image.open(os.path.join(TWIN_DIR, pick)).convert("RGB")
    cen = {p: c for p, c in vc.all_centers(vc.DEFAULT_CALIB, img.size).items() if p[0] == "G"}
    rr = vc.hole_radius(img.size)
    truth = {p: vc.Result(pid=p, label=t, status=TRUE_STATUS[t]) for p, t in info["labels"].items()}
    use_count = st.toggle("Give the checks the robot's placement count (from the twin)", value=True, key=f"twin_cnt_{key}")
    res = {p: vc.decide(p, vc.local_features(vc.crop_square(img, *cen[p], 56), rr), None) for p in cen}
    if use_count:
        vc.apply_placement_count(res, {"G": info["placed_G"]})
        vc.apply_fill_pattern(res, {"G": info["placed_G"]})
    else:
        vc.apply_estimated_count(res, {"G": vc.estimate_placed(res)["G"]})
        vc.apply_fill_pattern(res, {})
    c1, c2 = st.columns(2)
    c1.image(vc.draw_overlay(img, cen, truth), caption="Ground truth from the twin (red = defect, green = good, grey = empty)",
             use_container_width=True)
    c2.image(vc.draw_overlay(img, cen, res), caption="Our local checks, run live on this frame (no API calls)",
             use_container_width=True)
    lab = info["labels"]
    miss = [p for p, t in lab.items() if t == "missing_cap" or info["notes"].get(p) == "missed_placement"]
    crook = [p for p, t in lab.items() if t == "crooked_cap"]
    good = [p for p, t in lab.items() if t == "pass"]
    empty = [p for p, t in lab.items() if t == "no_vial" and info["notes"].get(p) != "missed_placement"]
    escaped = [p for p in miss + crook if res[p].status in ("EMPTY", "PASS")]
    leans = [p for p in good if info["notes"].get(p, "").startswith("lean")]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Missing caps / missed placements flagged", f"{sum(res[p].status in ('REJECT', 'REVIEW', 'GAP') for p in miss)}/{len(miss)}")
    m2.metric("Crooked caps flagged or sent to the model", f"{sum(res[p].status != 'EMPTY' for p in crook)}/{len(crook)}",
              help="Locally a crooked cap that still covers the opening shows as CAP; the NVIDIA model judges the seating.")
    m3.metric("Good caps recognised", f"{sum(res[p].status == 'CAP' for p in good)}/{len(good)}",
              help="Leaning vials whose cap sits off the hole become one-click quick checks instead.")
    m4.metric("Empty holes called empty", f"{sum(res[p].status == 'EMPTY' for p in empty)}/{len(empty)}")
    if leans:
        st.caption(f"Leaning vials in this frame (good, tipped 3-12°): {len(leans)} - "
                   f"{sum(res[p].status == 'CAP' for p in leans)} pass the local check, "
                   f"{sum(vc.lean_candidate(res[p]) for p in leans)} are one-click quick checks, "
                   f"{sum(res[p].status == 'REJECT' for p in leans)} rejected.")
    if escaped:
        st.warning("Slipped through as empty: " + ", ".join(escaped) + " - an uncapped vial that looks exactly like an empty "
                   "hole. Only the robot's placement count can catch these, so give the app that count on the line.")
    if real_img is not None:
        with st.expander("Side by side with the real camera frame"):
            a, b = st.columns(2)
            a.image(real_img, caption="Real CSI camera frame", use_container_width=True)
            b.image(img, caption="Digital twin render (same camera, same lens)", use_container_width=True)



# --------------------------------------------------------------------------------------
# Vial Helper - questions about the website and the inspection (helpbot.py)
# --------------------------------------------------------------------------------------
_HELPER_DRAWN = []


def _help_pick(title):
    st.session_state["help_pending"] = title


TYPING_HTML = (
    "<style>.vh-typing{display:inline-flex;gap:5px;align-items:center;padding:9px 13px;border-radius:14px;"
    "background:rgba(34,211,238,.10);border:1px solid rgba(34,211,238,.25);font-size:12.5px;color:#9fb3c8}"
    ".vh-typing i{width:7px;height:7px;border-radius:50%;background:#22d3ee;display:inline-block;"
    "animation:vhb 1.1s infinite ease-in-out}.vh-typing i:nth-child(2){animation-delay:.15s}"
    ".vh-typing i:nth-child(3){animation-delay:.3s}@keyframes vhb{0%,80%,100%{transform:translateY(0);opacity:.35}"
    "40%{transform:translateY(-5px);opacity:1}}</style>"
    "<div class='vh-typing'>Vial Helper is typing <i></i><i></i><i></i></div>")


def _bubble(role):
    """Chat bubble when this Streamlit has st.chat_message, else a plain container."""
    if hasattr(st, "chat_message"):
        return st.chat_message(role, avatar="🙋" if role == "user" else "🤖")
    return st.container()


def _type_out(slot, text, max_seconds=2.2):
    """Show the answer word by word, like someone typing (whole answer in at most ~2 s)."""
    import time
    words = text.split(" ")
    step = max(1, len(words) // 60)                         # ~60 updates at most, however long the answer
    delay = min(0.035, max_seconds / max(1, len(words) / step))
    for i in range(step, len(words) + step, step):
        slot.markdown(" ".join(words[:i]) + " ▌")
        time.sleep(delay)
    slot.markdown(text)


def render_helper(res=None):
    import time
    if _HELPER_DRAWN:
        return
    _HELPER_DRAWN.append(1)
    hist = st.session_state.setdefault("help_hist", [])
    with helper_slot.expander("💬 Vial Helper - ask me anything", expanded=bool(hist)):
        st.caption("Answers come from checked notes about this website and, for a position like G15, from the current "
                   "inspection. If I don't know, I say so.")
        with st.form("help_form", clear_on_submit=True):
            q = st.text_input("Your question", placeholder="e.g. Why is G15 flagged?  ·  How do I align the circles?",
                              label_visibility="collapsed")
            asked = st.form_submit_button("Send ➤", use_container_width=True)
        use_model = False
        if api_key:
            use_model = st.toggle("🧠 NVIDIA model writes the reply (1 small API call)", value=True, key="help_model",
                                  help="The model only sees the checked answers and must stick to them; if it adds "
                                       "anything else or the call fails, the checked answer is shown.")
        else:
            st.caption("Add an NVIDIA API key in the sidebar to let the NVIDIA model write the replies; without it you "
                       "get the checked answers directly.")
        new_q = None
        if asked and q.strip():
            new_q = q.strip()
        elif st.session_state.get("help_pending"):
            new_q = st.session_state.pop("help_pending")
        if new_q:
            with _bubble("user"):
                st.markdown(new_q)
            with _bubble("assistant"):
                slot = st.empty()
                slot.markdown(TYPING_HTML, unsafe_allow_html=True)        # typing indicator while thinking
                t0 = time.time()
                a = hb.by_title(new_q) or hb.answer(new_q, res)
                if use_model and a.get("confident") and a.get("candidates"):
                    txt = hb.rephrase_with_model(new_q, a, api_key)
                    if txt:
                        a = dict(a, text=txt, source=a["source"] + " · written by the NVIDIA model")
                time.sleep(max(0.0, 0.7 - (time.time() - t0)))            # let the dots show for a moment
                _type_out(slot, a["text"])
                st.caption(f"Source: {a['source']}")
            hist.insert(0, (new_q, a))
            del hist[6:]
            shown = 1
        else:
            shown = 0
        if not hist:
            st.markdown("**Try:**")
            for t in ("How do I start the website?", "The circles are not on the holes - how do I align?",
                      "What do PASS / REJECT / REVIEW / GAP / EMPTY / CAP / NO VIEW mean?",
                      "Leaning vial vs crooked cap - why is it flagged?", "How accurate is it?"):
                st.button(t, key=f"help_try_{t}", on_click=_help_pick, args=(t,), use_container_width=True)
        for i, (qq, a) in enumerate(hist[:4]):
            if i >= shown:                                                # earlier messages: no animation
                with _bubble("user"):
                    st.markdown(qq)
                with _bubble("assistant"):
                    st.markdown(a["text"])
                    st.caption(f"Source: {a['source']}")
            if a.get("related"):
                st.caption("Not what you meant?" if a.get("confident") else "Try one of these:")
                for j, t in enumerate(a["related"][:3]):
                    st.button(t, key=f"help_rel_{i}_{j}_{len(hist)}", on_click=_help_pick, args=(t,),
                              use_container_width=True)
            st.divider()
        if hist and st.button("Clear chat", key="help_clear"):
            st.session_state["help_hist"] = []
            st.rerun()


# --------------------------------------------------------------------------------------
# Image queue: upload many, flip between them, inspect one or all, start over
# --------------------------------------------------------------------------------------
st.session_state.setdefault("upl_n", 0)


def new_inspection():
    st.session_state["upl_n"] += 1
    for k in ("img_sel", "align_msg"):
        st.session_state.pop(k, None)


files = st.file_uploader("Drop one or more overhead camera images (you can add more at any time)",
                         type=["png", "jpg", "jpeg"], accept_multiple_files=True,
                         key=f"uploader_{st.session_state['upl_n']}")
if not files:
    html(ui.pipeline(0, use_vlm), pipe_slot)
    st.info("Upload one or more images from Dataset 1 to begin. Images in the app folder's demo_images/ "
            "already have saved model answers, so they cost no API calls.")
    with st.expander("🧪 Digital twin: synthetic test frames from NVIDIA Isaac Sim", expanded=False):
        twin_section("landing")
    render_helper(None)
    st.stop()

RAWS = {}
for f in files:
    RAWS.setdefault(f.name, f.getvalue())
names = list(RAWS)
cache = st.session_state.setdefault("vlm_cache", {})
DEFAULT_GRID = calib == {k: {c: tuple(map(float, v)) for c, v in d.items()} for k, d in vc.DEFAULT_CALIB.items()}


@st.cache_data(show_spinner=False)
def _load_saved(path, mtime):
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return {}


_saved_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_cache.json")
SAVED = _load_saved(_saved_path, os.path.getmtime(_saved_path)) if os.path.exists(_saved_path) else {}
PH = hashlib.md5(vc.PROMPT.encode()).hexdigest()[:8]
VH = {hashlib.md5(t.encode()).hexdigest()[:8] for t in (vc.VERIFY_CAP_PROMPT, vc.VERIFY_VIAL_PROMPT)}


def frame_label(name):
    p = name.split("_")
    if len(p) >= 4 and p[2].isdigit() and len(p[2]) == 8 and p[3][:6].isdigit():
        return f"{p[2][:4]}-{p[2][4:6]}-{p[2][6:8]} {p[3][:2]}:{p[3][2:4]}:{p[3][4:6]}"
    return name


def prepare(name):
    """Everything local for one image (grid, crops, rules), cached per image + settings."""
    raw0 = RAWS[name]
    if name in st.session_state.get("force_fit", set()) and auto_fit_on:
        arr, fcal, fit = _auto_fit_cached(raw0)
        img0 = Image.fromarray(arr)
    else:
        img0, fcal, fit = fitted_image(raw0)
    calib = st.session_state.setdefault("calib_of", {}).get(name) or fcal or _current_calib()   # per-image grid
    key = (name, hashlib.md5(RAWS[name]).hexdigest(), json.dumps(calib, sort_keys=True), crop_side, bool(fit),
           hide_edge, skip_empty, strict, model)
    store = st.session_state.setdefault("prep_store", {})
    if key in store:
        return store[key]
    raw = RAWS[name]
    img = img0
    centers = vc.all_centers(calib, img.size)
    r = vc.hole_radius(img.size)
    side = crop_side * img.width / vc.BASE_W
    crops = {p: vc.crop_square(img, cx, cy, side) for p, (cx, cy) in centers.items()}
    feats = {p: vc.local_features(crops[p], r) for p in centers}
    guess = {p: vc.local_guess(f) for p, f in feats.items()}
    geom_key = hashlib.md5(raw + json.dumps([calib, crop_side, bool(fit)], sort_keys=True).encode()
                           + vc.PROMPT.encode()).hexdigest()[:12]
    edge_hidden = set(vc.HIDDEN_POSITIONS) if hide_edge else set()
    blocked = set(vc.blocked_positions(img, centers)) - edge_hidden
    hidden = edge_hidden | blocked
    to_send = [p for p in centers if p not in hidden and not (skip_empty and guess[p] == "empty")]
    tags = ["wide", "tight"] if strict else ["wide"]
    prep = dict(name=name, raw=raw, img=img, centers=centers, r=r, crops=crops, feats=feats, guess=guess, calib=calib, fit=fit,
                geom_key=geom_key, blocked=blocked, hidden=hidden, to_send=to_send, tags=tags,
                tasks=[(p, t) for p in to_send for t in tags], reused=0, carried=0,
                sig=vc.position_signatures(img, centers))
    # reuse answers saved by batch_eval.py for this exact image (default grid only) -> no API calls
    _default_here = calib == {k: {c: tuple(map(float, v)) for c, v in d.items()} for k, d in vc.DEFAULT_CALIB.items()}
    if use_vlm and not strict and _default_here and not fit and SAVED:     # saved answers only fit the original photo
        for p in centers:
            k = f"{name}|{p}|{model}|{PH}|{crop_side}"
            if k in SAVED and (geom_key, p, "wide", model) not in cache:
                cache[(geom_key, p, "wide", model)] = SAVED[k]
                prep["reused"] += 1
            for h in VH:
                k2 = f"{name}|{p}|{model}|verify-{h}|{crop_side}"
                if k2 in SAVED and (geom_key, p, "second", model) not in cache:
                    cache[(geom_key, p, "second", model)] = SAVED[k2]
    store[key] = prep
    return prep


def pending_calls(prep):
    return [k for k in prep["tasks"] if (prep["geom_key"], k[0], k[1], model) not in cache]


def image_for(prep, pid, tag):
    base = prep["crops"][pid] if tag == "wide" else vc.mask_neighbors(prep["crops"][pid], prep["r"])
    return vc.crop_with_ring(base, prep["r"])


def run_model(prep, bar, done_before=0, total=None):
    """Ask the NVIDIA model about every uncertain position of one image (+ second looks). Progress on `bar`."""
    gk, pend = prep["geom_key"], pending_calls(prep)
    remember_grid(prep)
    total = total or max(1, len(pend))

    def work(task):
        pid, tag = task
        return task, vc.call_vlm(image_for(prep, pid, tag), api_key, model=model, timeout=timeout)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, k) for k in pend]), 1):
            (pid, tag), out = fut.result()
            if "error" not in out:
                cache[(gk, pid, tag, model)] = out
            else:
                st.session_state.setdefault("last_errors", {})[(gk, pid, tag, model)] = out
            bar.progress(min(1.0, (done_before + i) / total),
                         text=f"{frame_label(prep['name'])} · {done_before + i}/{total} model calls")
    # One automatic second try for calls that failed (a single timeout like G12 on Cap_seal2 otherwise leaves a
    # REVIEW with no model answer). Only when failures are few: if many calls fail the API is probably down, and
    # retrying everything would just double the wait.
    errs = st.session_state.setdefault("last_errors", {})
    failed = [k for k in pend if (gk, k[0], k[1], model) in errs and (gk, k[0], k[1], model) not in cache]
    if failed and len(failed) <= max(2, len(pend) // 4):
        bar.progress(1.0, text=f"{frame_label(prep['name'])} · retrying {len(failed)} failed model call(s)")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(work, k) for k in failed]):
                (pid, tag), out = fut.result()
                if "error" not in out:
                    cache[(gk, pid, tag, model)] = out
                    errs.pop((gk, pid, tag, model), None)
    looks = [p for p in prep["to_send"] if (gk, p, "wide", model) in cache
             and vc.second_look_prompt(prep["feats"][p], cache[(gk, p, "wide", model)])
             and (gk, p, "second", model) not in cache]

    def look(pid):
        return pid, vc.call_vlm(vc.crop_with_ring(prep["crops"][pid], prep["r"]), api_key, model=model, timeout=timeout,
                                prompt=vc.second_look_prompt(prep["feats"][pid], cache[(gk, pid, "wide", model)]))

    if looks:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(look, p) for p in looks]):
                pid, out = fut.result()
                if "error" not in out:
                    cache[(gk, pid, "second", model)] = out
    return len(pend)


def placed_for(name):
    g, s = st.session_state.get(f"pg_{name}"), st.session_state.get(f"ps_{name}")
    return {"G": None if g is None else int(g), "S": None if s is None else int(s)}


def results_for(prep):
    """Final verdict per position, or None while model answers are still missing."""
    gk, feats = prep["geom_key"], prep["feats"]

    def unseen(p):
        return vc.blocked_result(p, feats[p]) if p in prep["blocked"] else vc.hidden_result(p, feats[p])

    if not use_vlm:
        if (prep["name"], gk) not in st.session_state.setdefault("local_ok", set()):
            return None                      # local mode: the operator checks the grid first, then inspects
        res = {p: (unseen(p) if p in prep["hidden"] else vc.decide(p, feats[p], None, require_cap))
               for p in prep["centers"]}
    else:
        errs = st.session_state.get("last_errors", {})
        opinion = {}
        for (pid, tag) in prep["tasks"]:
            k = (gk, pid, tag, model)
            if k in cache:
                opinion[(pid, tag)] = cache[k]
            elif k in errs:
                opinion[(pid, tag)] = errs[k]            # errors become REVIEW
        if not all(t in opinion for t in prep["tasks"]):
            return None
        res = {}
        for p in prep["centers"]:
            if p in prep["hidden"]:
                res[p] = unseen(p)
            elif p in prep["to_send"] and not strict:
                res[p] = vc.decide(p, feats[p], opinion[(p, "wide")], require_cap, strict,
                                   second=cache.get((gk, p, "second", model)))
            elif p in prep["to_send"]:
                res[p] = vc.decide_multi(p, feats[p], [opinion[(p, t)] for t in prep["tags"]], prep["tags"],
                                         require_cap, strict)
            else:
                res[p] = vc.decide(p, feats[p], None, require_cap)
    placed = placed_for(prep["name"])
    vc.apply_placement_count(res, placed)
    if auto_fill:
        est = vc.estimate_placed(res)
        vc.apply_estimated_count(res, {fx: n for fx, n in est.items() if placed[fx] is None})
    vc.apply_fill_pattern(res, {k: (v or 0) for k, v in placed.items()})
    vc.apply_operator(res, st.session_state.setdefault("operator", {}).get(prep["name"], {}))
    return res


def carry_over():
    """Change-aware: for frames in time order, unchanged positions reuse the previous frame's model answers."""
    if not (use_vlm and change_aware and not strict):
        return
    import datetime as _dt

    def ts(n):
        p = n.split("_")
        try:
            return _dt.datetime.strptime(p[2] + p[3][:6], "%Y%m%d%H%M%S")
        except Exception:  # noqa: BLE001
            return None

    order = sorted(names)
    for i in range(1, len(order)):
        prev, cur = preps[order[i - 1]], preps[order[i]]
        t0, t1 = ts(prev["name"]), ts(cur["name"])
        if t0 is None or t1 is None or not (0 <= (t1 - t0).total_seconds() <= 600):
            continue                          # only consecutive frames of the same run (<= 10 min apart)
        changed = set(vc.changed_positions(prev["sig"], cur["sig"]))
        n = 0
        for p in cur["to_send"]:
            kc = (cur["geom_key"], p, "wide", model)
            kp = (prev["geom_key"], p, "wide", model)
            if kc in cache or p in changed or kp not in cache:
                continue
            cache[kc] = dict(cache[kp])
            k2 = (prev["geom_key"], p, "second", model)
            if k2 in cache:
                cache[(cur["geom_key"], p, "second", model)] = cache[k2]
            n += 1
        cur["carried"] += n


# ---- queue bar: previous / image picker / next / new inspection ----
if st.session_state.get("img_sel") not in names:
    st.session_state["img_sel"] = names[-1]


def step_image(delta, options):
    i = options.index(st.session_state["img_sel"]) if st.session_state["img_sel"] in options else 0
    st.session_state["img_sel"] = options[(i + delta) % len(options)]


preps = {n: prepare(n) for n in names}
carry_over()
done = {n: (results_for(preps[n]) is not None) for n in names}
qa, qb, qc, qd = st.columns([1, 6, 1, 2.2])
qa.button("◀", use_container_width=True, on_click=step_image, args=(-1, names), disabled=len(names) < 2,
          help="Previous image")
qb.selectbox("Image", names, key="img_sel", label_visibility="collapsed",
             format_func=lambda n: f"{'✅' if done[n] else '⏳'}  {frame_label(n)}   ·   {n}")
qc.button("▶", use_container_width=True, on_click=step_image, args=(1, names), disabled=len(names) < 2,
          help="Next image")
qd.button("➕ New inspection", use_container_width=True, on_click=new_inspection,
          help="Clear all images and start over with new photos (saved model answers are kept).")
chosen = st.session_state["img_sel"]
prep = preps[chosen]
img, centers, r, crops, feats = prep["img"], prep["centers"], prep["r"], prep["crops"], prep["feats"]
geom_key, blocked, hidden, to_send = prep["geom_key"], prep["blocked"], prep["hidden"], prep["to_send"]
raw = prep["raw"]
html(ui.topbar(MODEL_TAG, frame=frame_label(chosen), live=use_vlm, mode=MODE_TAG), top_slot)

pa, pb, pc_ = st.columns([2.2, 1, 1])
pa.markdown(f"**{len(names)} image{'s' if len(names) != 1 else ''} in this inspection** · "
            f"{sum(done.values())} inspected · {len(names) - sum(done.values())} waiting<br>"
            "<span style='color:#8b9ab4;font-size:12.5px'>Vials placed so far for <b>this</b> image (from the robot / "
            "PLC) - leave blank if unknown.</span>", unsafe_allow_html=True)
pb.number_input("Good placed", min_value=0, max_value=30, value=None, step=1, placeholder="unknown",
                key=f"pg_{chosen}", help="With a count, a no-cap hole beyond it is simply waiting (EMPTY) and one "
                                         "within it is a defect (missing cap or missed placement).")
pc_.number_input("Scrap placed", min_value=0, max_value=30, value=None, step=1, placeholder="unknown",
                 key=f"ps_{chosen}")
_placed = placed_for(chosen)

if prep.get("fit"):
    _fi = prep["fit"]
    if _fi.get("ok"):
        _w, _h = _fi["source_size"]
        _found = ", ".join(f"{'good' if fx == 'G' else 'scrap'} fixture ({n} holes found)" for fx, n in _fi["fixtures"])
        st.info(f"🧭 Auto-fit: this {_w}x{_h} photo was straightened and scaled into the standard camera view before "
                f"inspection - {_found}. Grid on holes: good {_fi['quality']['G']}/30, scrap {_fi['quality']['S']}/30. "
                "Check that the circles sit on the holes; 🎯 Auto-align fine-tunes, ↺ Reset goes back.")
    else:
        st.warning("🧭 Auto-fit could not find the fixture hole pattern in this photo - it was only resized. Check the "
                   "circles carefully, or use a sharper / less cropped photo.")
elif img.width and abs(img.width / img.height - vc.BASE_W / vc.BASE_H) > 0.05:
    st.warning(f"Image is {img.width}x{img.height}; the default calibration assumes a 4:3 camera frame (640x480). "
               "Use the raw camera frame, not a screenshot, and check that the circles sit on the holes.")
if prep["carried"]:
    st.caption(f"🔁 Change-aware: {prep['carried']} position(s) look exactly as in the previous frame, so their answers were "
               "reused - only new or changed positions go to the model.")
if prep["reused"]:
    st.caption(f"♻️ Reused {prep['reused']} saved model answers for this image from the batch test (no API calls).")
if blocked:
    st.warning(f"🦾 The robot arm covers {len(blocked)} position(s) in this frame "
               f"({', '.join(sorted(blocked))}). They are marked NO VIEW, not judged. "
               "For a full inspection use a frame where the arm is out of the way.")

# ---- inspect all ----
if use_vlm and len(names) > 1 and all(done.values()):
    st.success(f"✅ All {len(names)} images are inspected. Flip with ◀ ▶, or open 📚 All images below to see them "
               "side by side.")
elif not use_vlm and len(names) > 1 and not all(done.values()):
    waiting = [n for n in names if not done[n]]
    ia, ib = st.columns([2, 3])
    if ia.button(f"⏩ Inspect all {len(waiting)} waiting image{'s' if len(waiting) != 1 else ''} (instant, no API)",
                 type="primary", use_container_width=True):
        for n in waiting:
            st.session_state.setdefault("local_ok", set()).add((n, preps[n]["geom_key"]))
            remember_grid(preps[n])
        st.rerun()
    ib.caption("Uses the grid shown below for all images - check it first (🎯 Auto-align if the circles are off).")
elif use_vlm and len(names) > 1:
    waiting = [n for n in names if not done[n]]
    n_calls = sum(len(pending_calls(preps[n])) for n in waiting)
    ia, ib = st.columns([2, 3])
    if not api_key:
        ib.warning("Add your NVIDIA API key in the sidebar to inspect the waiting images.")
    if ia.button(f"⏩ Inspect all {len(waiting)} waiting image{'s' if len(waiting) != 1 else ''} (up to {n_calls} model calls)",
                 type="primary", use_container_width=True, disabled=not (waiting and api_key)):
        html(ui.pipeline(2, use_vlm), pipe_slot)
        bar = st.progress(0.0, text="Starting...")
        sofar = 0
        for n in sorted(waiting):
            carry_over()                       # earlier frames' answers flow into later, unchanged positions
            sofar += run_model(preps[n], bar, sofar, max(1, n_calls))
        bar.empty()
        st.rerun()
    ib.caption("Runs every waiting image in this inspection one after another. Uses the grid shown below for all "
               "images - check it first. Images with saved answers cost no calls.")

align_q = vc.alignment_quality(img, prep["calib"])
results = None
if not use_vlm:
    local_done = (chosen, geom_key) in st.session_state.setdefault("local_ok", set())
    with st.expander("Step 1 · Align the grid and inspect", expanded=not local_done):
        la, ra = st.columns([3, 2])
        la.image(vc.draw_overlay(img, centers), caption="Every circle should sit on a large hole.", use_container_width=True)
        with ra:
            html(ui.align_gauges(align_q, {"G": 30, "S": 25}))
            b1, b2 = st.columns(2)
            b1.button("🎯 Auto-align", use_container_width=True, on_click=do_auto_align, args=(raw, chosen), key="aa_local",
                      help="Find the round holes in this image and snap both 6x5 grids onto them "
                           "(fixes camera shifts of up to ~40 px and a few degrees).")
            b2.button("↺ Reset", use_container_width=True, on_click=do_reset_align, args=(chosen,), key="ar_local",
                      help="Back to the default calibration for the CSI camera.")
            msg = st.session_state.get("align_msg")
            if msg:
                (bq, aq) = msg
                st.caption(f"Auto-align: Good {bq['G']} → {aq['G']}/30 holes matched, Scrap {min(bq['S'], 25)} → "
                           f"{min(aq['S'], 25)}/25.")
            elif (align_q["G"] < 0.65 * 30 or align_q["S"] < 0.65 * 25) and blocked:
                st.info("Low hole match because the robot arm covers part of the fixture - that is expected here.")
            elif align_q["G"] < 0.65 * 30 or align_q["S"] < 0.65 * 25:
                st.warning("The grid may be off the holes - try 🎯 Auto-align.")
            aligned = st.checkbox("Alignment looks correct", value=local_done, key=f"aligned_local_{chosen}")
            if st.button("▶ Inspect this image (instant, no API)", type="primary", use_container_width=True,
                         disabled=not aligned or local_done, key="inspect_local"):
                st.session_state["local_ok"].add((chosen, geom_key))
                remember_grid(prep)
                st.rerun()
            if local_done:
                st.success("Inspected with the local checks. Moved the grid? Inspect again after re-aligning.")
            else:
                st.caption("Check that every circle sits on a hole, tick the box, then inspect.")
    results = results_for(prep)
else:
    n_pending = len(pending_calls(prep))
    with st.expander("Step 1 · Align the grid and run", expanded=n_pending > 0):
        la, ra = st.columns([3, 2])
        la.image(vc.draw_overlay(img, centers), caption="Every circle should sit on a large hole.", use_container_width=True)
        with ra:
            html(ui.align_gauges(align_q, {"G": 30, "S": 25}))
            b1, b2 = st.columns(2)
            b1.button("🎯 Auto-align", use_container_width=True, on_click=do_auto_align, args=(raw, chosen),
                      help="Find the round holes in this image and snap both 6x5 grids onto them "
                           "(fixes camera shifts of up to ~40 px and a few degrees).")
            b2.button("↺ Reset", use_container_width=True, on_click=do_reset_align, args=(chosen,),
                      help="Back to the default calibration for the CSI camera.")
            msg = st.session_state.get("align_msg")
            if msg:
                (bq, aq) = msg
                st.caption(f"Auto-align: Good {bq['G']} → {aq['G']}/30 holes matched, Scrap {min(bq['S'], 25)} → "
                           f"{min(aq['S'], 25)}/25. Saved answers only apply to the default grid, so a moved grid "
                           "needs new model calls.")
            elif (align_q["G"] < 0.65 * 30 or align_q["S"] < 0.65 * 25) and blocked:
                st.info("Low hole match because the robot arm covers part of the fixture - that is expected here.")
            elif align_q["G"] < 0.65 * 30 or align_q["S"] < 0.65 * 25:
                st.warning("The grid may be off the holes - try 🎯 Auto-align.")
            aligned = st.checkbox("Alignment looks correct", value=n_pending == 0, key=f"aligned_{chosen}")
            if st.button(f"▶ Inspect this image ({n_pending} model calls)", type="primary", use_container_width=True,
                         disabled=not (aligned and api_key) or n_pending == 0):
                html(ui.pipeline(2, use_vlm), pipe_slot)
                bar = st.progress(0.0, text="Starting...")
                run_model(prep, bar)
                bar.empty()
                st.rerun()
            if not api_key:
                st.warning("Add your NVIDIA API key in the sidebar.")
            elif n_pending == 0:
                st.success("All positions inspected for this image, model and settings.")
            else:
                st.caption(f"{len(to_send)} of 60 positions need the model; {60 - len(to_send) - len(hidden)} were "
                           f"decided locally{f' and {len(hidden)} are hidden from the camera' if hidden else ''}.")
    results = results_for(prep)


def session_summary():
    rows = []
    for n in names:
        res = results_for(preps[n]) if done[n] else None
        c = {}
        if res:
            for x in res.values():
                c[x.status] = c.get(x.status, 0) + 1
        pl = placed_for(n)
        rows.append({"": "👉" if n == chosen else "", "frame": frame_label(n),
                     "status": "inspected" if res else "waiting",
                     "REJECT": c.get("REJECT", 0) if res else None, "REVIEW": c.get("REVIEW", 0) if res else None,
                     "GAP": c.get("GAP", 0) if res else None, "PASS": c.get("PASS", c.get("CAP", 0)) if res else None,
                     "placed (G/S)": f"{'-' if pl['G'] is None else pl['G']} / {'-' if pl['S'] is None else pl['S']}",
                     "file": n})
    return pd.DataFrame(rows)


def gallery():
    """All inspected images side by side, each with its verdict."""
    shown = [n for n in names if done[n]]
    if not shown:
        st.caption("No image inspected yet.")
        return
    for i in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, n in zip(cols, shown[i:i + 2]):
            pr = preps[n]
            res = results_for(pr)
            c = {}
            for x in res.values():
                c[x.status] = c.get(x.status, 0) + 1
            with col.container(border=True):
                st.image(vc.draw_overlay(pr["img"], pr["centers"], res, scale=1.3), use_container_width=True)
                html(ui.verdict(c, local_only=not use_vlm))
                st.caption(f"{frame_label(n)} · {n}" + ("   👈 shown above" if n == chosen else ""))


def _b64png(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return __import__("base64").b64encode(buf.getvalue()).decode()


def report_html(which):
    """Self-contained inspection record for the given images (only inspected ones)."""
    secs = []
    for n in which:
        pr = preps[n]
        res = results_for(pr)
        if res is None:
            continue
        c = {}
        for x in res.values():
            c[x.status] = c.get(x.status, 0) + 1
        flagged = sorted([p for p, x in res.items() if x.status in ("REJECT", "REVIEW", "GAP")],
                         key=lambda p: ({"REJECT": 0, "GAP": 1, "REVIEW": 2}[res[p].status], p))
        secs.append(dict(
            name=n, frame=frame_label(n), counts=c, msg=vc.line_actions(res, placed_for(n), frame=frame_label(n)),
            overlay_b64=_b64png(vc.draw_overlay(pr["img"], pr["centers"], res, scale=1.5)),
            flagged=[(p, res[p].status, res[p].label, res[p].reason,
                      _b64png(vc.crop_with_ring(pr["crops"][p], pr["r"], color=vc.STATUS_COLOR.get(res[p].status))))
                     for p in flagged],
            rows=[(p, x.status, x.label, x.source, x.reason) for p, x in sorted(res.items())]))
    meta = {"Generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"), "Model": model if use_vlm else "local rules only",
            "Mode": "strict" if strict else ("balanced" if use_vlm else "local"),
            "Grid": "default CSI calibration" if DEFAULT_GRID else "adjusted / auto-aligned",
            "Placement counts": ", ".join(f"{frame_label(n)}: G={placed_for(n)['G']}" for n in which)}
    return ui.build_report(secs, meta)


if len(names) > 1:
    with st.expander(f"📚 All images in this inspection ({sum(done.values())}/{len(names)} inspected) - side by side",
                     expanded=all(done.values())):
        g_tab, s_tab = st.tabs(["🖼 Side by side", "📋 Summary table"])
        with g_tab:
            gallery()
        with s_tab:
            summ = session_summary()
            st.dataframe(summ, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download session summary (CSV)", summ.to_csv(index=False),
                               file_name="inspection_session.csv", mime="text/csv")
            if any(done.values()):
                sk = f"session_{sorted(n for n in names if done[n])}"
                if st.session_state.get("session_report_key") != sk:
                    if st.button("📄 Build session report (all inspected images)"):
                        st.session_state["session_report"] = report_html(sorted(names))
                        st.session_state["session_report_key"] = sk
                        st.rerun()
                else:
                    st.download_button("⬇ Download session report (HTML)", st.session_state["session_report"],
                                       file_name="inspection_session_report.html", mime="text/html", type="primary")

if results is None:
    html(ui.pipeline(1, use_vlm), pipe_slot)
    render_helper(None)
    st.stop()

html(ui.pipeline(4, use_vlm), pipe_slot)

# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
counts = pd.Series([r_.status for r_ in results.values()]).value_counts().to_dict()
statuses = {p: r_.status for p, r_ in results.items()}
bad = [p for p, r_ in results.items() if r_.status in ("REJECT", "REVIEW", "GAP")]
rank = {"REJECT": 0, "GAP": 1, "REVIEW": 2}
lean_q = [p for p in bad if vc.lean_candidate(results[p])]
next_q = [p for p in bad if vc.next_in_line_candidate(results[p])]
bad.sort(key=lambda p: (rank[results[p].status], p in lean_q or p in next_q, p))   # one-click checks last
order = (["REJECT", "REVIEW", "GAP", "PASS" if use_vlm else "CAP"] + (["PASS"] if not use_vlm and "PASS" in counts else [])
         + ["EMPTY"] + (["NO VIEW"] if hidden else []))

left, right = st.columns([3, 2], gap="medium")
with left:
    st.image(vc.draw_overlay(img, centers, results), use_container_width=True)
    lat = [r_.latency_ms for r_ in results.values() if r_.latency_ms]
    if lat:
        st.caption(f"⏱ {sum(lat) / len(lat) / 1000:.1f} s model time per position · {len(lat)} positions sent to the "
                   f"model · {'strict' if strict else 'balanced'} mode")
    rk = f"report_{geom_key}_{_placed}"
    if st.session_state.get("report_key") != rk:
        if st.button("📄 Build inspection report (this image)", use_container_width=True):
            st.session_state["report_html"], st.session_state["report_key"] = report_html([chosen]), rk
            st.rerun()
    else:
        st.download_button("⬇ Download inspection report (HTML)", st.session_state["report_html"],
                           file_name=f"inspection_{chosen.rsplit('.', 1)[0]}.html", mime="text/html",
                           use_container_width=True, type="primary")
    if len(names) > 1:
        n1, n2 = st.columns(2)
        n1.button("◀ Previous image", use_container_width=True, on_click=step_image, args=(-1, names), key="prev2")
        n2.button("Next image ▶", use_container_width=True, on_click=step_image, args=(1, names), key="next2")
with right:
    html(ui.verdict(counts, local_only=not use_vlm))
    html(ui.metric_cards(counts, order, len(results)))
    _est = vc.estimate_placed(results)
    _marker = {fx: (_placed[fx] if _placed[fx] is not None else
                    (_est[fx] if auto_fill and fx in vc.ESTIMATE_FIXTURES else None)) for fx in ("G", "S")}
    html(ui.fill_strips(statuses, _marker))
    if auto_fill and (_placed["G"] is None or _placed["S"] is None):
        st.caption("Fill estimated from the image (last seated cap): "
                   + " · ".join(("Scrap: not estimated (scrap vials are often uncapped)" if fx == "S" else
                                 f"Good up to G{_est[fx]:02d}" if _est[fx] else "Good: none yet")
                                for fx in ("G", "S") if _placed[fx] is None)
                   + ". Type the robot's count above to override.")
    html(ui.flagged_list([(p, results[p].status, results[p].reason) for p in bad[:8]]))
    line_msg = vc.line_actions(results, _placed, frame=frame_label(chosen))
    line_msg["fill_estimated_from_image"] = {fx: _est[fx] for fx in ("G", "S") if _placed[fx] is None} if auto_fill else {}
    html(ui.line_panel(line_msg))
    st.download_button("⬇ Line message (JSON for robot / PLC)", json.dumps(line_msg, indent=2),
                       file_name=f"line_message_{chosen.rsplit('.', 1)[0]}.json", mime="application/json",
                       use_container_width=True)

t_bad, t_any, t_map, t_all, t_ev, t_twin, t_eval = st.tabs(
    [f"🚨 Needs attention ({len(bad)})", "🔎 Review every vial", "🧭 Fixture map", "📋 All positions", "📈 Evidence",
     "🧪 Digital twin", "🎯 Check accuracy"])

ICON = {"PASS": "✅", "CAP": "🟦", "REJECT": "❌", "REVIEW": "🟠", "GAP": "🟣", "EMPTY": "⚪", "NO VIEW": "⬛"}
REV = st.session_state.setdefault("reviewed", {})
NOTES = st.session_state.setdefault("notes", {})
POS_ORDER = [f"{fx}{i:02d}" for fx in ("G", "S") for i in range(1, 31)]


def _rev_key():
    return f"rev_pick_{chosen}"


def rev_step(delta):
    opts = [p for p in POS_ORDER if p in results]
    cur = st.session_state.get(_rev_key(), opts[0])
    st.session_state[_rev_key()] = opts[(opts.index(cur) + delta) % len(opts)] if cur in opts else opts[0]


def rev_next_unconfirmed():
    """Next position nobody has confirmed yet - flagged ones first, then the rest; then the next image."""
    done_here = REV.get(chosen, set())
    todo = [p for p in POS_ORDER if p in results and p not in done_here and results[p].status != "NO VIEW"]
    todo.sort(key=lambda p: (results[p].status not in ("REJECT", "REVIEW", "GAP"), POS_ORDER.index(p)))
    if todo:
        st.session_state[_rev_key()] = todo[0]
        return
    for n in names:                                   # this image is done: jump to the next image with work left
        if n != chosen and done[n] and len(REV.get(n, set())) < 60:
            st.session_state["img_sel"] = n
            return


def rev_set_class(pid):
    val = st.session_state[f"rev_cls_{chosen}_{pid}"]
    ops = st.session_state.setdefault("operator", {}).setdefault(chosen, {})
    if val == st.session_state.get(f"rev_sys_{chosen}_{pid}"):
        ops.pop(pid, None)                          # back to the system's own verdict
    elif val != "needs_review":
        ops[pid] = f"label:{val}"
    REV.setdefault(chosen, set()).add(pid)          # choosing a class counts as reviewing it


def rev_toggle(pid):
    s_ = REV.setdefault(chosen, set())
    (s_.add if st.session_state[f"rev_ok_{chosen}_{pid}"] else s_.discard)(pid)


def rev_note(pid):
    NOTES.setdefault(chosen, {})[pid] = st.session_state[f"rev_note_{chosen}_{pid}"]


def rev_confirm_unflagged():
    s_ = REV.setdefault(chosen, set())
    s_.update(p for p, x in results.items() if x.status in ("PASS", "CAP", "EMPTY"))


with t_any:
    reviewed_here = REV.get(chosen, set())
    seen = [p for p in results if results[p].status != "NO VIEW"]
    n_rev_all = sum(len(REV.get(n, set())) for n in names if done[n])
    n_all = sum(sum(1 for x in results_for(preps[n]).values() if x.status != "NO VIEW") for n in names if done[n])
    t1, t2, t3, t4 = st.columns([1, 1, 1.3, 2.4])
    t1.button("← Previous", use_container_width=True, on_click=rev_step, args=(-1,), key="rev_prev")
    t2.button("Next →", use_container_width=True, on_click=rev_step, args=(1,), key="rev_next")
    t3.button("⏭ Next unconfirmed", use_container_width=True, on_click=rev_next_unconfirmed, key="rev_nextun",
              help="Flagged positions first, then the rest; moves to the next image when this one is done.")
    t4.markdown(f"**{n_rev_all} / {n_all}** positions reviewed in this inspection · "
                f"**{len(reviewed_here & set(seen))}/{len(seen)}** in this image")
    st.progress(min(1.0, len(reviewed_here & set(seen)) / max(1, len(seen))))

    opts = [p for p in POS_ORDER if p in results]
    if st.session_state.get(_rev_key()) not in opts:
        st.session_state[_rev_key()] = next((p for p in opts if results[p].status in ("REJECT", "REVIEW", "GAP")), opts[0])
    pick = st.session_state[_rev_key()]
    classes = {p: vc.system_class(results[p]) for p in results}
    left_, right_ = st.columns([1.7, 1], gap="medium")
    with left_:
        st.image(vc.draw_review_overlay(img, centers, classes, reviewed_here, selected=pick, scale=2.0),
                 use_container_width=True)
        html('<div style="font-size:12.5px;color:#a8b3c7;line-height:1.9">' + " &nbsp; ".join(
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:rgb{vc.CLASS_COLOR[c]};'
            f'margin-right:4px"></span>{c.replace("_", " ")}' for c in vc.REVIEW_CLASSES)
             + ' &nbsp; · &nbsp; dashed ring = awaiting review &nbsp; · &nbsp; ✔ solid ring = reviewed</div>')
        st.button("✔ Confirm every PASS / EMPTY verdict in this image", key="rev_conf_all", on_click=rev_confirm_unflagged,
                  help="For speed: marks all positions the system passed or found empty as reviewed. Flagged ones stay "
                       "for you to look at one by one.")
    with right_:
        st.selectbox("Position", opts, key=_rev_key(),
                     format_func=lambda p: f"{'✔' if p in reviewed_here else '·'}  {ICON.get(results[p].status, '')}  {p}"
                                           f"  ·  {results[p].status}{'  ·  ' + results[p].label if results[p].label else ''}")
        r_ = results[pick]
        fx_name = "good" if pick[0] == "G" else "scrap"
        n_ = int(pick[1:]); col_from_right, row_ = (n_ - 1) // 5 + 1, (n_ - 1) % 5 + 1
        st.markdown(f"**{pick}** · {fx_name} fixture · row {row_}, column {col_from_right} from right")
        st.image(vc.crop_with_ring(crops[pick], r, color=vc.STATUS_COLOR.get(r_.status)), use_container_width=True)
        sys_cls = vc.system_class(r_) if r_.source != "operator" else st.session_state.get(f"rev_sys_{chosen}_{pick}", "needs_review")
        if r_.source != "operator":
            st.session_state[f"rev_sys_{chosen}_{pick}"] = sys_cls
        cur_cls = vc.system_class(r_)
        if cur_cls == "no_view":
            st.info("Hidden from the camera (plate edge / robot arm) - no verdict.")
        else:
            st.session_state.setdefault(f"rev_cls_{chosen}_{pick}", cur_cls)
            st.selectbox("Visible vial class", vc.REVIEW_CLASSES, key=f"rev_cls_{chosen}_{pick}",
                         on_change=rev_set_class, args=(pick,),
                         help="Change it if the system is wrong. The line message and reports use your class.")
            st.session_state[f"rev_ok_{chosen}_{pick}"] = pick in reviewed_here
            st.checkbox("I reviewed this position", key=f"rev_ok_{chosen}_{pick}", on_change=rev_toggle, args=(pick,))
            if r_.label in vc.OFFICIAL_LABELS:
                st.caption(f"Official definition · {r_.label}: {vc.OFFICIAL_LABELS[r_.label]}")
        st.text_area("Observation or correction", value=NOTES.get(chosen, {}).get(pick, ""),
                     key=f"rev_note_{chosen}_{pick}", on_change=rev_note, args=(pick,), height=70)
        with st.expander("Why the system decided this", expanded=True):
            st.markdown("\n".join(f"- {line}" for line in vc.explain(r_)))
            _v = cache.get((geom_key, pick, "wide", model))
            _s = cache.get((geom_key, pick, "second", model)) if _v and vc.second_look_prompt(feats[pick], _v) else None
            html(ui.trace(r_, _v, _s))

    # ---- export / load the review --------------------------------------------------------------------------------
    def _review_json():
        out = {}
        for n in names:
            if not done[n]:
                continue
            res_n = results_for(preps[n])
            out[n] = {p: {"system_status": x.status, "final_class": vc.system_class(x), "label": x.label,
                          "decided_by": x.source, "reviewed": p in REV.get(n, set()),
                          "note": NOTES.get(n, {}).get(p, ""), "reason": x.reason}
                      for p, x in res_n.items()}
        return json.dumps({"tool": "Vial Inspector review", "images": out}, indent=1)

    e1, e2 = st.columns(2)
    e1.download_button("⬇ Export review JSON", _review_json(), file_name="vial_review.json", mime="application/json",
                       use_container_width=True)
    up_rev = e2.file_uploader("Load review JSON", type=["json"], key="rev_load", label_visibility="collapsed")
    if up_rev is not None and st.session_state.get("rev_loaded") != up_rev.name + str(up_rev.size):
        try:
            data = json.loads(up_rev.getvalue())["images"]
            for n, pos in data.items():
                if n not in names:
                    continue
                for p, d_ in pos.items():
                    if d_.get("reviewed"):
                        REV.setdefault(n, set()).add(p)
                    if d_.get("note"):
                        NOTES.setdefault(n, {})[p] = d_["note"]
                    if d_.get("decided_by") == "operator" and d_.get("final_class") in vc.REVIEW_CLASSES[:5]:
                        st.session_state.setdefault("operator", {}).setdefault(n, {})[p] = f"label:{d_['final_class']}"
            st.session_state["rev_loaded"] = up_rev.name + str(up_rev.size)
            st.rerun()
        except Exception as ex:  # noqa: BLE001
            st.error(f"Could not read that review file: {ex}")

with t_twin:
    twin_section("results", real_img=img)

with t_ev:
    st.markdown("#### What we measured")
    html(ui.evidence(vc.MEASURED))
    st.caption("All real-frame numbers come from CSI camera frames (Dataset 1), labelled by eye before seeing system "
               "output; held-out frames were only scored, never tuned on. Details: evaluation/RESULTS.md.")
    st.download_button("⬇ Results summary (Markdown, for the report)", vc.evidence_markdown(),
                       file_name="vial_inspector_results.md", mime="text/markdown", key="ev_md")


def operator_decide(name, pid, d):
    st.session_state.setdefault("operator", {}).setdefault(name, {})[pid] = d
    st.session_state.setdefault("reviewed", {}).setdefault(name, set()).add(pid)   # a one-click answer is a review


with t_bad:
    if not bad:
        st.success("No defects, reviews, or gaps found.")
    if next_q:
        st.info(f"⚡ {len(next_q)} quick check{'s' if len(next_q) != 1 else ''} right after the last capped vial: empty "
                "hole (waiting) or the newest vial without its cap? One click settles it - or type the robot's placement "
                "count and these disappear. (Kept on purpose: in a real frame, G20 is an uncapped vial exactly here.)")
    if lean_q:
        st.info(f"⚡ {len(lean_q)} quick check{'s' if len(lean_q) != 1 else ''}: a whole cap sits off the hole - a leaning "
                "vial (OK) or a tipped cap (defect)? From straight above they look alike (in the digital twin, 44% of good "
                "leaning vials land here), so one click from the operator settles it. They are listed last.")
    for i in range(0, len(bad), 4):
        row = st.columns(4)
        for col, pid in zip(row, bad[i:i + 4]):
            r_ = results[pid]
            with col.container(border=True):
                st.image(vc.crop_with_ring(crops[pid], r, color=vc.STATUS_COLOR.get(r_.status)), use_container_width=True)
                html(ui.card_head(pid, r_.status, r_.label, r_.confidence))
                html(ui.reason(r_.reason))
                if r_.label in vc.OFFICIAL_LABELS:
                    st.caption(f"Official definition · {r_.label}: {vc.OFFICIAL_LABELS[r_.label]}")
                _v = cache.get((geom_key, pid, "wide", model))
                _s = cache.get((geom_key, pid, "second", model)) if vc.second_look_prompt(feats[pid], _v) else None
                html(ui.trace(r_, _v, _s))
                if pid in next_q:
                    b1, b2 = st.columns(2)
                    b1.button("✔ Empty hole", key=f"empty_{chosen}_{pid}", use_container_width=True,
                              on_click=operator_decide, args=(chosen, pid, "empty"))
                    b2.button("✖ Uncapped vial", key=f"unc_{chosen}_{pid}", use_container_width=True,
                              on_click=operator_decide, args=(chosen, pid, "uncapped"))
                if pid in lean_q:
                    b1, b2 = st.columns(2)
                    b1.button("✔ Just leaning", key=f"lean_{chosen}_{pid}", use_container_width=True,
                              on_click=operator_decide, args=(chosen, pid, "lean"))
                    b2.button("✖ Cap tipped", key=f"tip_{chosen}_{pid}", use_container_width=True,
                              on_click=operator_decide, args=(chosen, pid, "tipped"))
    _ops = st.session_state.get("operator", {}).get(chosen, {})
    if _ops:
        _names = {"lean": "leaning (OK)", "tipped": "cap tipped", "empty": "empty hole", "uncapped": "uncapped vial"}
        st.caption("Operator decisions on this image: " + ", ".join(f"{p} {_names.get(d, d)}" for p, d in sorted(_ops.items())))
        if st.button("↩ Undo operator decisions on this image", key=f"undo_ops_{chosen}"):
            st.session_state["operator"].pop(chosen, None)
            st.rerun()

with t_map:
    html(ui.fixture_maps({p: r_.status for p, r_ in results.items()}, vc.position_id))
    st.caption("Laid out like the physical plates. Hover a position for details. "
               "Numbering follows the robot fill order: rightmost column top-to-bottom is 01–05.")

df = pd.DataFrame([dataclasses.asdict(r_) for r_ in results.values()])
for c_, d_ in (("yellow", 2), ("center_cov", 2), ("p90", 0), ("latency_ms", 0)):
    if c_ in df:
        df[c_] = pd.to_numeric(df[c_], errors="coerce").round(d_)
with t_all:
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download results (CSV)", df.to_csv(index=False), file_name="vial_results.csv", mime="text/csv")
    with st.expander("All crops"):
        pids = list(centers)
        for i in range(0, len(pids), 6):
            row = st.columns(6)
            for col, pid in zip(row, pids[i:i + 6]):
                col.image(vc.crop_with_ring(crops[pid], r), caption=f"{pid} · {results[pid].status}", use_container_width=True)

# --------------------------------------------------------------------------------------
# Accuracy check - label what is really in the image, get real numbers
# --------------------------------------------------------------------------------------
with t_eval:
    st.markdown("#### Label what is really there, get real numbers")
    st.caption("Look at the Scene and the crops, then pick the TRUE label for each position. Label every position "
               "(or a random set), not just the flagged ones, or the numbers will be biased. Don't tune settings on the "
               "same images you report results on.")
    pred = {p: (r_.label if r_.status in ("PASS", "REJECT", "EMPTY", "GAP", "CAP") else "REVIEW")
            for p, r_ in results.items() if r_.status != "NO VIEW"}
    st.caption("Positions marked NO VIEW are left out of the accuracy numbers, and that is stated as a coverage limit.")
    base = pd.DataFrame({"position_id": list(pred), "system_status": [results[p].status for p in pred],
                         "system_label": [pred[p] for p in pred], "true_label": [None] * len(pred)})
    up = st.file_uploader("Optional: load labels from CSV (position_id,true_label)", type=["csv"], key="truth")
    if up is not None:
        t = pd.read_csv(up)
        m = dict(zip(t["position_id"].astype(str).str.strip(), t["true_label"].astype(str).str.strip().str.lower()))
        base["true_label"] = base["position_id"].map(lambda p: m.get(p) if m.get(p) in vc.LABELS else None)
    edited = st.data_editor(
        base, hide_index=True, use_container_width=True, key=f"labels_{geom_key}",
        disabled=["position_id", "system_status", "system_label"],
        column_config={"true_label": st.column_config.SelectboxColumn("true_label", options=vc.LABELS)})
    labelled = edited.dropna(subset=["true_label"])
    truth = dict(zip(labelled["position_id"], labelled["true_label"]))
    st.download_button("⬇ Save labels (CSV)", edited[["position_id", "true_label"]].to_csv(index=False),
                       file_name=f"labels_{chosen.rsplit('.', 1)[0]}.csv")
    if not truth:
        st.info("Pick true labels in the table to see accuracy.")
    else:
        ev = vc.evaluate(pred, truth)
        exact = sum(pred[p] == truth[p] for p in truth)
        c1, c2, c3 = st.columns(3)
        c1.metric("Exact label correct", f"{exact}/{ev['n']}", help="REVIEW counts as not exact.")
        if ev["defect_recall"] is not None:
            c2.metric("Defects caught", f"{ev['defects_caught']}/{ev['defects_total']}",
                      help="A REVIEW counts as caught (a human will look). Missed = predicted pass or no_vial.")
        if ev["false_alarm_rate"] is not None:
            c3.metric("False alarms on good vials", f"{len(ev['false_alarms'])}/{ev['good_total']}")
        if ev["defects_missed"]:
            st.error(f"Missed defects (false negatives): {', '.join(ev['defects_missed'])}")
        if ev["false_alarms"]:
            st.warning(f"False alarms (good vials flagged): {', '.join(ev['false_alarms'])}")
        st.markdown("**Confusion matrix** (rows = truth, columns = system)")
        st.dataframe(pd.DataFrame(ev["confusion"]).T.rename_axis("true \\ system"), use_container_width=True)
        st.dataframe(pd.DataFrame(ev["per_class"]).T, use_container_width=True)

render_helper(results)
html(ui.footer())
