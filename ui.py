"""
ui.py - look & feel for the Vial Inspector dashboard ("inspection control room").
Pure functions that return HTML/CSS strings (no Streamlit import), so they can be previewed in any browser.
IMPORTANT: Streamlit's markdown treats blank lines / 4-space indents as code, so every builder returns ONE line.
"""
from __future__ import annotations

import math
from html import escape
from typing import Dict, Iterable, List, Optional

# status -> (rgb triplet, short label, tooltip)
STATUS = {
    "PASS":    ("52,211,153",  "PASS",    "Vial with a cap, nothing wrong found"),
    "CAP":     ("56,189,248",  "CAP",     "Yellow cap present - local check only, not verified for crookedness or damage"),
    "REJECT":  ("248,113,113", "REJECT",  "Defect found (missing cap / crooked cap / damaged vial)"),
    "REVIEW":  ("251,191,36",  "REVIEW",  "Uncertain, model error or disagreement - a human must look"),
    "GAP":     ("192,132,252", "GAP",     "Empty position inside the fill pattern - possible missed placement"),
    "EMPTY":   ("100,116,139", "EMPTY",   "Empty position - waiting for vials"),
    "NO VIEW": ("71,85,105",   "NO VIEW", "Hidden from the camera - not judged, inspect by eye"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
:root{
  --bg:#060a13; --panel:rgba(17,25,43,.72); --panel-solid:#0f1729; --line:rgba(148,163,184,.12); --line2:rgba(148,163,184,.22);
  --text:#e6edf7; --muted:#8b9ab4; --dim:#5d6b85; --accent:#22d3ee; --accent2:#818cf8;
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,-apple-system,Segoe UI,sans-serif; }
.stApp{ color:var(--text); background:
  radial-gradient(1200px 600px at 100% -10%, rgba(129,140,248,.16), transparent 60%),
  radial-gradient(900px 500px at -10% 0%, rgba(34,211,238,.10), transparent 55%),
  linear-gradient(180deg,#060a13,#070b16 40%,#060a13); }
.stApp:before{ content:""; position:fixed; inset:0; pointer-events:none; opacity:.35;
  background-image:linear-gradient(rgba(148,163,184,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.05) 1px,transparent 1px);
  background-size:44px 44px; mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%); -webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 30%,transparent 75%); }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:1rem; max-width:1360px; }
h1,h2,h3,h4{ font-family:'Space Grotesk',Inter,sans-serif !important; letter-spacing:-.01em; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b1222,#070c18); border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1{ font-family:'Space Grotesk',sans-serif; font-size:20px; letter-spacing:.02em; }

/* ---------- top bar ---------- */
.vi-top{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:12px 18px; margin-bottom:16px;
  border:1px solid var(--line); border-radius:16px; background:linear-gradient(90deg,rgba(15,23,41,.9),rgba(15,23,41,.55)); backdrop-filter:blur(10px); }
.vi-brand{ display:flex; align-items:center; gap:12px; }
.vi-logo{ width:36px; height:36px; border-radius:11px; display:grid; place-items:center;
  background:linear-gradient(135deg,#22d3ee,#818cf8); box-shadow:0 6px 22px -6px rgba(34,211,238,.6); }
.vi-name{ font:700 17px/1 'Space Grotesk',sans-serif; letter-spacing:.06em; }
.vi-name small{ display:block; margin-top:4px; font:500 11px/1 'JetBrains Mono',monospace; color:var(--muted); letter-spacing:.12em; }
.vi-status{ display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
.vi-tag{ font:600 11.5px/1 'JetBrains Mono',monospace; color:#cbd5e1; padding:7px 10px; border-radius:999px; border:1px solid var(--line2);
  background:rgba(255,255,255,.03); display:inline-flex; align-items:center; gap:7px; white-space:nowrap; }
.vi-tag i{ width:7px; height:7px; border-radius:50%; background:#34d399; box-shadow:0 0 10px #34d399; display:inline-block; }
.vi-tag.warn i{ background:#fbbf24; box-shadow:0 0 10px #fbbf24; }

/* ---------- hero ---------- */
.vi-hero{ position:relative; overflow:hidden; border:1px solid var(--line); border-radius:22px; padding:28px 30px; margin-bottom:16px;
  background:linear-gradient(120deg,rgba(34,211,238,.10),rgba(129,140,248,.12) 55%,rgba(6,10,19,.2)),var(--panel-solid);
  display:grid; grid-template-columns:1.5fr 1fr; gap:24px; align-items:center; }
@media (max-width:980px){ .vi-hero{ grid-template-columns:1fr; } }
.vi-eyebrow{ font:700 11px/1 'JetBrains Mono',monospace; letter-spacing:.22em; text-transform:uppercase; color:var(--accent); margin-bottom:12px; }
.vi-title{ font:700 46px/1.02 'Space Grotesk',sans-serif; letter-spacing:-.035em; margin:0 0 12px;
  background:linear-gradient(95deg,#fff 15%,#a5f3fc 55%,#c7d2fe); -webkit-background-clip:text; background-clip:text; color:transparent; }
.vi-sub{ color:var(--muted); font-size:15.5px; max-width:640px; line-height:1.6; margin:0; }
.vi-stats{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }
.vi-stat{ border:1px solid var(--line2); border-radius:14px; padding:13px 14px; background:rgba(6,10,19,.45); }
.vi-stat b{ display:block; font:700 24px/1 'Space Grotesk',sans-serif; color:#fff; }
.vi-stat span{ display:block; margin-top:6px; color:var(--muted); font-size:11.5px; line-height:1.35; }

/* ---------- stepper ---------- */
.vi-steps{ display:flex; align-items:center; gap:0; margin:0 0 18px; padding:14px 18px; border:1px solid var(--line); border-radius:16px; background:var(--panel); }
.vi-st{ display:flex; align-items:center; gap:10px; flex:none; }
.vi-st .dot{ width:30px; height:30px; border-radius:50%; display:grid; place-items:center; font:700 12px/1 'JetBrains Mono',monospace;
  border:1.5px solid var(--line2); color:var(--muted); background:rgba(255,255,255,.02); }
.vi-st .lbl{ font-weight:600; font-size:13.5px; color:var(--muted); }
.vi-st .lbl small{ display:block; font-weight:500; font-size:11px; color:var(--dim); margin-top:2px; }
.vi-st.done .dot{ background:rgba(52,211,153,.15); border-color:#34d399; color:#34d399; }
.vi-st.done .lbl{ color:#d1fae5; }
.vi-st.active .dot{ background:linear-gradient(135deg,#22d3ee,#818cf8); border:0; color:#06111f; box-shadow:0 0 0 5px rgba(34,211,238,.15); }
.vi-st.active .lbl{ color:#fff; }
.vi-line{ flex:1; height:2px; margin:0 14px; border-radius:2px; background:var(--line2); min-width:18px; }
.vi-line.done{ background:linear-gradient(90deg,#34d399,#22d3ee); }
@media (max-width:980px){ .vi-st .lbl small{ display:none; } }

/* ---------- panels ---------- */
.vi-panel{ border:1px solid var(--line); border-radius:18px; padding:16px 18px; background:var(--panel); margin-bottom:14px; }
.vi-h{ font:600 12px/1 'JetBrains Mono',monospace; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin:2px 0 12px; display:flex; justify-content:space-between; }
.vi-h em{ font-style:normal; color:var(--dim); letter-spacing:.06em; }

/* alignment gauges */
.vi-gauges{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.vi-gauge{ display:flex; align-items:center; gap:12px; border:1px solid var(--line2); border-radius:14px; padding:10px 12px; background:rgba(6,10,19,.4); }
.vi-gauge b{ display:block; font:700 18px/1 'Space Grotesk',sans-serif; color:#fff; }
.vi-gauge span{ display:block; font-size:11.5px; color:var(--muted); margin-top:4px; }

/* verdict */
.vi-verdict{ display:flex; align-items:center; gap:14px; border-radius:18px; padding:16px 18px; margin:0 0 14px; border:1px solid rgba(var(--rgb),.45);
  background:linear-gradient(110deg,rgba(var(--rgb),.20),rgba(var(--rgb),.04) 70%),var(--panel-solid); box-shadow:0 18px 40px -24px rgba(var(--rgb),.7); }
.vi-verdict .big{ font:700 22px/1.15 'Space Grotesk',sans-serif; letter-spacing:-.02em; }
.vi-verdict .small{ color:var(--muted); font-size:13px; margin-top:4px; line-height:1.45; }
.vi-verdict .ic{ width:44px; height:44px; border-radius:13px; display:grid; place-items:center; font-weight:800; font-size:21px; color:#06111f; background:rgb(var(--rgb)); flex:none; }

/* KPI cards */
.vi-cards{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:10px; margin:0 0 14px; }
.vi-card{ position:relative; overflow:hidden; background:var(--panel-solid); border:1px solid var(--line); border-radius:14px; padding:12px 14px; }
.vi-card:after{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:rgb(var(--rgb)); box-shadow:0 0 14px rgb(var(--rgb)); }
.vi-card .v{ font:700 28px/1 'Space Grotesk',sans-serif; letter-spacing:-.03em; color:#fff; }
.vi-card .l{ font:700 10.5px/1 'JetBrains Mono',monospace; letter-spacing:.14em; color:rgb(var(--rgb)); margin-top:8px; }

/* fill strips */
.vi-fill{ margin:4px 0 12px; }
.vi-fill .t{ display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-bottom:6px; }
.vi-fill .t b{ color:#fff; font-family:'JetBrains Mono',monospace; }
.vi-segs{ display:grid; grid-template-columns:repeat(30,1fr); gap:3px; }
.vi-seg{ height:16px; border-radius:4px; background:rgba(var(--rgb),.85); position:relative; }
.vi-seg.empty{ background:rgba(148,163,184,.12); }
.vi-seg.cut:after{ content:""; position:absolute; right:-3px; top:-4px; bottom:-4px; width:2px; background:#fff; border-radius:2px; box-shadow:0 0 8px #fff; }
.vi-fill .k{ display:flex; justify-content:space-between; font:500 10px/1 'JetBrains Mono',monospace; color:var(--dim); margin-top:5px; }

/* flagged list */
.vi-list{ display:flex; flex-direction:column; gap:7px; }
.vi-row{ display:flex; align-items:center; gap:10px; padding:9px 11px; border-radius:12px; background:rgba(6,10,19,.45); border:1px solid var(--line); }
.vi-row .pid{ font:700 14px/1 'JetBrains Mono',monospace; color:#fff; width:42px; flex:none; }
.vi-row .why{ font-size:12px; color:#b6c2d6; line-height:1.35; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
.vi-ok{ padding:14px; border-radius:12px; text-align:center; color:#a7f3d0; background:rgba(52,211,153,.08); border:1px dashed rgba(52,211,153,.35); font-size:13px; }

/* pills + position cards */
.vi-pill{ display:inline-block; font:700 10.5px/1 'JetBrains Mono',monospace; letter-spacing:.1em; padding:5px 8px; border-radius:999px; flex:none;
  color:rgb(var(--rgb)); background:rgba(var(--rgb),.13); border:1px solid rgba(var(--rgb),.5); }
.vi-pid{ font:700 20px/1 'JetBrains Mono',monospace; color:#fff; margin-right:8px; }
.vi-head{ display:flex; align-items:center; justify-content:space-between; margin:8px 0 6px; }
.vi-reason{ color:#c3cee4; font-size:13px; line-height:1.5; }
.vi-meta{ color:var(--muted); font:500 11px/1.4 'JetBrains Mono',monospace; margin-top:4px; }
[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:16px !important; border-color:var(--line) !important; background:var(--panel); }

/* fixture maps */
.vi-maps{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:900px){ .vi-maps{ grid-template-columns:1fr; } }
.vi-map{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:16px 18px 18px; }
.vi-map h4{ margin:0 0 12px; font-size:14px; font-weight:700; display:flex; justify-content:space-between; }
.vi-map h4 span{ color:var(--muted); font-weight:500; font-size:12px; font-family:Inter,sans-serif; }
.vi-grid{ display:grid; grid-template-columns:repeat(6,1fr); gap:8px; }
.vi-tile{ aspect-ratio:1/1; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center;
  background:radial-gradient(circle at 35% 30%,rgba(var(--rgb),.30),rgba(var(--rgb),.08) 70%); border:2px solid rgba(var(--rgb),.75);
  color:rgb(var(--rgb)); font:700 12px/1 'JetBrains Mono',monospace; transition:transform .15s, box-shadow .15s; }
.vi-tile:hover{ transform:scale(1.1); box-shadow:0 0 18px rgba(var(--rgb),.5); }
.vi-tile i{ font-style:normal; font-size:8.5px; opacity:.85; margin-top:3px; letter-spacing:.06em; }
.vi-legend{ display:flex; flex-wrap:wrap; gap:14px; margin:14px 2px 0; color:var(--muted); font-size:12px; }
.vi-legend span:before{ content:""; display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; background:rgb(var(--rgb)); }

/* widgets */
button[data-baseweb="tab"]{ font-weight:600; font-size:14.5px; padding:10px 14px; }
[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ background:linear-gradient(90deg,#22d3ee,#818cf8); height:3px; }
.stButton > button, .stDownloadButton > button{ border-radius:12px; font-weight:600; border:1px solid var(--line2); background:rgba(255,255,255,.03); color:#e2e8f0; }
.stButton > button:hover{ border-color:#22d3ee; color:#fff; }
.stButton > button[kind="primary"]{ border:0; background:linear-gradient(95deg,#06b6d4,#6366f1); color:#fff; padding:.6rem 1.1rem;
  box-shadow:0 10px 28px -12px rgba(99,102,241,.9); }
.stButton > button[kind="primary"]:disabled{ background:#1e293b; color:#64748b; box-shadow:none; }
[data-testid="stFileUploaderDropzone"]{ border:1.5px dashed rgba(34,211,238,.4); border-radius:18px; background:rgba(34,211,238,.035); }
[data-testid="stImage"] img{ border-radius:14px; }
div[data-testid="stExpander"]{ border-radius:16px; border:1px solid var(--line); background:var(--panel); }
[data-testid="stMetric"]{ background:var(--panel-solid); border:1px solid var(--line); border-radius:14px; padding:12px 14px; }
.vi-foot{ margin:26px 0 8px; text-align:center; color:var(--dim); font:500 11px/1.6 'JetBrains Mono',monospace; letter-spacing:.08em; }
/* line actions */
.vi-state{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }
.vi-state .big{ font:700 22px/1 'Space Grotesk',sans-serif; color:rgb(var(--rgb)); letter-spacing:.04em; }
.vi-state .sub{ color:var(--muted); font-size:12px; }
.vi-act{ display:grid; grid-template-columns:46px 1fr; gap:10px; align-items:center; padding:8px 10px; border-radius:11px;
  background:rgba(6,10,19,.45); border:1px solid var(--line); margin-bottom:6px; }
.vi-act .pid{ font:700 13px/1 'JetBrains Mono',monospace; color:#fff; }
.vi-act .what{ font:700 11px/1.2 'JetBrains Mono',monospace; letter-spacing:.06em; color:rgb(var(--rgb)); }
.vi-act .why{ font-size:11.5px; color:var(--muted); margin-top:3px; line-height:1.35; }
.vi-tags{ margin-top:8px; font:500 10.5px/1.6 'JetBrains Mono',monospace; color:var(--dim); word-break:break-all; }

/* decision trace */
.vi-trace{ margin-top:10px; padding-top:10px; border-top:1px dashed var(--line2); }
.vi-tr{ display:grid; grid-template-columns:92px 1fr auto; gap:8px; align-items:center; font-size:11.5px; color:var(--muted); margin:4px 0; }
.vi-bar{ height:6px; border-radius:6px; background:rgba(148,163,184,.15); overflow:hidden; }
.vi-bar span{ display:block; height:100%; border-radius:6px; background:linear-gradient(90deg,#22d3ee,#818cf8); }
.vi-tr b{ color:#e2e8f0; font:600 11px/1 'JetBrains Mono',monospace; }

/* evidence */
.vi-ev{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
.vi-evc{ border:1px solid var(--line); border-radius:18px; padding:16px 18px; background:var(--panel); }
.vi-evc h5{ margin:0 0 4px; font:700 15px/1.3 'Space Grotesk',sans-serif; color:#fff; }
.vi-pc td,.vi-pc th{ padding:5px 4px; border-bottom:1px solid var(--line); vertical-align:top; }
.vi-evc p{ margin:0 0 12px; color:var(--muted); font-size:12.5px; line-height:1.45; }
.vi-cmp{ display:grid; grid-template-columns:110px 1fr 54px; gap:8px; align-items:center; margin:7px 0; font-size:12px; color:#cbd5e1; }
.vi-cmp .tr{ height:12px; border-radius:6px; background:rgba(148,163,184,.12); overflow:hidden; }
.vi-cmp .tr span{ display:block; height:100%; border-radius:6px; background:rgb(var(--rgb)); }
.vi-cmp b{ font:700 12px/1 'JetBrains Mono',monospace; color:#fff; text-align:right; }
.vi-big{ font:700 40px/1 'Space Grotesk',sans-serif; color:#fff; letter-spacing:-.03em; }
.vi-big small{ font-size:15px; color:var(--muted); letter-spacing:0; }
</style>
"""

LOGO = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#06111f" stroke-width="2.2" stroke-linecap="round" '
        'stroke-linejoin="round"><path d="M9 3h6"/><path d="M10 3v6.5L5.5 18a2 2 0 0 0 1.8 3h9.4a2 2 0 0 0 1.8-3L14 9.5V3"/>'
        '<path d="M7.5 15h9"/></svg>')


def _rgb(status: Optional[str]) -> str:
    return STATUS.get(status or "", ("34,211,238", "", ""))[0]


def topbar(model: str, frame: str = "", live: bool = True, mode: str = "") -> str:
    tags = [f'<span class="vi-tag{"" if live else " warn"}"><i></i>{"MODEL READY" if live else "LOCAL ONLY"}</span>']
    if model:
        tags.append(f'<span class="vi-tag">{escape(model)}</span>')
    if mode:
        tags.append(f'<span class="vi-tag">{escape(mode)}</span>')
    if frame:
        tags.append(f'<span class="vi-tag">FRAME {escape(frame)}</span>')
    return (f'<div class="vi-top"><div class="vi-brand"><div class="vi-logo">{LOGO}</div><div class="vi-name">VIAL INSPECTOR'
            f'<small>CSI TESTBED · FINAL STATION · DATASET 1</small></div></div><div class="vi-status">{"".join(tags)}</div></div>')


def hero(chips: Iterable[str] = ()) -> str:
    stats = [("60", "numbered positions checked per frame"), ("5", "official labels per position"),
             ("0", "training images - zero-shot vision model"), ("100%", "of uncertain calls go to a human, never PASS")]
    s = "".join(f'<div class="vi-stat"><b>{a}</b><span>{b}</span></div>' for a, b in stats)
    return ('<div class="vi-hero"><div><div class="vi-eyebrow">Physical AI · final vial inspection</div>'
            '<div class="vi-title">Every vial checked.<br>No defect waved through.</div>'
            '<p class="vi-sub">Locates all 60 fixture positions from the overhead camera, checks each one with physical rules and an '
            'NVIDIA vision model, and shows exactly which vials are wrong and why.</p></div>'
            f'<div class="vi-stats">{s}</div></div>')


def pipeline(step: int, use_vlm: bool = True) -> str:
    """step: 0 = waiting for an image, 1 = aligning, 2 = model running, 4 = results ready"""
    steps = [("1", "Upload", "camera frame"), ("2", "Align", "60 positions"),
             ("3", "Inspect", "rules + NVIDIA model" if use_vlm else "local rules"), ("4", "Decide", "pass · reject · review")]
    out = []
    for i, (n, t, d) in enumerate(steps):
        done = step > i if step != 4 else True
        active = (not done) and (step == i or (step == 2 and i == 2))
        cls = "done" if done else ("active" if active else "")
        dot = "&#10003;" if done else n
        out.append(f'<div class="vi-st {cls}"><div class="dot">{dot}</div><div class="lbl">{t}<small>{d}</small></div></div>')
        if i < len(steps) - 1:
            out.append(f'<div class="vi-line{" done" if done else ""}"></div>')
    return f'<div class="vi-steps">{"".join(out)}</div>'


def _ring(frac: float, rgb: str, size: int = 46) -> str:
    r = size / 2 - 5
    c = 2 * math.pi * r
    frac = max(0.0, min(1.0, frac))
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}"><circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" '
            f'stroke="rgba(148,163,184,.18)" stroke-width="5"/><circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" '
            f'stroke="rgb({rgb})" stroke-width="5" stroke-linecap="round" stroke-dasharray="{c * frac:.1f} {c:.1f}" '
            f'transform="rotate(-90 {size/2} {size/2})"/></svg>')


def align_gauges(q: Dict[str, int], expect: Dict[str, int]) -> str:
    cells = []
    for fx, name in (("G", "Good fixture"), ("S", "Scrap fixture")):
        n, e = q.get(fx, 0), expect.get(fx, 30)
        frac = n / e if e else 0
        # Caps hide the round hole edge, so a correct grid on a full fixture matches ~21-30 of 30;
        # a grid one hole off matches almost none.
        rgb = "52,211,153" if frac >= 0.65 else ("251,191,36" if frac >= 0.4 else "248,113,113")
        word = "locked on" if frac >= 0.65 else ("check alignment" if frac >= 0.4 else "misaligned")
        cells.append(f'<div class="vi-gauge">{_ring(frac, rgb)}<div><b>{n}/{e}</b><span>{name} · {word}</span></div></div>')
    return ('<div class="vi-panel"><div class="vi-h">Alignment <em>holes found under the grid</em></div>'
            f'<div class="vi-gauges">{"".join(cells)}</div></div>')


def verdict(counts: Dict[str, int], local_only: bool = False) -> str:
    rej, rev, gap = counts.get("REJECT", 0), counts.get("REVIEW", 0), counts.get("GAP", 0)
    if rej:
        rgb, ic, big = STATUS["REJECT"][0], "!", f"{rej} defective vial{'s' if rej != 1 else ''} found"
        small = f"{rev} more to check by eye" + (f" · {gap} possible missed placement{'s' if gap != 1 else ''}" if gap else "")
    elif rev or gap:
        rgb, ic, big = STATUS["REVIEW"][0], "?", f"{rev + gap} position{'s' if rev + gap != 1 else ''} to check by eye"
        small = "No confirmed defects - but nothing uncertain is passed automatically."
    else:
        rgb, ic, big = STATUS["PASS"][0], "&#10003;", "All clear"
        small = "Every filled position passed all checks."
    hidden = counts.get("NO VIEW", 0)
    if hidden:
        small += f" · {hidden} hidden from the camera."
    if local_only:
        small += " Local-only mode cannot judge crooked caps or damage."
    return f'<div class="vi-verdict" style="--rgb:{rgb}"><div class="ic">{ic}</div><div><div class="big">{big}</div><div class="small">{small}</div></div></div>'


def metric_cards(counts: Dict[str, int], order: List[str], total: int) -> str:
    cards = []
    for s in order:
        rgb, label, tip = STATUS[s]
        cards.append(f'<div class="vi-card" style="--rgb:{rgb}" title="{escape(tip)}"><div class="v">{counts.get(s, 0)}</div><div class="l">{label}</div></div>')
    return f'<div class="vi-cards">{"".join(cards)}</div>'


def fill_strips(statuses: Dict[str, str], placed: Optional[Dict[str, Optional[int]]] = None) -> str:
    """One bar per fixture: positions 01..30 in ROBOT FILL ORDER, coloured by status; a marker after the placement count."""
    placed = placed or {}
    out = ['<div class="vi-panel"><div class="vi-h">Fill progress <em>robot order 01 → 30</em></div>']
    for fx, name in (("G", "Good"), ("S", "Scrap")):
        segs, filled = [], 0
        n = placed.get(fx)
        for i in range(1, 31):
            st = statuses.get(f"{fx}{i:02d}", "EMPTY")
            filled += st in ("PASS", "REJECT", "REVIEW", "CAP")
            cls = "vi-seg" + (" empty" if st == "EMPTY" else "") + (" cut" if n and i == n else "")
            segs.append(f'<div class="{cls}" style="--rgb:{_rgb(st)}" title="{fx}{i:02d} · {st}"></div>')
        count = f"{filled} occupied" + (f" · {n} placed" if n is not None else "")
        out.append(f'<div class="vi-fill"><div class="t"><span>{name} fixture</span><b>{count}</b></div>'
                   f'<div class="vi-segs">{"".join(segs)}</div><div class="k"><span>01</span><span>15</span><span>30</span></div></div>')
    out.append("</div>")
    return "".join(out)


def flagged_list(rows: List[tuple]) -> str:
    """rows: (pid, status, reason)"""
    if not rows:
        return '<div class="vi-panel"><div class="vi-h">Flagged</div><div class="vi-ok">Nothing flagged in this frame.</div></div>'
    items = "".join(f'<div class="vi-row"><div class="pid">{escape(p)}</div>{pill(s)}<div class="why">{escape(w or "-")}</div></div>'
                    for p, s, w in rows)
    return f'<div class="vi-panel"><div class="vi-h">Flagged <em>{len(rows)} position{"s" if len(rows) != 1 else ""}</em></div><div class="vi-list">{items}</div></div>'


def pill(status: str) -> str:
    return f'<span class="vi-pill" style="--rgb:{_rgb(status)}">{escape(status)}</span>'


def card_head(pid: str, status: str, label: str = "", confidence: Optional[float] = None) -> str:
    meta = []
    if label:
        meta.append(f"label: {label}")
    if confidence is not None:
        meta.append(f"confidence {confidence:.2f}")
    m = f'<div class="vi-meta">{escape(" · ".join(meta))}</div>' if meta else ""
    return f'<div class="vi-head"><div><span class="vi-pid">{escape(pid)}</span></div>{pill(status)}</div>{m}'


def reason(text: str) -> str:
    return f'<div class="vi-reason">{escape(text or "-")}</div>'


def fixture_maps(statuses: Dict[str, str], position_id) -> str:
    """Two 6x5 grids drawn like the real fixtures (left column = leftmost physical column)."""
    blocks = []
    for prefix, title, sub in (("G", "Good fixture", "left plate"), ("S", "Scrap fixture", "right plate")):
        tiles, nbad = [], 0
        for row in range(5):
            for col in range(6):
                pid = position_id(prefix, col, row)
                st = statuses.get(pid, "EMPTY")
                nbad += st in ("REJECT", "REVIEW", "GAP")
                rgb, label, tip = STATUS.get(st, STATUS["EMPTY"])
                tiles.append(f'<div class="vi-tile" style="--rgb:{rgb}" title="{pid} &middot; {st}: {escape(tip)}">{pid}<i>{label}</i></div>')
        blocks.append(f'<div class="vi-map"><h4>{title}<span>{sub} &middot; {nbad} flagged</span></h4><div class="vi-grid">{"".join(tiles)}</div></div>')
    legend = "".join(f'<span style="--rgb:{rgb}">{label.title()}</span>' for k, (rgb, label, _) in STATUS.items() if k != "CAP")
    return f'<div class="vi-maps">{"".join(blocks)}</div><div class="vi-legend">{legend}</div>'


def footer() -> str:
    return ('<div class="vi-foot">ZERO-SHOT · NO TRAINING ON REAL LINE DATA · ERRORS NEVER PASS · '
            'NVIDIA NIM + PHYSICAL RULES</div>')


STATE_RGB = {"HOLD": "248,113,113", "CHECK": "251,191,36", "CONTINUE": "52,211,153"}
ACTION_RGB = {"DIVERT_TO_SCRAP": "248,113,113", "REPLACE_MISSED_VIAL": "192,132,252", "OPERATOR_CHECK": "251,191,36",
              "RECAPTURE_FRAME": "100,116,139", "LOG_SCRAP_DEFECT": "148,163,184"}


def line_panel(msg: dict) -> str:
    st_ = msg["line_state"]
    sub = {"HOLD": "defect in the good fixture or a missed placement - act before continuing",
           "CHECK": "no confirmed defect - an operator should look at the flagged positions",
           "CONTINUE": "nothing to do - the robot can keep filling"}[st_]
    acts = "".join(
        f'<div class="vi-act" style="--rgb:{ACTION_RGB.get(a["action"], "148,163,184")}"><div class="pid">{escape(a["position"][:12])}</div>'
        f'<div><div class="what">{escape(a["action"].replace("_", " "))}</div><div class="why">{escape(a["why"][:110])}</div></div></div>'
        for a in msg["actions"][:6])
    more = f'<div class="vi-tags">+ {len(msg["actions"]) - 6} more in the message</div>' if len(msg["actions"]) > 6 else ""
    g = msg["plc_tags"]
    tags = " · ".join(f"{k.split('.')[-1]}={v}" for k, v in g.items() if k.split(".")[-1].startswith("G_"))
    return (f'<div class="vi-panel"><div class="vi-h">Line action <em>message for the robot / PLC</em></div>'
            f'<div class="vi-state" style="--rgb:{STATE_RGB[st_]}"><div><div class="big">{st_}</div><div class="sub">{sub}</div></div>'
            f'</div>'
            f'{acts}{more}<div class="vi-tags">PLC tags (30-bit masks, bit n-1 = position n): {escape(tags)}</div></div>')


def trace(res, vlm: Optional[dict], second: Optional[dict]) -> str:
    """Evidence behind one verdict: local measurements, model answers, and which rule decided."""
    def bar(lbl, frac, val):
        frac = max(0.0, min(1.0, frac))
        return f'<div class="vi-tr"><span>{lbl}</span><div class="vi-bar"><span style="width:{frac * 100:.0f}%"></span></div><b>{val}</b></div>'
    rows = [bar("cap over hole", res.center_cov, f"{res.center_cov:.0%}"), bar("yellow nearby", res.yellow, f"{res.yellow:.0%}")]
    if vlm and "label" in vlm:
        c = vlm.get("confidence")
        rows.append(f'<div class="vi-tr"><span>model said</span><span style="color:#e2e8f0">{escape(vlm["label"])}</span>'
                    f'<b>{"" if c is None else f"{c:.2f}"}</b></div>')
    elif vlm and "error" in vlm:
        rows.append('<div class="vi-tr"><span>model said</span><span style="color:#fbbf24">no answer (error)</span><b></b></div>')
    if second and "label" in second:
        rows.append(f'<div class="vi-tr"><span>second look</span><span style="color:#e2e8f0">{escape(second["label"])}</span><b></b></div>')
    rows.append(f'<div class="vi-tr"><span>decided by</span><span style="color:#a5f3fc">{escape(res.source or "-")}</span><b></b></div>')
    return f'<div class="vi-trace">{"".join(rows)}</div>'


def per_class_table(pc: dict) -> str:
    """Per-class table: defects caught (missed = escapes) and false alarms, model mode vs camera checks only."""
    m, l = pc["model"], pc["local"]
    def cell(t, bad_when_short=True):
        if t is None:
            return '<td style="color:var(--muted)">-</td>'
        k, n = t
        col = "#34d399" if (k == n if bad_when_short else k == 0) else ("#f87171" if bad_when_short else "#fbbf24")
        return f'<td style="color:{col};font-weight:700">{k}/{n}</td>'
    rows = [("missing_cap caught", m.get("missing_cap"), l.get("missing_cap"), True),
            ("crooked_cap caught", m.get("crooked_cap"), l.get("crooked_cap"), True),
            ("missed placement caught (GAP)", m.get("missed_placement"), l.get("missed_placement"), True),
            ("good vials flagged", m.get("pass_flagged"), l.get("pass_flagged"), False),
            ("empty holes flagged (robot count)", m.get("empty_flagged_count"), l.get("empty_flagged_count"), False),
            ("empty holes flagged (fill from image)", m.get("empty_flagged_est"), l.get("empty_flagged_est"), False)]
    body = "".join(f'<tr><td style="color:#cbd5e1">{escape(lbl)}</td>{cell(a, w)}{cell(b, w)}</tr>' for lbl, a, b, w in rows)
    return (f'<table style="width:100%;border-collapse:collapse;font-size:12.5px" class="vi-pc">'
            f'<tr style="color:var(--muted);text-align:left"><th></th><th>NVIDIA model<br><small>{m["frames"]} frames</small></th>'
            f'<th>Camera only<br><small>{l["frames"]} frames</small></th></tr>{body}</table>'
            f'<p style="margin-top:8px">Missed (escapes): {escape(pc["misses"])}. damaged_vial: no real example.</p>')


def evidence(ev: dict) -> str:
    """Measured results, rendered as comparison cards. ev holds only numbers that were measured."""
    def cmp(rows, maxv):
        return "".join(f'<div class="vi-cmp" style="--rgb:{rgb}"><span>{escape(lbl)}</span><div class="tr"><span style="width:{(v / maxv * 100) if maxv else 0:.0f}%"></span></div><b>{escape(txt)}</b></div>'
                       for lbl, v, txt, rgb in rows)
    t = ev["test"]
    cards = [
        ('Defects caught on held-out frames',
         f'6 real frames never used for tuning ({t["n"]} positions, {t["defects"]} defects). Higher is better.',
         cmp([("Model alone", t["base_caught"], f'{t["base_caught"]}/{t["defects"]}', "148,163,184"),
              ("This system", t["sys_caught"], f'{t["sys_caught"]}/{t["defects"]}', "52,211,153")], t["defects"])),
        ('Good vials wrongly flagged',
         f'Same frames, {t["good"]} good vials. Lower is better.',
         cmp([("Model alone", t["base_fa"], f'{t["base_fa"]}/{t["good"]}', "248,113,113"),
              ("This system", t["sys_fa"], f'{t["sys_fa"]}/{t["good"]}', "52,211,153")], max(1, t["base_fa"]))),
        ('Empty holes wrongly flagged',
         'The model sees "phantom caps" in bright empty holes. Using the fill order removes them - estimated from the '
         'image, or exact with the robot\'s count. Defects caught stay 6/6 in every mode.',
         cmp([("Model + rules", ev["empty"]["no_count"], f'{ev["empty"]["no_count"]}/{ev["empty"]["n"]}', "251,191,36"),
              ("+ fill from image", ev["empty"]["estimated"], f'{ev["empty"]["estimated"]}/{ev["empty"]["n"]}', "34,211,238"),
              ("+ robot count", ev["empty"]["with_count"], f'{ev["empty"]["with_count"]}/{ev["empty"]["n"]}', "52,211,153")],
             max(1, ev["empty"]["no_count"]))),
        ('Change-aware inspection',
         f'{ev["change"]["frames"]} real frames from one shift: only positions that changed since they were last inspected '
         'go to the model.',
         f'<div class="vi-big">{ev["change"]["saved"]}<small> fewer model calls</small></div>'
         + cmp([("Every frame", ev["change"]["all"], f'{ev["change"]["all"]:,}', "148,163,184"),
                ("Change-aware", ev["change"]["aware"], f'{ev["change"]["aware"]:,}', "34,211,238")], ev["change"]["all"])),
        ('Auto-align robustness',
         f'{ev["align"]["cases"]} cases: real frames moved by known amounts (up to 40 px and 3&deg;).',
         f'<div class="vi-big">&le; {ev["align"]["worst"]} px</div><p style="margin:6px 0 0">worst error after one click</p>'
         f'<p style="margin-top:8px">A hole\'s radius is about 10.5 px; median error {ev["align"]["median"]} px; '
         f'{ev["align"]["seconds"]} s per image.</p>'),
        ('28 more real photos, labelled first',
         f'Labelled by eye before the system saw them; 15 to look for mistakes, {ev["round2"]["locked_frames"]} kept locked '
         f'and only scored. Local checks only (no API):',
         cmp([("Good vials flagged", ev["round2"]["locked_good_flagged"], f'{ev["round2"]["locked_good_flagged"]}/{ev["round2"]["locked_good"]}', "52,211,153"),
              ("Defects flagged", ev["round2"]["locked_defects_flagged"], f'{ev["round2"]["locked_defects_flagged"]}/{ev["round2"]["locked_defects"]}', "52,211,153")],
             ev["round2"]["locked_defects"])
         + f'<p style="margin-top:8px">The 2 unflagged are both G19 (one vial, two frames), labelled crooked_cap by eye; '
           f'it leans and its cap reads as seated. We count them as misses. Empty holes: '
           f'{ev["round2"]["locked_empty_checks"]}/{ev["round2"]["locked_empty"]} one-click checks right after the last cap, '
           f'{ev["round2"]["locked_empty_checks_count"]} with the robot count.</p>'),
        ('Any camera: auto-fit',
         f'{ev["autofit"]["frames"]} real frames, each changed {ev["autofit"]["variants"]} ways a different camera would '
         f'(2x resolution, 16:9, rotated 7&deg;, zoomed out, zoomed in, moved). The system finds both fixtures, '
         f'straightens the photo into the standard view, and inspects it:',
         f'<div class="vi-big">{ev["autofit"]["same_min"]:.0f}-{ev["autofit"]["same_max"]:.0f}%<small> same decisions as the '
         f'original</small></div><p style="margin-top:8px">The only defect label that changed was a leaning vial (G16), now '
         f'seen as capped. Standard frames are left untouched ({ev["autofit"]["standard_fitted"]} of '
         f'{ev["autofit"]["standard_checked"]} auto-fitted).</p>'),
        ('Results per class (held-out frames only)',
         'Scored on real frames never used for tuning, labels as written. "Caught" = REJECT, REVIEW or GAP. '
         'This is the operating point that catches every defect the model saw.',
         per_class_table(ev["perclass"])),
        ('Trained on the twin only: does it transfer?',
         'A small classifier trained only on digital-twin images (no real images), then scored on real frames.',
         f'<div class="vi-big">{ev["twin"]["clf_real_good_flagged"]}/{ev["twin"]["clf_real_good"]}<small> good vials '
         f'flagged</small></div><p style="margin-top:8px">At the threshold that catches every real defect '
         f'({ev["twin"]["clf_real_caught"]}/{ev["twin"]["clf_real_defects"]}), it flags every good vial - so it is '
         f'<b>not</b> used in the decision. The twin is used to <b>test and harden</b> the rules instead (escaped '
         f'defects on twin frames {ev["twin"]["esc_before"]} &rarr; {ev["twin"]["esc_after"]}).</p>'),
        ('Honest limits',
         'Small real test: 14 held-out defects, and 3 of the crooked caps are one leaning vial (G19) seen in 3 frames. '
         'No damaged_vial example in the real data, so that class is untested on real frames. Labels by eye from '
         '640x480 frames. The cap check looks for the lemon-yellow cap: a new cap colour needs new settings. On a '
         'new camera (Cap_seal2) caps can sit off the hole centre and get flagged. The model needs internet and can '
         'time out (then REVIEW, never PASS). An empty bright hole and an open glass vial look alike: the robot\'s '
         'placement count settles it. Not yet run live on the CSI line.', ''),
    ]
    body = "".join(f'<div class="vi-evc"><h5>{h}</h5><p>{p}</p>{c}</div>' for h, p, c in cards)
    return f'<div class="vi-ev">{body}</div>'


def build_report(sections: List[dict], meta: dict) -> str:
    """Self-contained HTML inspection record (images embedded). Each section: name, frame, overlay_b64, counts,
    msg (line_actions output), flagged [(pid, status, label, reason, crop_b64)], rows [(pid, status, label, source, reason)]."""
    def esc(x):
        return escape(str(x if x is not None else ""))
    parts = []
    for s in sections:
        cards = "".join(
            f'<div class="fc"><img src="data:image/png;base64,{b}"><div class="fh"><b>{esc(p)}</b>{pill(st_)}</div>'
            f'<div class="fl">{esc(lab)}</div><div class="fr">{esc(why)}</div></div>'
            for p, st_, lab, why, b in s["flagged"]) or '<p class="ok">Nothing flagged in this frame.</p>'
        acts = "".join(f'<tr><td>{esc(a["position"])}</td><td><b>{esc(a["action"])}</b></td><td>{esc(a["why"])}</td></tr>'
                       for a in s["msg"]["actions"]) or '<tr><td colspan="3">No action needed.</td></tr>'
        rows = "".join(f'<tr><td>{esc(p)}</td><td>{pill(st_)}</td><td>{esc(lab)}</td><td>{esc(src)}</td><td>{esc(why)}</td></tr>'
                       for p, st_, lab, src, why in s["rows"])
        parts.append(
            f'<section><h2>{esc(s["frame"])} <small>{esc(s["name"])}</small></h2>'
            f'{verdict(s["counts"])}'
            f'<div class="two"><img class="ov" src="data:image/png;base64,{s["overlay_b64"]}">'
            f'<div><h3>Line state: <span class="state {esc(s["msg"]["line_state"])}">{esc(s["msg"]["line_state"])}</span></h3>'
            f'<table class="t"><tr><th>Position</th><th>Action</th><th>Why</th></tr>{acts}</table></div></div>'
            f'<h3>Flagged positions</h3><div class="grid">{cards}</div>'
            f'<details><summary>All 60 positions</summary><table class="t"><tr><th>Pos</th><th>Status</th><th>Label</th>'
            f'<th>Decided by</th><th>Reason</th></tr>{rows}</table></details></section>')
    m = "".join(f"<li><b>{esc(k)}:</b> {esc(v)}</li>" for k, v in meta.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Vial inspection record</title>
<style>
body{{font-family:Inter,Segoe UI,system-ui,sans-serif;background:#0b1120;color:#e6edf7;margin:0;padding:28px 36px;}}
h1{{font-size:28px;margin:0 0 6px}} h2{{font-size:20px;margin:28px 0 12px}} h2 small{{color:#8b9ab4;font-size:12px;font-weight:500}}
h3{{font-size:15px;margin:18px 0 8px}} ul.meta{{color:#a9b6cc;font-size:13px;columns:2;padding-left:18px}}
section{{border:1px solid rgba(148,163,184,.18);border-radius:16px;padding:6px 20px 20px;margin-top:22px;background:#0f1729}}
.two{{display:grid;grid-template-columns:3fr 2fr;gap:18px}} img.ov{{width:100%;border-radius:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}}
.fc{{background:#0b1222;border:1px solid rgba(148,163,184,.18);border-radius:12px;padding:8px}} .fc img{{width:100%;border-radius:8px}}
.fh{{display:flex;justify-content:space-between;align-items:center;margin-top:6px}} .fl{{color:#8b9ab4;font-size:11px}}
.fr{{font-size:12px;color:#c3cee4;margin-top:4px}} .ok{{color:#a7f3d0}}
table.t{{width:100%;border-collapse:collapse;font-size:12px}} .t th,.t td{{border-bottom:1px solid rgba(148,163,184,.15);padding:6px;text-align:left;vertical-align:top}}
.state{{padding:3px 10px;border-radius:999px;font-family:monospace}} .HOLD{{background:#7f1d1d}} .CHECK{{background:#78350f}} .CONTINUE{{background:#14532d}}
.vi-pill{{display:inline-block;font:700 10px/1 monospace;padding:4px 7px;border-radius:999px;color:rgb(var(--rgb));background:rgba(var(--rgb),.14);border:1px solid rgba(var(--rgb),.5)}}
.vi-verdict{{display:flex;gap:12px;align-items:center;border-radius:14px;padding:12px 14px;border:1px solid rgba(var(--rgb),.45);background:rgba(var(--rgb),.12)}}
.vi-verdict .ic{{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:rgb(var(--rgb));color:#06111f;font-weight:800}}
.vi-verdict .big{{font-weight:700;font-size:18px}} .vi-verdict .small{{color:#a9b6cc;font-size:12px}}
footer{{margin-top:26px;color:#64748b;font-size:11px;text-align:center}}
@media print{{body{{background:#fff;color:#111}} section{{background:#fff}}}}
</style></head><body><h1>Vial inspection record</h1><ul class="meta">{m}</ul>{"".join(parts)}
<footer>Generated by Vial Inspector - physical rules + NVIDIA NIM vision model (zero-shot). Uncertain or failed calls are never auto-passed.</footer>
</body></html>"""


def twin_cards(t: dict) -> str:
    """Digital-twin (Isaac Sim) results as evidence cards. Only measured numbers."""
    def cmp(rows, maxv):
        return "".join(f'<div class="vi-cmp" style="--rgb:{rgb}"><span>{escape(lbl)}</span><div class="tr"><span style="width:{(v / maxv * 100) if maxv else 0:.0f}%"></span></div><b>{escape(txt)}</b></div>'
                       for lbl, v, txt, rgb in rows)
    cards = [
        ('Synthetic data from the digital twin',
         f'NVIDIA Isaac Sim renders the CSI twin through its own CapSealCam (real lens model). Every frame gets a random '
         f'fill level, defects and lighting, so the labels are exact by construction.',
         f'<div class="vi-big">{t["positions"]:,}<small> labelled positions</small></div>'
         f'<p style="margin-top:8px">{t["frames"]} frames &middot; {t["missing"]} missing caps &middot; {t["crooked"]} crooked caps '
         f'&middot; {t["missed_placements"]} missed placements. The twin\'s camera puts every hole within '
         f'{t["grid_px"]} px (median) of our grid.</p>'),
        ('The twin found a weakness: dim light',
         f'With the station light 40% lower, good caps fell below our fixed brightness rule. Fixed by making the rule follow '
         f'the light. Good caps recognised under dim light ({t["dim"]} positions):',
         cmp([("Before", t["dim_before"], f'{t["dim_before"]}/{t["dim"]}', "248,113,113"),
              ("After", t["dim_after"], f'{t["dim_after"]}/{t["dim"]}', "52,211,153")], t["dim"])
         + f'<p style="margin-top:8px">Checked on {t["real_checked"]:,} real positions: {t["real_changed"]} decisions changed.</p>'),
        ('Defects that could slip through as "empty"',
         f'{t["defects"]} synthetic defects, judged by the local checks + fill order. Lower is better.',
         cmp([("Fill from image, before", t["esc_before"], f'{t["esc_before"]}', "248,113,113"),
              ("Fill from image, after", t["esc_after"], f'{t["esc_after"]}', "251,191,36"),
              ("With robot count", t["esc_count"], f'{t["esc_count"]}', "52,211,153")], max(1, t["esc_before"]))
         + f'<p style="margin-top:8px">Fill level read from the image exact on {t["est_before"]} &rarr; {t["est_after"]} of '
           f'{t["frames"]} frames.</p>'),
        ('Leaning vials: fewer needless stops',
         f'Leaning vials are common on the line and are OK by the official rule. We had the twin tip {t["lean_n"]} good vials '
         f'3-12&deg; in their holes ({t["lean_frames"]} frames). Before, {t["lean_review"]} went to review and '
         f'{t["lean_rej_before"]} were even rejected as "missing cap" (cap fully off the hole). Now a vial whose own whole '
         f'cap sits next to the hole is a one-click quick check:',
         cmp([("Rejected before", t["lean_rej_before"], f'{t["lean_rej_before"]}', "248,113,113"),
              ("Rejected now", t["lean_rej_after"], f'{t["lean_rej_after"]}', "52,211,153"),
              ("One-click checks", t["lean_quick"], f'{t["lean_quick"]}', "34,211,238")], t["lean_quick"])
         + '<p style="margin-top:8px">No crooked or missing cap passes: every defect is still flagged.</p>'),
        ('Why not pass leaning vials automatically?',
         f'From straight above, {t["crooked_round"]} of {t["crooked_measured"]} tipped caps look like neat round caps. A '
         f'shape rule learned on the twin cleared {t["shape_cleared"]} of 221 leaning vials on held-out frames, but also let '
         f'{t["shape_slipped"]} of {t["shape_crooked"]} crooked caps through. A missed defect costs more than one click, so '
         f'a person confirms. A side camera would settle it.', ''),
        ('Learning from its mistakes (twin-only model)',
         'Each round: train on all synthetic frames so far, test on a fixed held-out twin set, find the mistakes '
         '(e.g. 12-16&deg; tilts, 11-16&deg; leans) and render more of them. Good vials wrongly flagged on the twin test set:',
         cmp([(f'Round {x["round"]}', int(x["twin_good"].split("/")[0]), x["twin_good"], "34,211,238") for x in t["loop"]],
             max(int(x["twin_good"].split("/")[0]) for x in t["loop"]))
         + '<p style="margin-top:8px">Leaning vials misread: ' + ' &rarr; '.join(x["twin_lean"] for x in t["loop"])
         + '. On the real camera (looked at, never trained on) it still flags ' + t["loop"][-1]["real_good"]
         + ' good vials, so it stays out of the decisions.</p>'),
        ('Honest sim-to-real gap',
         f'A classifier trained only on the synthetic crops caught {t["clf_real_caught"]}/{t["clf_real_defects"]} real defects '
         f'but also flagged {t["clf_real_good_flagged"]}/{t["clf_real_good"]} real good vials: the twin\'s plate, glass and '
         f'lighting look different from the real camera. So the twin is used to test and tune rules, not to make the '
         f'final call. In the twin an uncapped vial can look exactly like an empty hole - only the robot\'s count '
         f'catches every one, which is why the app takes that count.', ''),
    ]
    body = "".join(f'<div class="vi-evc"><h5>{h}</h5><p>{p}</p>{c}</div>' for h, p, c in cards)
    return f'<div class="vi-ev">{body}</div>'
