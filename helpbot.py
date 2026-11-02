"""
helpbot.py - "Vial Helper": answers questions about the Vial Inspector website and the inspection model.

Design (why it can be trusted):
  * Answers come from a curated knowledge base written against the real button / tab names in app.py, and every
    number is filled in from vial_core.MEASURED / the rule constants at answer time - so answers cannot go stale.
  * Questions about a position ("why is G15 flagged?") are answered from the CURRENT inspection result
    (vial_core.explain), not from memory.
  * If nothing matches well, it says so and suggests topics - it never guesses.
  * Optional: an NVIDIA text model may re-phrase the matched answer, but it is given ONLY the matched knowledge-base
    text and is told to refuse anything not in it; any API error falls back to the offline answer.
"""
import re
from typing import Dict, List, Optional, Tuple

import vial_core as vc

RUN_CMD = 'cmd /k "cd /d <your app folder> && python -m streamlit run app.py"'


def _facts() -> dict:
    m = vc.MEASURED
    t, e, ch, al = m["test"], m["empty"], m["change"], m["align"]
    r2, af, tw = m.get("round2", {}), m.get("autofit", {}), m.get("twin", {})
    return dict(
        t_n=t["n"], t_def=t["defects"], t_good=t["good"], t_sys=t["sys_caught"], t_base=t["base_caught"],
        t_sysfa=t["sys_fa"], t_basefa=t["base_fa"], e_n=e["n"], e_nc=e["no_count"], e_est=e["estimated"],
        e_cnt=e["with_count"], ch_frames=ch["frames"], ch_all=ch["all"], ch_aware=ch["aware"], ch_saved=ch["saved"],
        al_cases=al["cases"], al_worst=al["worst"], al_med=al["median"],
        r2_frames=r2.get("frames", "-"), r2_locked=r2.get("locked_frames", "-"), r2_good=r2.get("locked_good", "-"),
        r2_gf=r2.get("locked_good_flagged", "-"), r2_def=r2.get("locked_defects", "-"),
        r2_df=r2.get("locked_defects_flagged", "-"),
        af_min=af.get("same_min", "-"), af_max=af.get("same_max", "-"), af_var=af.get("variants", "-"),
        af_frames=af.get("frames", "-"),
        tw_pos=tw.get("positions", "-"), tw_frames=tw.get("frames", "-"),
        cov_min=f"{vc.CENTER_COV_MIN:.0%}", cov_seated=f"{vc.CAP_CLEARLY_SEATED:.0%}", margin=vc.AUTO_MARGIN,
        model=vc.DEFAULT_MODEL,
    )


from helpbot_kb import KB  # noqa: E402  (130+ checked topics - see helpbot_kb.py)

for _e in KB:
    if "OFFICIAL_LIST" in _e["a"]:
        _e["a"] = _e["a"].replace("OFFICIAL_LIST", "\n".join(f"- **{k}** - {v}" for k, v in vc.OFFICIAL_LABELS.items()))
KB_BY_ID = {k["id"]: k for k in KB}

_STOP = set("a an the is are was were do does did i you it to of in on for and or what how why can my me this that "
            "with be by at as from please there about which when where who will would should could get got".split())
POS_RE = re.compile(r"\b([GS])\s?0?([1-9]|[12][0-9]|30)\b", re.I)


def _stem(w: str) -> str:
    """Tiny suffix stripper so 'running'~'run', 'zoomed'~'zoom', 'defects'~'defect', 'trained'~'train'."""
    for suf, rep in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)] + rep
            if suf in ("ing", "ed") and len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
                w = w[:-1]                      # running -> runn -> run, stopped -> stopp -> stop
            break
    return w


def _tokens(s: str) -> List[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9_%:'/]+", s.lower()) if w not in _STOP and len(w) > 1]


_TFIDF = {}


def _tfidf_scores(q: str) -> Dict[str, float]:
    """Cosine similarity between the question and each topic's full text (title + keywords + answer)."""
    import math
    if not _TFIDF:
        docs = {e["id"]: _tokens(e["title"] + " " + " ".join(e["keys"]) * 2 + " " + e["a"]) for e in KB}
        df: Dict[str, int] = {}
        for toks in docs.values():
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log((1 + len(docs)) / (1 + c)) + 1 for t, c in df.items()}
        vecs = {}
        for i, toks in docs.items():
            tf: Dict[str, float] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            v = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items()}
            n = math.sqrt(sum(x * x for x in v.values()))
            vecs[i] = {t: x / n for t, x in v.items()}
        _TFIDF.update(idf=idf, vecs=vecs)
    idf, vecs = _TFIDF["idf"], _TFIDF["vecs"]
    qt = [t for t in _tokens(q) if t in idf]
    if not qt:
        return {}
    qv: Dict[str, float] = {}
    for t in qt:
        qv[t] = qv.get(t, 0) + idf[t]
    n = sum(x * x for x in qv.values()) ** 0.5
    return {i: sum(qv[t] / n * v.get(t, 0) for t in qv) for i, v in vecs.items()}


_DF: Dict[str, int] = {}


def _df(k: str) -> int:
    if not _DF:
        for e in KB:
            for kk in set(e["keys"]):
                _DF[kk] = _DF.get(kk, 0) + 1
    return _DF.get(k, 1)


def _score(q: str, entry: dict) -> float:
    """Phrase matches count more than single words, and a keyword that belongs to only one topic counts more than
    one shared by several topics (so the specific topic wins over the general one)."""
    import math
    ql = " " + re.sub(r"\s+", " ", q.lower()) + " "
    s = 0.0
    for k in entry["keys"]:
        w = 1.0 + math.log(len(KB) / _df(k))
        if " " in k or "-" in k or "'" in k:
            s += 1.8 * w if k in ql else 0.0
        elif re.search(rf"\b{re.escape(k)}\b", ql):
            s += 1.0 * w
    s += 0.8 * len(set(_tokens(q)) & set(_tokens(entry["title"])))
    if entry["id"] == "autofit" and re.search(r"\b\d{3,4}\s?[x×]\s?\d{3,4}\b", ql):
        s += 6.0                                   # "1920x1080" = a question about photo size
    if entry["id"] == "labels" and re.search(r"\bwhat(?:'s| is| are) (?:a |an |the )?(missing|crooked|damaged)", ql):
        s += 6.0                                   # "what is a crooked cap" = definition
    return s


def _kw_hits(q: str, entry: dict) -> Tuple[int, int]:
    """(phrase matches, single-word matches) of an entry's keywords in the question."""
    ql = " " + re.sub(r"\s+", " ", q.lower()) + " "
    ph = sum(1 for k in entry["keys"] if (" " in k or "-" in k or "'" in k) and k in ql)
    wd = sum(1 for k in entry["keys"] if not (" " in k or "-" in k or "'" in k) and re.search(rf"\b{re.escape(k)}\b", ql))
    return ph, wd


GENERIC = {"color", "colour", "colors", "colours", "time", "tell", "new", "use", "work", "first", "help", "sure", "good",
           "value", "limit", "table", "map", "mark", "note", "order", "free", "pay", "money", "cost", "safe",
           "fail", "wrong", "light", "dark", "bright", "night", "lamp", "score", "numbers", "about", "project",
           "issue", "problem", "stuck", "broken", "start", "open", "launch", "run", "boot", "close", "quit", "update",
           "percent", "percentage", "progress", "summary", "overview", "purpose", "team", "authors", "impact", "business", "scale", "reuse",
           "favorite", "favourite", "best", "solve", "times", "joke", "song"}


RESULT_INTENT = re.compile(r"\b(why|flag\w*|pass\w*|reject\w*|review\w*|status|explain|wrong|ok|okay|happened|decid\w*|"
                           r"result|verdict|empty|missing|crooked|lean\w*|tipped|cap|gap|what about|is it)\b", re.I)


def match(q: str, top: int = 3) -> List[Tuple[float, dict]]:
    tf = _tfidf_scores(q)
    ranked = sorted(((_score(q, e) + 12.0 * tf.get(e["id"], 0.0), e) for e in KB), key=lambda x: -x[0])
    return [x for x in ranked[:top] if x[0] > 0]


def _fmt(entry: dict) -> str:
    return entry["a"].format(**_facts())


def _position_answer(pids: List[str], results: Optional[Dict[str, "vc.Result"]]) -> str:
    if not results:
        return ("I can explain a specific position once an image has been inspected - upload a photo, check the grid, "
                "and click **▶ Inspect this image**. Then ask again (e.g. *\"why is G15 flagged?\"*).")
    out = []
    for p in pids:
        r = results.get(p)
        if r is None:
            out.append(f"**{p}** is not a position in this image (positions are G01-G30 and S01-S30).")
            continue
        lines = vc.explain(r)
        defn = f"\n  - Official definition · {r.label}: {vc.OFFICIAL_LABELS[r.label]}" if r.label in vc.OFFICIAL_LABELS else ""
        out.append(f"**{p} · {r.status}{' · ' + r.label if r.label else ''}**\n" + "\n".join(f"  - {x}" for x in lines)
                   + defn)
    return "\n\n".join(out) + "\n\nSee its close-up in **🔎 Review every vial**."


def _summary_answer(results: Optional[Dict[str, "vc.Result"]]) -> Optional[str]:
    if not results:
        return None
    c: Dict[str, List[str]] = {}
    for p, r in results.items():
        c.setdefault(r.status, []).append(p)
    order = ["REJECT", "GAP", "REVIEW", "PASS", "CAP", "EMPTY", "NO VIEW"]
    parts = [f"- **{s}**: {len(c[s])}" + (f" ({', '.join(sorted(c[s])[:12])}{'…' if len(c[s]) > 12 else ''})"
                                          if s in ('REJECT', 'GAP', 'REVIEW') else "")
             for s in order if s in c]
    return "In the current image:\n" + "\n".join(parts) + "\n\nAsk *\"why is G..?\"* for any of them."


DOMAIN_RE = re.compile(r"\b(it|this|model|vial|vials|cap|caps|inspect\w*|website|app|site|tool|system|image|images|photo|photos|"
                       r"picture|pictures|camera|grid|circle|circles|hole|holes|fixture|plate|tray|robot|plc|nvidia|"
                       r"isaac|twin|api|key|label|labels|defect\w*|flag\w*|pass|reject|review|gap|empty|align\w*|"
                       r"accura\w*|result\w*|report\w*|helper|button|tab|sidebar|scrap|count|fill)\b", re.I)


SUMMARY_RE = re.compile(r"\b(summary|summarize|summarise|how many|what (was|is) (flagged|wrong)|which (vials|positions)"
                        r"|overview of (this|the) image|results? (of|for) this)\b", re.I)


def answer(q: str, results: Optional[Dict[str, "vc.Result"]] = None) -> dict:
    """Returns {'text', 'source', 'related': [titles], 'confident': bool}."""
    q = (q or "").strip()
    if not q:
        return dict(text="Ask me anything about the website or the inspection - e.g. *How do I align the circles?*, "
                         "*Why is G15 flagged?*, *What does GAP mean?*", source="help", related=[], confident=True)
    if re.search(r"nvapi-[A-Za-z0-9_\-]{8,}", q):
        return dict(text="That looks like an API key - please don't paste keys into chats. Put it only in the sidebar "
                         "box **NVIDIA API key**, and if it was shared anywhere, revoke it on build.nvidia.com.",
                    source="safety", related=[], confident=True)
    bad_pos = [m.group(0).upper() for m in re.finditer(r"\b[GS]\s?0?(3[1-9]|[4-9]\d|\d{3,})\b", q, re.I)]
    if bad_pos:
        return dict(text=f"**{', '.join(bad_pos)}** is not a position. Each plate has 30: good fixture **G01-G30**, "
                         "scrap fixture **S01-S30** (01-05 = the rightmost column, top to bottom, then one column left).",
                    source="help", related=[KB_BY_ID["count"]["title"]], confident=True)
    pids = sorted({f"{m.group(1).upper()}{int(m.group(2)):02d}" for m in POS_RE.finditer(q)})
    hits = match(q)
    if pids:
        general = hits and hits[0][0] >= 8 and hits[0][1]["id"] not in ("whyflag",)
        # "why is G15 flagged?" -> the position's real result; "which one is G01?" / "turn off the S01-S05 rule" -> topic
        if results and RESULT_INTENT.search(q) and not (general and hits[0][1]["id"] in ("numbering", "hiddenrule")):
            return dict(text=_position_answer(pids, results), source="current inspection",
                        related=[KB_BY_ID["whyflag"]["title"], KB_BY_ID["statuses"]["title"]], confident=True)
        if not general:
            return dict(text=_position_answer(pids, results), source="current inspection" if results else "help",
                        related=[KB_BY_ID["whyflag"]["title"], KB_BY_ID["statuses"]["title"]], confident=True)
    if SUMMARY_RE.search(q) and results:
        return dict(text=_summary_answer(results), source="current inspection", related=[KB_BY_ID["whyflag"]["title"]],
                    confident=True)
    strong_domain = bool(DOMAIN_RE.search(re.sub(r"\b(it|this)\b", " ", q, flags=re.I)))
    in_domain = bool(DOMAIN_RE.search(q))
    if hits and not strong_domain:            # "it"/"this" alone needs a specific keyword too ("what time is it")
        e0 = hits[0][1]
        ph, wd = _kw_hits(q, e0)
        ql = " " + q.lower() + " "
        specific = [k for k in e0["keys"] if " " not in k and k not in GENERIC and re.search(rf"\b{re.escape(k)}\b", ql)]
        if ph == 0 and wd < 2 and not specific:   # off-topic: one everyday word ("color", "time") is not enough
            hits = []
    if not hits or (hits[0][0] < 2.2 and not in_domain) or (hits[0][0] < 1.0):
        sug = [e["title"] for _, e in hits] or [KB_BY_ID[i]["title"] for i in ("start", "align", "statuses", "lean")]
        return dict(text="I'm not sure about that one, and I'd rather not guess. I can help with using the website, "
                         "what the results mean, the official labels, accuracy, the digital twin and troubleshooting. "
                         "Try one of these:", source="none", related=sug, confident=False)
    best = hits[0][1]
    alts = [e["title"] for _, e in hits[1:3]]                     # "Not what you meant?" - next best topics
    rel = [t for t in alts + [KB_BY_ID[i]["title"] for i in best.get("rel", []) if i in KB_BY_ID] if t != best["title"]]
    return dict(text=_fmt(best), source=best["title"], related=list(dict.fromkeys(rel))[:4], confident=True,
                kb_id=best["id"], candidates=[e["id"] for _, e in hits[:3]])


def by_title(title: str) -> Optional[dict]:
    for e in KB:
        if e["title"] == title:
            return dict(text=_fmt(e), source=e["title"], related=[KB_BY_ID[i]["title"] for i in e.get("rel", [])],
                        confident=True, kb_id=e["id"], candidates=[e["id"]])
    return None


def rephrase_with_model(q: str, base: dict, api_key: str, model: str = "meta/llama-3.1-8b-instruct",
                        timeout: int = 15) -> Optional[str]:
    """Optional: an NVIDIA text model answers in a friendlier way, GROUNDED ONLY in the best 3 checked answers (it may
    pick the one that fits better). Returns None on any problem - the caller keeps the offline answer."""
    import requests
    if not api_key or not base.get("confident"):
        return None
    refs = [KB_BY_ID[i] for i in base.get("candidates", []) if i in KB_BY_ID]
    ref_text = "\n\n".join(f"[{k + 1}] {e['title']}\n{_fmt(e)}" for k, e in enumerate(refs)) if refs else base["text"]
    sys_msg = ("You are Vial Helper for the Vial Inspector website. Answer the user's question using ONLY the facts in "
               "the REFERENCES (use the one that actually answers it). Keep button and tab names exactly as written. "
               "If no reference answers the question, reply exactly: NOT_IN_REFERENCE. Never invent numbers, features "
               "or steps. Max 120 words.")
    payload = {"model": model, "max_tokens": 300, "temperature": 0.1, "stream": False,
               "messages": [{"role": "system", "content": sys_msg},
                            {"role": "user", "content": f"REFERENCES:\n{ref_text}\n\nQUESTION: {q}"}]}
    try:
        r = requests.post(vc.NVIDIA_URL, json=payload, timeout=timeout,
                          headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
        if r.status_code != 200:
            return None
        txt = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:  # noqa: BLE001
        return None
    if not txt or "NOT_IN_REFERENCE" in txt:
        return None
    ref_nums = set(re.findall(r"\d+(?:\.\d+)?", ref_text))       # guard: no number that is not in the references
    if any(n not in ref_nums for n in re.findall(r"\d+(?:\.\d+)?", txt)):
        return None
    return txt
