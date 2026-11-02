"""
batch_eval.py - run the full inspection pipeline (local rules + NVIDIA model) on a list of camera
images, with no browser, and save everything to one report so results can be scored and debugged.

Usage (from this folder):
    python batch_eval.py                      # images listed in test_images.txt, read from the dataset zip
    python batch_eval.py --model meta/llama-3.2-11b-vision-instruct

The NVIDIA key is read from the NVIDIA_API_KEY environment variable (setx NVIDIA_API_KEY ... once).
Model answers are cached in vlm_cache.json, so re-running after a logic change costs no API calls.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vial_core as vc  # noqa: E402

DEFAULT_ZIP = Path.home() / "Downloads" / "images-20261101T005142Z-1-001.zip"


def load_images(zip_path: Path, names):
    out = {}
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            index = {Path(n).name: n for n in z.namelist()}
            for n in names:
                if n in index:
                    out[n] = Image.open(io.BytesIO(z.read(index[n]))).convert("RGB")
    for n in names:                         # fall back to loose files next to the zip / in Downloads
        if n not in out:
            for d in (zip_path.parent, HERE, HERE / "images"):
                if (d / n).exists():
                    out[n] = Image.open(d / n).convert("RGB")
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(DEFAULT_ZIP))
    ap.add_argument("--list", default=str(HERE / "test_images.txt"))
    ap.add_argument("--model", default=vc.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--crop", type=float, default=56)
    ap.add_argument("--no-api", action="store_true", help="local rules only")
    ap.add_argument("--no-estimate", action="store_true", help="do not estimate the fill from the image")
    args = ap.parse_args()

    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key and (HERE / "nvidia_key.txt").exists():
        key = (HERE / "nvidia_key.txt").read_text().strip()
    if not key and not args.no_api:
        print("ERROR: no NVIDIA key. Run:  setx NVIDIA_API_KEY your_key   then open a NEW cmd window.")
        sys.exit(1)

    # each line: image name, optionally followed by the robot's placement counts, e.g. "img.jpg G=12 S=0"
    names, counts = [], {}
    for line in Path(args.list).read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        names.append(parts[0])
        c = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in parts[1:] if "=" in kv}
        if c:
            counts[parts[0]] = {"G": c.get("G"), "S": c.get("S")}
    images = load_images(Path(args.zip), names)
    missing = [n for n in names if n not in images]
    if missing:
        print("WARNING: could not find", missing)

    cache_path = HERE / "vlm_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    phash = hashlib.md5(vc.PROMPT.encode()).hexdigest()[:8]
    vhashes = {vc.VERIFY_CAP_PROMPT: hashlib.md5(vc.VERIFY_CAP_PROMPT.encode()).hexdigest()[:8],
               vc.VERIFY_VIAL_PROMPT: hashlib.md5(vc.VERIFY_VIAL_PROMPT.encode()).hexdigest()[:8]}

    run_dir = HERE / "runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    crops_zip = zipfile.ZipFile(run_dir / "crops.zip", "w", zipfile.ZIP_DEFLATED)
    report = {"model": args.model, "prompt_hash": phash, "images": {}, "started": time.ctime()}
    t_all = time.time()

    for n, img in images.items():
        centers = vc.all_centers(vc.DEFAULT_CALIB, img.size)
        r = vc.hole_radius(img.size)
        side = args.crop * img.width / vc.BASE_W
        crops = {p: vc.crop_square(img, cx, cy, side) for p, (cx, cy) in centers.items()}
        feats = {p: vc.local_features(crops[p], r) for p in centers}
        guess = {p: vc.local_guess(f) for p, f in feats.items()}
        blocked = set(vc.blocked_positions(img, centers)) - set(vc.HIDDEN_POSITIONS)
        hidden = set(vc.HIDDEN_POSITIONS) | blocked
        to_send = [] if args.no_api else [p for p in centers if p not in hidden and guess[p] != "empty"]

        vlm = {}
        todo = []
        for p in to_send:
            k = f"{n}|{p}|{args.model}|{phash}|{args.crop}"
            if k in cache:
                vlm[p] = cache[k]
            else:
                todo.append((p, k))

        def work(item):
            p, k = item
            return p, k, vc.call_vlm(vc.crop_with_ring(crops[p], r), key, model=args.model, timeout=args.timeout)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(work, it) for it in todo]), 1):
                p, k, out = fut.result()
                vlm[p] = out
                if "error" not in out:
                    cache[k] = out
                print(f"  {n}  {i}/{len(todo)}  {p}: {out.get('label', 'ERROR ' + out.get('error', '')[:60])}", flush=True)
        # second look where the model contradicted a visible cap
        second = {}
        todo2 = []
        for p in vlm:
            prompt2 = vc.second_look_prompt(feats[p], vlm[p])
            if prompt2:
                k = f"{n}|{p}|{args.model}|verify-{vhashes[prompt2]}|{args.crop}"
                if k in cache:
                    second[p] = cache[k]
                else:
                    todo2.append((p, k, prompt2))

        def work2(item):
            p, k, prompt2 = item
            return p, k, vc.call_vlm(vc.crop_with_ring(crops[p], r), key, model=args.model, timeout=args.timeout,
                                     prompt=prompt2)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(work2, it) for it in todo2]):
                p, k, out = fut.result()
                second[p] = out
                if "error" not in out:
                    cache[k] = out
                print(f"  {n}  second look {p}: {out.get('label', 'ERROR')}", flush=True)
        cache_path.write_text(json.dumps(cache))

        results = {}
        for p in centers:
            if p in blocked:
                results[p] = vc.blocked_result(p, feats[p])
            elif p in hidden:
                results[p] = vc.hidden_result(p, feats[p])
            elif p in vlm:
                results[p] = vc.decide(p, feats[p], vlm[p], second=second.get(p))
            else:
                results[p] = vc.decide(p, feats[p], None)
        placed = counts.get(n)
        vc.apply_placement_count(results, placed)
        if not args.no_estimate:               # no robot count for a fixture -> estimate the fill from the image
            est = vc.estimate_placed(results)
            vc.apply_estimated_count(results, {fx: k for fx, k in est.items() if (placed or {}).get(fx) is None})
        vc.apply_fill_pattern(results, {k: (v or 0) for k, v in (placed or {}).items()})

        for p in to_send:
            buf = io.BytesIO()
            vc.crop_with_ring(crops[p], r).save(buf, format="PNG")
            crops_zip.writestr(f"{Path(n).stem}__{p}.png", buf.getvalue())
        report["images"][n] = {
            "seconds_model": round(time.time() - t0, 1), "sent": to_send, "blocked": sorted(blocked), "placed": placed,
            "positions": {p: {**asdict(results[p]), "guess": guess[p], "std": feats[p].std,
                              "vlm": vlm.get(p), "second": second.get(p)} for p in centers},
        }
        counts = {}
        for res in results.values():
            counts[res.status] = counts.get(res.status, 0) + 1
        print(f"{n}: {counts}  ({len(todo)} new model calls, {time.time() - t0:.0f}s)", flush=True)

    crops_zip.close()
    report["seconds_total"] = round(time.time() - t_all, 1)
    (run_dir / "report.json").write_text(json.dumps(report, indent=1))
    (HERE / "runs" / "LATEST.txt").write_text(run_dir.name)
    print(f"\nDONE in {time.time() - t_all:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
