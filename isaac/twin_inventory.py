"""
twin_inventory.py - open the CSI digital twin's tray-inspection station and write down what is in it:
the prim tree, every vial / cap / tray-like object, its world position, and where it lands in the
real 640x480 cap-seal camera image (using the twin's own CapSealCam calibration + lens model).
It also maps each vial to the app's numbered positions (G01-G30, S01-S30).

Run with the Isaac Sim environment (run_inventory.bat does this). No rendering - just reads the scene.
Writes twin_inventory.txt and twin_inventory.json next to this file.
"""
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TWIN = r"C:\csi_twin\csi_digital_twin_v1"
STAGE = os.path.join(TWIN, "stations", "tray_inspection", "tray_inspection.usdc")
CAM = json.load(open(os.path.join(TWIN, "cameras", "CapSealCam.json")))
OUT_TXT = os.path.join(HERE, "twin_inventory.txt")
OUT_JSON = os.path.join(HERE, "twin_inventory.json")

# the app's default grid (640x480 pixels): corner hole centres of each fixture's 6x5 grid
CALIB = {
    "G": {"TL": (228, 166), "TR": (378, 163), "BR": (382, 305), "BL": (232, 304)},
    "S": {"TL": (421, 259), "TR": (556, 252), "BR": (556, 380), "BL": (424, 392)},
}


def grid_centres():
    out = {}
    for fx, c in CALIB.items():
        for col in range(6):
            for row in range(5):
                u, v = col / 5.0, row / 4.0
                tx = c["TL"][0] * (1 - u) + c["TR"][0] * u
                ty = c["TL"][1] * (1 - u) + c["TR"][1] * u
                bx = c["BL"][0] * (1 - u) + c["BR"][0] * u
                by = c["BL"][1] * (1 - u) + c["BR"][1] * u
                n = (5 - col) * 5 + row + 1
                out[f"{fx}{n:02d}"] = (tx * (1 - v) + bx * v, ty * (1 - v) + by * v)
    return out


def project(p):
    """World point (m) -> final 640x480 pixel (pinhole + the twin's k1 lens distortion), or None if behind."""
    R, C = CAM["R_world_to_cv"], CAM["position_m"]
    d = [p[i] - C[i] for i in range(3)]
    x, y, z = (sum(R[r][i] * d[i] for i in range(3)) for r in range(3))
    if z <= 1e-6:
        return None
    xn, yn = x / z, y / z
    k = 1 + CAM["lens_k1"] * (xn * xn + yn * yn)
    W, H = CAM["final_image_px"]
    f = CAM["focal_px_final"]
    return (f * xn * k + W / 2, f * yn * k + H / 2)


def main():
    log = []
    t0 = time.time()
    try:
        try:
            from isaacsim import SimulationApp
        except ImportError:
            from isaacsim.simulation_app import SimulationApp
        app = SimulationApp({"headless": True})
    except Exception:
        open(OUT_TXT, "w").write("FAILED TO START ISAAC SIM\n" + traceback.format_exc())
        raise
    try:
        from pxr import Usd, UsdGeom, UsdShade, UsdLux
        stage = Usd.Stage.Open(STAGE)
        log.append(f"stage: {STAGE}  opened in {time.time() - t0:.0f}s")
        log.append(f"metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)} upAxis={UsdGeom.GetStageUpAxis(stage)}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # 1) tree overview (depth <= 6, collapse long lists of siblings)
        log.append("\n== PRIM TREE (depth<=6) ==")
        def walk(prim, depth):
            if depth > 6:
                return
            kids = list(prim.GetChildren())
            log.append(f"{'  ' * depth}{prim.GetName()}  [{prim.GetTypeName()}]  children={len(kids)}")
            shown = kids if len(kids) <= 12 else kids[:6] + kids[-2:]
            for k in shown:
                walk(k, depth + 1)
            if len(kids) > 12:
                log.append(f"{'  ' * (depth + 1)}... {len(kids) - 8} more like {kids[6].GetName()}")
        walk(stage.GetPseudoRoot(), 0)

        # 2) interesting objects
        keys = ("vial", "cap", "lid", "tube", "bottle", "tray", "fixture", "plate", "rack", "hole", "seal", "glass")
        objs = []
        types = {}
        for prim in stage.Traverse():
            types[prim.GetTypeName()] = types.get(prim.GetTypeName(), 0) + 1
            name = prim.GetName().lower()
            if not any(k in name for k in keys) or not prim.IsA(UsdGeom.Imageable):
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            mn, mx = box.GetMin(), box.GetMax()
            ctr = [(mn[i] + mx[i]) / 2 for i in range(3)]
            size = [mx[i] - mn[i] for i in range(3)]
            top = [ctr[0], ctr[1], mx[2]]
            px = project(top)
            mat = ""
            try:
                m, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if m:
                    mat = m.GetPath().pathString
            except Exception:
                pass
            objs.append({"path": prim.GetPath().pathString, "type": prim.GetTypeName(),
                         "center_m": [round(v, 4) for v in ctr], "size_m": [round(v, 4) for v in size],
                         "top_px": None if px is None else [round(px[0], 1), round(px[1], 1)],
                         "material": mat, "children": len(prim.GetChildren())})
        log.append("\n== PRIM TYPES ==")
        for t, n in sorted(types.items(), key=lambda x: -x[1])[:25]:
            log.append(f"{n:6d}  {t}")

        # 3) map vial-like objects to app positions
        grid = grid_centres()
        log.append(f"\n== OBJECTS MATCHING {keys} ({len(objs)}) ==")
        for o in objs[:400]:
            near = ""
            if o["top_px"]:
                pid, d = min(((p, math.dist(o["top_px"], c)) for p, c in grid.items()), key=lambda x: x[1])
                near = f"  nearest {pid} at {d:.1f}px"
                o["nearest"], o["nearest_px"] = pid, round(d, 1)
            log.append(f"{o['path']}  [{o['type']}]  size_m={o['size_m']}  top_px={o['top_px']}{near}  mat={o['material']}")

        # 3b) vial-sized meshes anywhere (names in CAD exports are often generic): small round-ish things
        #     whose top lands near a fixture position
        sized = []
        for prim in stage.Traverse():
            if prim.GetTypeName() not in ("Mesh", "Cylinder", "Capsule"):
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            mn, mx = box.GetMin(), box.GetMax()
            sx, sy, sz = mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
            if not (0.008 <= sx <= 0.045 and 0.008 <= sy <= 0.045):
                continue
            px = project([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mx[2]])
            if px is None:
                continue
            pid, d = min(((q, math.dist(px, c)) for q, c in grid.items()), key=lambda x: x[1])
            if d > 25:
                continue
            sized.append((pid, d, prim.GetPath().pathString, round(sx, 4), round(sy, 4), round(sz, 4), [round(px[0], 1), round(px[1], 1)]))
        sized.sort()
        log.append(f"\n== VIAL-SIZED MESHES NEAR A POSITION ({len(sized)}) ==")
        for pid, d, path, sx, sy, sz, px in sized[:500]:
            log.append(f"{pid}  {d:5.1f}px  size=({sx},{sy},{sz})  px={px}  {path}")

        # 4) lights
        log.append("\n== LIGHTS ==")
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI) or "Light" in prim.GetTypeName():
                a = prim.GetAttribute("inputs:intensity")
                log.append(f"{prim.GetPath()}  [{prim.GetTypeName()}]  intensity={a.Get() if a else None}")

        json.dump({"stage": STAGE, "objects": objs, "sized": sized, "grid": grid}, open(OUT_JSON, "w"), indent=1)
        log.append(f"\nDONE in {time.time() - t0:.0f}s")
    except Exception:
        log.append("ERROR:\n" + traceback.format_exc())
    finally:
        open(OUT_TXT, "w", encoding="utf-8").write("\n".join(log))
        print("wrote", OUT_TXT, flush=True)
        app.close()


if __name__ == "__main__":
    main()
