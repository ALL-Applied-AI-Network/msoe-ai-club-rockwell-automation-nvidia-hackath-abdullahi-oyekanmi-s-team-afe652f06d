"""
gen_synth.py - synthetic, automatically labelled CapSealCam images from the CSI digital twin (NVIDIA Isaac Sim).

For every frame it re-arranges the good fixture like the real line would:
  * fill count n (robot order 01..30); positions after n are empty (no_vial, waiting)
  * among placed positions, random defects:  missing_cap (cap removed), crooked_cap (cap tilted 12-40 deg and
    shifted, pivoting on its rim), missed placement (vial removed inside the pattern -> no_vial / gap)
  * leaning vials: some good vials tip 3-12 deg in their hole with the cap still seated (label stays pass)
  * lighting: both station lights scaled 0.55-1.45x and slightly warm/cool tinted
It renders the twin's own CapSealCam (path tracing, 768x576 with overscan), applies the real lens distortion
exactly like the twin's tools/apply_lens.py, and writes 640x480 PNGs plus a labels JSON.

Run with the Isaac Sim environment (run_synth.bat). Output: isaac/synth/ and isaac/synth.zip
"""
import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TWIN = r"C:\csi_twin\csi_digital_twin_v1"
STAGE = os.path.join(TWIN, "stations", "tray_inspection", "tray_inspection.usdc")
CAM = json.load(open(os.path.join(TWIN, "cameras", "CapSealCam.json")))
OUT = os.path.join(HERE, "synth")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60, help="number of frames")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--subframes", type=int, default=16, help="render samples per frame (quality vs speed)")
ap.add_argument("--p-missing", type=float, default=0.12)
ap.add_argument("--p-crooked", type=float, default=0.12)
ap.add_argument("--p-gap", type=float, default=0.05)
ap.add_argument("--p-lean", type=float, default=0.25, help="share of good vials that lean in their hole (cap seated)")
ap.add_argument("--lean-max", type=float, default=15.0, help="largest lean, degrees")
ap.add_argument("--cam-jitter", type=float, default=0.0,
                help="camera wobble in Isaac Sim (1 = up to ~4 mm and 0.5 deg). Off by default: moving the camera made the "
                     "NVIDIA driver crash every few frames on an RTX 50; learn_loop.py shifts the images instead")
ap.add_argument("--tilt-min", type=float, default=12.0, help="smallest crooked-cap tilt, degrees")
ap.add_argument("--tilt-max", type=float, default=40.0, help="largest crooked-cap tilt, degrees")
ap.add_argument("--tint", type=float, default=0.18, help="largest warm/cool light tint")
ap.add_argument("--renderer", default="RaytracedLighting", help="RaytracedLighting (stable, fast) or PathTracing")
ap.add_argument("--fresh", action="store_true", help="delete earlier images and start from frame 0")
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
STATUS = os.path.join(HERE, "synth_status.txt")
LABELS = os.path.join(OUT, "labels.json")
ATTEMPTS = os.path.join(OUT, "attempts.json")
if a.fresh:
    for fn in os.listdir(OUT):
        os.remove(os.path.join(OUT, fn))
    open(STATUS, "w").write("")
# Resume: if Isaac Sim crashed (it can, inside the NVIDIA driver), run_synth.bat starts it again and we carry on
# from the next frame. Each frame has its own random seed, so a restart gives exactly the same dataset.
labels = json.load(open(LABELS)) if os.path.exists(LABELS) else {}
attempts = json.load(open(ATTEMPTS)) if os.path.exists(ATTEMPTS) else {}
done = {int(k[6:10]) for k in labels} | {int(k) for k, v in attempts.items() if v >= 2}
start = max(done) + 1 if done else 0


def status(msg):
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)


if start >= a.n:
    status("RENDER_DONE"); sys.exit(0)
status(f"start n={a.n} seed={a.seed} subframes={a.subframes} renderer={a.renderer} from frame {start}")
rw, rh = CAM["render_px"]
try:
    from isaacsim import SimulationApp
except ImportError:
    from isaacsim.simulation_app import SimulationApp
# crash reporter off: after a driver crash its helper processes kept the log file locked, blocking restarts
kit = SimulationApp({"headless": True, "renderer": a.renderer, "width": rw, "height": rh,
                     "extra_args": ["--/crashreporter/enabled=false", "--/crashreporter/skipOldDumpUpload=true"]})

try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np
    from PIL import Image          # NOTE: no OpenCV inside Isaac Sim (it crashes Kit on Windows); lens is applied later
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, Usd, UsdGeom

    ctx = omni.usd.get_context()
    if not ctx.open_stage(STAGE):
        raise RuntimeError("could not open " + STAGE)
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    cam_path = CAM["prim_in_station_file"]
    if not stage.GetPrimAtPath(cam_path).IsValid():
        raise RuntimeError("camera not found: " + cam_path)
    status("stage open")

    # ---- objects we will change ----
    G = [f"G{i:02d}" for i in range(1, 31)]
    tube = {p: stage.GetPrimAtPath(f"/TrayInspection/{p}_tube") for p in G}
    cap = {p: stage.GetPrimAtPath(f"/TrayInspection/{p}_cap") for p in G}
    missing = [p for p in G if not (tube[p].IsValid() and cap[p].IsValid())]
    if missing:
        raise RuntimeError(f"missing tube/cap prims: {missing}")
    lights = [stage.GetPrimAtPath(p) for p in ("/TrayInspection/key_overhead/key_overhead",
                                               "/TrayInspection/fill_side/fill_side")]
    base_int = [l.GetAttribute("inputs:intensity").Get() for l in lights]
    base_col = [l.GetAttribute("inputs:color").Get() if l.GetAttribute("inputs:color") else None for l in lights]

    # All edits go to the session layer, so the twin's files are never modified.
    stage.SetEditTarget(stage.GetSessionLayer())
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    tube_local0, tube_op = {}, {}
    for p in G:
        xf = UsdGeom.Xformable(tube[p])
        m0 = xf.GetLocalTransformation()
        xf.ClearXformOpOrder()
        tube_op[p] = xf.AddTransformOp(opSuffix="synth")
        tube_op[p].Set(m0)
        tube_local0[p] = m0
    tparent_w = {p: xc.GetParentToWorldTransform(tube[p]) for p in G}
    tube_base = {}
    for p in G:
        tb = bc.ComputeWorldBound(tube[p]).ComputeAlignedRange()
        tube_base[p] = Gf.Vec3d((tb.GetMin()[0] + tb.GetMax()[0]) / 2, (tb.GetMin()[1] + tb.GetMax()[1]) / 2, tb.GetMin()[2])
    cap_local0, cap_op, cap_info = {}, {}, {}
    for p in G:
        xf = UsdGeom.Xformable(cap[p])
        m0 = xf.GetLocalTransformation()
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp(opSuffix="synth")
        op.Set(m0)
        cap_local0[p], cap_op[p] = m0, op
        box = bc.ComputeWorldBound(cap[p]).ComputeAlignedRange()
        mn, mx = box.GetMin(), box.GetMax()
        cap_info[p] = dict(center=Gf.Vec3d((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]),
                           radius=max(mx[0] - mn[0], mx[1] - mn[1]) / 2)
    parent_w = {p: xc.GetParentToWorldTransform(cap[p]) for p in G}

    AWAY = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, -50))      # 50 m below the floor = out of the camera view

    def remove_cap(p):
        cap_op[p].Set(cap_local0[p] * parent_w[p] * AWAY * parent_w[p].GetInverse())

    def remove_vial(p):
        remove_cap(p)
        tube_op[p].Set(tube_local0[p] * tparent_w[p] * AWAY * tparent_w[p].GetInverse())

    def tilt_cap(p, deg, az):
        """Rotate the cap about a horizontal axis through a point on its bottom rim (it hinges on the vial lip),
        plus a small slide - in world space, then back to the cap's local space."""
        info = cap_info[p]
        ax = Gf.Vec3d(-math.sin(az), math.cos(az), 0)          # horizontal axis
        rim = info["center"] + Gf.Vec3d(math.cos(az), math.sin(az), 0) * info["radius"]
        R = Gf.Matrix4d().SetRotate(Gf.Rotation(ax, deg))
        slide = Gf.Matrix4d().SetTranslate(Gf.Vec3d(math.cos(az), math.sin(az), 0) * info["radius"] * rng.uniform(0.1, 0.45))
        T1 = Gf.Matrix4d().SetTranslate(-rim)
        T2 = Gf.Matrix4d().SetTranslate(rim)
        world0 = cap_local0[p] * parent_w[p]
        world1 = world0 * T1 * R * T2 * slide
        cap_op[p].Set(world1 * parent_w[p].GetInverse())

    def lean_vial(p, deg, az):
        """Tip the whole vial (tube + its seated cap) about the bottom of the tube - a leaning vial, still good."""
        ax = Gf.Vec3d(-math.sin(az), math.cos(az), 0)
        M = Gf.Matrix4d().SetTranslate(-tube_base[p]) * Gf.Matrix4d().SetRotate(Gf.Rotation(ax, deg)) * \
            Gf.Matrix4d().SetTranslate(tube_base[p])
        tube_op[p].Set(tube_local0[p] * tparent_w[p] * M * tparent_w[p].GetInverse())
        cap_op[p].Set(cap_local0[p] * parent_w[p] * M * parent_w[p].GetInverse())

    def reset():
        for p in G:
            tube_op[p].Set(tube_local0[p])
            cap_op[p].Set(cap_local0[p])

    # ---- renderer ----
    rp = rep.create.render_product(cam_path, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()
    # same RTX clipping-range re-sync workaround as the twin's render_example.py
    clip = UsdGeom.Camera(stage.GetPrimAtPath(cam_path)).GetClippingRangeAttr()
    nf = clip.Get()
    clip.Set(Gf.Vec2f(nf[0] * 0.5, nf[1]))
    kit.update()
    clip.Set(nf)
    kit.update()

    # camera wobble: a fixed camera still gets knocked, re-mounted or drifts a little - the grid has to cope
    camx = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
    cam_local0 = camx.GetLocalTransformation()
    camx.ClearXformOpOrder()
    cam_op = camx.AddTransformOp(opSuffix="synth")
    cam_op.Set(cam_local0)
    cam_parent = xc.GetParentToWorldTransform(stage.GetPrimAtPath(cam_path))
    cam_world0 = cam_local0 * cam_parent
    cam_pos0 = cam_world0.ExtractTranslation()
    kit.update()

    def wobble_camera(rng):
        if a.cam_jitter <= 0:
            return [0, 0, 0, 0]
        d = Gf.Vec3d(*(rng.uniform(-0.004, 0.004) * a.cam_jitter for _ in range(3)))
        ax = Gf.Vec3d(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
        ang = rng.uniform(0, 0.5) * a.cam_jitter
        R = Gf.Matrix4d().SetRotate(Gf.Rotation(ax.GetNormalized(), ang))
        M = Gf.Matrix4d().SetTranslate(-cam_pos0) * R * Gf.Matrix4d().SetTranslate(cam_pos0 + d)
        cam_op.Set(cam_world0 * M * cam_parent.GetInverse())
        return [round(d[0] * 1000, 1), round(d[1] * 1000, 1), round(d[2] * 1000, 1), round(ang, 2)]

    def render():
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
        return np.asarray(ann.get_data())[..., :3].copy()          # RGB, 768x576 pinhole (overscan)

    t0 = time.time()
    rendered = 0
    for i in range(start, a.n):
        if attempts.get(str(i), 0) >= 2:
            status(f"frame {i} crashed Isaac Sim twice - skipped"); continue
        attempts[str(i)] = attempts.get(str(i), 0) + 1
        json.dump(attempts, open(ATTEMPTS, "w"))
        rng = random.Random(a.seed * 100003 + i)
        reset()
        # frame 0 = the twin exactly as built (all 30 capped) for a side-by-side check
        n = 30 if i == 0 else rng.choice([rng.randint(0, 30), rng.randint(10, 30), 30])
        lab, meta = {}, {}
        for k, p in enumerate(G, start=1):
            if k > n:
                remove_vial(p)
                lab[p] = "no_vial"; continue
            r = rng.random() if i else 1.0
            if r < a.p_gap and k < n:
                remove_vial(p)
                lab[p] = "no_vial"; meta[p] = "missed_placement"
            elif r < a.p_gap + a.p_missing:
                remove_cap(p)
                lab[p] = "missing_cap"
            elif r < a.p_gap + a.p_missing + a.p_crooked:
                deg = rng.uniform(a.tilt_min, a.tilt_max)
                tilt_cap(p, deg, rng.uniform(0, 2 * math.pi))
                lab[p] = "crooked_cap"; meta[p] = f"tilt {deg:.0f} deg"
            else:
                lab[p] = "pass"
                if i and rng.random() < a.p_lean:
                    deg = rng.uniform(3, a.lean_max)
                    lean_vial(p, deg, rng.uniform(0, 2 * math.pi))
                    meta[p] = f"lean {deg:.0f} deg"
        # lighting
        li = []
        for l, b, c in zip(lights, base_int, base_col):
            s = 1.0 if i == 0 else rng.uniform(0.55, 1.45)
            l.GetAttribute("inputs:intensity").Set(b * s)
            if c is not None:
                warm = 0.0 if i == 0 else rng.uniform(-0.6 * a.tint, a.tint)
                l.GetAttribute("inputs:color").Set(Gf.Vec3f(min(1, c[0] * (1 + warm)), c[1], min(1, c[2] * (1 - warm))))
            li.append(round(s, 2))
        cam = [0, 0, 0, 0] if i == 0 else wobble_camera(rng)
        if i == 0:
            cam_op.Set(cam_local0)
        kit.update()
        img = render()
        name = f"synth_{i:04d}.png"
        Image.fromarray(img).save(os.path.join(OUT, "raw_" + name))
        labels[name] = {"labels": lab, "placed_G": n, "lights": li, "notes": meta, "camera_mm_deg": cam}
        json.dump(labels, open(LABELS, "w"), indent=1)
        rendered += 1
        status(f"frame {i + 1}/{a.n}  placed={n}  {(time.time() - t0) / rendered:.1f}s/frame")

    status("RENDER_DONE")
    # lens + zip in a separate plain-Python process (OpenCV must not load inside Isaac Sim)
    import subprocess
    subprocess.run([sys.executable, os.path.join(HERE, "finalize_synth.py")], cwd=HERE)
except BaseException:
    status("ERROR\n" + traceback.format_exc())
finally:
    sys.stdout.flush()
    os._exit(0)          # skip Kit's shutdown (its TaskGroup assertion popup) - everything is already saved
