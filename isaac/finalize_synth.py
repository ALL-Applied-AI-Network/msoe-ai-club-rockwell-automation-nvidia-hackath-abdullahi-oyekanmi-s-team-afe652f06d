"""finalize_synth.py - turn the raw 768x576 pinhole renders into real-camera 640x480 images (the twin's own lens
model, same maths as csi_digital_twin_v1/tools/apply_lens.py) and pack everything into synth.zip.
Runs as a normal Python process (no Isaac Sim), so OpenCV is safe here."""
import json, os, sys, time, zipfile
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "synth")
CAM = json.load(open(r"C:\csi_twin\csi_digital_twin_v1\cameras\CapSealCam.json"))
W, H = CAM["final_image_px"]
rw, rh = CAM["render_px"]
f, k1 = CAM["focal_px_final"], CAM["lens_k1"]
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
und = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1)[:, None, :], K, np.array([k1, 0, 0, 0, 0.0])).reshape(H, W, 2)
mapx = (und[..., 0] * f + rw / 2).astype(np.float32)
mapy = (und[..., 1] * f + rh / 2).astype(np.float32)

raws = sorted(fn for fn in os.listdir(OUT) if fn.startswith("raw_") and fn.endswith(".png"))
for fn in raws:
    img = cv2.imread(os.path.join(OUT, fn), cv2.IMREAD_COLOR)
    if img is None or img.shape[:2] != (rh, rw):
        print("skip", fn); continue
    cv2.imwrite(os.path.join(OUT, fn[4:]), cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
    os.remove(os.path.join(OUT, fn))
with zipfile.ZipFile(os.path.join(HERE, "synth.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("synth_") or fn == "labels.json":
            z.write(os.path.join(OUT, fn), fn)
with open(os.path.join(HERE, "synth_status.txt"), "a") as s:
    s.write(time.strftime("%H:%M:%S ") + f"SYNTH_DONE ({len(raws)} frames lens-corrected, synth.zip written)\n")
print("SYNTH_DONE", len(raws))
