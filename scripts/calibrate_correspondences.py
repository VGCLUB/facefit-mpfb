"""One-off: derive MediaPipe-landmark -> MPFB basemesh vertex correspondences.
Renders the default MPFB human (front ortho), runs MediaPipe on the render, and ray-casts
each selected landmark pixel onto the mesh to find the nearest (non-helper) vertex.
Requires a running Blender bridge. Writes data/mp_to_mpfb.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facefit.client import BlenderClient
from facefit.landmarks import detect
from facefit import landmark_sets as LS

RES = 1024
c = BlenderClient()
c.call("reset_parameters")
img = os.path.abspath("out/calib_default.png")
c.call("render", path=img, res=RES)
r = detect(img)
assert r, "MediaPipe failed on default render"
px = [r["landmarks_px"][i][:2] for i in LS.ALL]
vids = c.call("raycast_pixels", pixels=px, res=RES)
proj = c.call("project_vertices", vertex_ids=[v for v in vids if v >= 0], res=RES)
import numpy as np
ok = [(i, v) for i, v in zip(LS.ALL, vids) if v >= 0]
err = np.linalg.norm(np.array(proj) - np.array([r["landmarks_px"][i][:2] for i, _ in ok]), axis=1)
print("mapped", len(ok), "/", len(LS.ALL), "reprojection px mean %.2f max %.2f" % (err.mean(), err.max()))
json.dump({"source": "auto-calibrated on default MPFB2 human render (MediaPipe + raycast)",
           "pairs": [{"mp": i, "vertex": v} for i, v in ok]}, open("data/mp_to_mpfb.json", "w"), indent=1)
