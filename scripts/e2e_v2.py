"""Phase-2 end-to-end: fit (2D + depth) -> texture projection -> textured renders. Needs the Blender bridge."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facefit.session import FaceFitSession

img = sys.argv[1] if len(sys.argv) > 1 else "testdata/guoman_face.png"
S = FaceFitSession(); O = S.workdir
S.bl.call("create_human")          # fresh default human (removes old eyes/materials state)
S.load_reference_image(img); S.detect_landmarks()
fit2d = S.auto_fit(depth_weight=0.0)
S.bl.call("reset_parameters")
fit3d = S.auto_fit(depth_weight=0.3)
tex = S.project_texture()
S.render_textured(os.path.join(O, "v2_textured_front.png"), yaw_deg=0, samples=48)
S.render_textured(os.path.join(O, "v2_textured_34.png"), yaw_deg=35, samples=48)
S.render_view(os.path.join(O, "v2_fit_overlay.png"), overlay=True, markers=True)
for f in ("glb", "obj"): S.export_model(os.path.join(O, f"v2_fitted.{f}"))
def slim(e): return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k in ("rms_px", "mean_px", "max_px", "rms_iod_pct", "depth_rms_iod_pct")}
rep = {"default_head": slim(fit3d["error_before"]), "fit_2d_only": slim(fit2d["error_after"]),
       "fit_2d_plus_depth": slim(fit3d["error_after"]),
       "depth_params": {k: v for k, v in fit3d["parameters"].items() if k in __import__("facefit.fit", fromlist=["x"]).DEPTH_FIT_PARAMS},
       "texture": tex}
json.dump(rep, open(os.path.join(O, "report_v2.json"), "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("default_head", "fit_2d_only", "fit_2d_plus_depth", "depth_params")}, indent=1))
print("eyes:", tex["eyes"], "visible tris:", tex["files"]["human"]["visible_triangles"], "/", tex["files"]["human"]["triangles"])
