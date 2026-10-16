"""End-to-end headless test. Needs the Blender bridge running (scripts/run_blender_headless.sh)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np
from facefit.session import FaceFitSession

img = sys.argv[1] if len(sys.argv) > 1 else "testdata/guoman_face.png"
S = FaceFitSession(); O = S.workdir
S.bl.call("reset_parameters")
S.load_reference_image(img); print("landmarks:", S.detect_landmarks())
S.render_view(os.path.join(O, "before.png"))
S.render_view(os.path.join(O, "before_overlay.png"), overlay=True, markers=True)
err_b = S.get_landmark_error(); val_b = S.mediapipe_validation(os.path.join(O, "before.png"))
fit = S.auto_fit()
S.render_view(os.path.join(O, "after.png"))
S.render_view(os.path.join(O, "after_overlay.png"), overlay=True, markers=True)
val_a = S.mediapipe_validation(os.path.join(O, "after.png"))
S.save_state("fitted")
for f in ("glb", "obj", "stl"): S.export_model(os.path.join(O, f"fitted.{f}"))
# side-by-side: reference | after (aligned overlay 50% + markers) | before overlay
ref = cv2.imread(S.ref_image); H = ref.shape[0]
tiles = [ref, cv2.imread(os.path.join(O, "before_overlay.png")), cv2.imread(os.path.join(O, "after_overlay.png"))]
labels = ["reference", "before (50% overlay)", "after fit (50% overlay)"]
for t, l in zip(tiles, labels): cv2.putText(t, l, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.imwrite(os.path.join(O, "overlay.png"), np.hstack(tiles))
rep = dict(error_before=err_b, error_after=fit["error_after"], mediapipe_on_render_before=val_b,
           mediapipe_on_render_after=val_a, fit=fit)
for e in (rep["error_before"], rep["error_after"]): e.pop("T_model_to_ref", None)
json.dump(rep, open(os.path.join(O, "report.json"), "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("error_before", "error_after", "mediapipe_on_render_before", "mediapipe_on_render_after")}, indent=1))
print("params:", json.dumps(fit["parameters"]))
