"""Bald / full-head reference: macros -> sliders -> full-head vertex refine -> photo projection (no hair cap)
-> eye calibration (iris size/centre from MediaPipe iris) -> soft-light renders + compare sheet."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facefit.session import FaceFitSession
from PIL import Image, ImageDraw
img = sys.argv[1] if len(sys.argv) > 1 else "testdata/bald_face.jpg"
tag = sys.argv[2] if len(sys.argv) > 2 else "bald"
S = FaceFitSession(); O = S.workdir
S.bl.call("create_human")
S.load_reference_image(img); S.detect_landmarks()
default_err = S.get_landmark_error()
mac = S.fit_macros(candidates=[dict(gender=0.0, age=0.5, asian=0.5, proportions=0.5, weight=0.5)] if os.environ.get('FAST') else None)
S.set_parameters({"ear-flap-decr-incr": 1.0})   # heuristic: frontal photo shows ears face-on (protruding)
ref = S.vertex_refine()
S.add_style_keys()
S.bl.call("reframe")   # skull may have grown: re-centre camera on the new head top
tex = S.project_face(hair_cap=False)
eyes = S.calibrate_eyes(rounds=2)
S.render_view(os.path.join(O, f"{tag}_clay_front.png"))
S.render_view(os.path.join(O, f"{tag}_overlay.png"), overlay=True, markers=True)
S.render_textured(os.path.join(O, f"{tag}_front.png"), yaw_deg=0, samples=64)
S.render_textured(os.path.join(O, f"{tag}_34.png"), yaw_deg=35, samples=64)
val = S.mediapipe_validation(os.path.join(O, f"{tag}_front.png"))
iou = S.silhouette_iou(os.path.join(O, f"{tag}_silhouette.png"))
S.save_state(tag)
H = 800
def load(p):
    im = Image.open(p).convert("RGB"); return im.resize((int(im.width * H / im.height), H))
wf = S.warp_to_ref(os.path.join(O, f"{tag}_front.png"), os.path.join(O, f"_{tag}_front_warp.png"))
wc = S.warp_to_ref(os.path.join(O, f"{tag}_clay_front.png"), os.path.join(O, f"_{tag}_clay_warp.png"))
tiles = [load(img), load(wf), load(wc)]
sheet = Image.new("RGB", (sum(t.width for t in tiles), H)); x = 0
for t, lab in zip(tiles, ["reference", "result (front)", "result (clay)"]):
    sheet.paste(t, (x, 0)); ImageDraw.Draw(sheet).text((x + 10, 10), lab, fill=(255, 255, 255)); x += t.width
sheet.save(os.path.join(O, f"{tag}_compare.png"))
slim = lambda e: {k: round(v, 3) for k, v in e.items() if k in ("rms_px", "rms_iod_pct", "max_px", "depth_rms_iod_pct") and v is not None}
rep = dict(default=slim(default_err), macros=mac["best"], after_macros_sliders=slim(mac["fit_error"]),
           after_vertex_refine=slim(ref["error_after"]), full_head=ref["full_head"], refine_passes=ref["passes"],
           eyes=eyes, mediapipe_on_front_render=val, silhouette_iou_above_chin=iou, texture={k: v for k, v in tex.items()})
json.dump(rep, open(os.path.join(O, f"report_{tag}.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in rep.items() if k != "texture"}, indent=1, default=str))
