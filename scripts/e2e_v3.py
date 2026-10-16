"""Phase-3 end-to-end (likeness): macros -> sliders(+depth) -> vertex refine -> projected face texture -> hair cap -> renders."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facefit.session import FaceFitSession
img = sys.argv[1] if len(sys.argv) > 1 else "testdata/guoman_face.png"
style = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
S = FaceFitSession(); O = S.workdir
S.bl.call("create_human")
S.load_reference_image(img); S.detect_landmarks()
S.render_view(os.path.join(O, "v3_default_clay.png"))
default_err = S.get_landmark_error()
mac = S.fit_macros()
ref = S.vertex_refine()
styles = S.add_style_keys()
if style: S.set_style(style)
S.render_view(os.path.join(O, "v3_clay_front.png"))
S.render_view(os.path.join(O, "v3_fit_overlay.png"), overlay=True, markers=True)
err_final = S.get_landmark_error()
tex = S.project_face()
S.render_textured(os.path.join(O, "v3_front.png"), yaw_deg=0, samples=64)
S.render_textured(os.path.join(O, "v3_34.png"), yaw_deg=35, samples=64)
val = S.mediapipe_validation(os.path.join(O, "v3_front.png"))
S.save_state("v3")
S.export_model(os.path.join(O, "v3_fitted.glb"))
slim = lambda e: {k: round(v, 3) for k, v in e.items() if k in ("rms_px", "rms_iod_pct", "max_px", "depth_rms_iod_pct") and v is not None}
rep = dict(default=slim(default_err), macros=mac["best"], after_macros_sliders=slim(mac["fit_error"]),
           after_vertex_refine=slim(ref["error_after"]), final=slim(err_final), refine_passes=ref["passes"],
           mediapipe_on_v3_front_render=val, style=style, texture=tex)
json.dump(rep, open(os.path.join(O, "report_v3.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "texture"}, indent=1)); print(tex)
# stylised variant (optional look-dev; not used for the error numbers)
S.set_style({"style_big_eyes": 0.5, "style_v_chin": 0.4, "style_small_nose": 0.3, "style_full_lips": 0.3})
S.project_face()
S.render_textured(os.path.join(O, "v3_front_styled.png"), yaw_deg=0, samples=64)
print("styled error:", slim(S.get_landmark_error()))
S.set_style({"style_big_eyes": 0, "style_v_chin": 0, "style_small_nose": 0, "style_full_lips": 0})
# compare: reference | v2 | v3 (front)
from PIL import Image, ImageDraw
H = 700
def load(p):
    im = Image.open(p).convert("RGB"); return im.resize((int(im.width * H / im.height), H))
tiles = [load(img), load(os.path.join(O, "v2_textured_front.png")), load(os.path.join(O, "v3_front.png"))]
cmp_ = Image.new("RGB", (sum(t.width for t in tiles), H)); x = 0
for t, lab in zip(tiles, ["reference", "v2", "v3"]):
    cmp_.paste(t, (x, 0)); ImageDraw.Draw(cmp_).text((x + 10, 10), lab, fill=(255, 255, 255)); x += t.width
cmp_.save(os.path.join(O, "v3_compare.png"))
