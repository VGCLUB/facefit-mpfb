"""MediaPipe Face Landmarker (CPU) wrapper. Runs in a normal Python env (not Blender)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("FACEFIT_LANDMARKER", os.path.join(HERE, "..", "testdata", "face_landmarker.task"))

def detect(image_path, model=MODEL):
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model,
              delegate=BaseOptions.Delegate.CPU), num_faces=1)
    with vision.FaceLandmarker.create_from_options(opts) as det:
        img = mp.Image.create_from_file(image_path)
        res = det.detect(img)
    if not res.face_landmarks:
        return None
    w, h = img.width, img.height
    pts = [[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]]
    return {"image": os.path.abspath(image_path), "width": w, "height": h, "landmarks_px": pts}

if __name__ == "__main__":
    r = detect(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else None
    if r is None:
        print("NO FACE"); sys.exit(2)
    print("detected", len(r["landmarks_px"]), "landmarks")
    if out: json.dump(r, open(out, "w"))
