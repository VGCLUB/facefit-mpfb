"""Face parsing (BiSeNet ResNet-34, CelebAMask-HQ 19 classes) via onnxruntime on CPU.
Model: facefusion-assets models-3.0.0/bisenet_resnet_34.onnx (MIT-licensed weights distribution)."""
import os
import numpy as np, cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.environ.get("FACEFIT_PARSER", os.path.join(ROOT, "models", "bisenet_resnet_34.onnx"))
MODEL_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx"
CLASSES = ["background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear", "ear_r",
           "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l", "cloth", "hair", "hat"]
REGIONS = {"skin": [1, 10, 7, 8, 14], "brows": [2, 3], "eyes": [4, 5], "lips": [12, 13], "mouth": [11], "hair": [17, 18]}


def parse(image_bgr, landmarks_px=None, size=512, scale=1.6, shift_up=0.0):
    import onnxruntime as ort
    H, W = image_bgr.shape[:2]
    if landmarks_px is not None:  # square crop around the face
        p = np.asarray(landmarks_px)[:, :2]
        c = (p.min(0) + p.max(0)) / 2; s = (p.max(0) - p.min(0)).max() * scale; c[1] -= shift_up * s
        x0, y0 = int(max(c[0] - s / 2, 0)), int(max(c[1] - s / 2, 0))
        x1, y1 = int(min(c[0] + s / 2, W)), int(min(c[1] + s / 2, H))
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    crop = image_bgr[y0:y1, x0:x1]
    x = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (size, size)).astype(np.float32) / 255
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    lab = out.argmax(0).astype(np.uint8)
    lab = cv2.resize(lab, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = lab
    return full


def region_masks(labels):
    return {k: np.isin(labels, v).astype(np.uint8) * 255 for k, v in REGIONS.items()}


def colorize(labels):
    rng = np.random.RandomState(3); pal = rng.randint(0, 255, (19, 3)); pal[0] = 0
    return pal[labels].astype(np.uint8)
