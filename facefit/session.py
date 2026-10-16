"""High-level FaceFit session used by the MCP server and the e2e test."""
import json, os, time, copy
import numpy as np
from .client import BlenderClient
from . import fit as F, landmark_sets as LS
from .landmarks import detect

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORR = os.path.join(ROOT, "data", "mp_to_mpfb.json")


class FaceFitSession:
    def __init__(self, host="127.0.0.1", port=9876, workdir=None):
        self.bl = BlenderClient(host, port)
        self.workdir = os.path.abspath(workdir or os.path.join(ROOT, "out"))
        os.makedirs(self.workdir, exist_ok=True)
        pairs = json.load(open(CORR))["pairs"]
        self.mp_ids = [p["mp"] for p in pairs]
        self.vids = [p["vertex"] for p in pairs]
        self.weights = np.array([LS.weight_of(i) for i in self.mp_ids])
        F.set_index(self.mp_ids)
        self.ref_image = None; self.ref = None
        self.history = []; self.saved = {}

    # ---------------- reference
    def load_reference_image(self, path):
        from PIL import Image
        path = os.path.abspath(path)
        w, h = Image.open(path).size
        self.ref_image, self.ref = path, None
        return dict(path=path, width=w, height=h)

    def detect_landmarks(self, path=None):
        if path: self.load_reference_image(path)
        if not self.ref_image: raise ValueError("load_reference_image first")
        r = detect(self.ref_image)
        if r is None: raise RuntimeError("MediaPipe found no face in the reference")
        self.ref = r
        p = np.array(r["landmarks_px"])[:, :2]
        out = os.path.join(self.workdir, "ref_landmarks.json"); json.dump(r, open(out, "w"))
        return dict(count=len(p), inter_ocular_px=self.ref_iod(), json=out,
                    used_for_fit=len(self.mp_ids))

    def ref2d(self):
        if self.ref is None: self.detect_landmarks()
        return np.array(self.ref["landmarks_px"])[self.mp_ids, :2]

    def ref_iod(self):
        p = np.array(self.ref["landmarks_px"])
        return float(np.linalg.norm(p[F.L_EYE_OUT, :2] - p[F.R_EYE_OUT, :2]))

    # ---------------- parameters
    def list_parameters(self):
        return self.bl.call("list_parameters")

    def get_parameters(self, nonzero_only=True):
        v = self.bl.call("get_parameters")
        return {k: x for k, x in v.items() if abs(x) > 1e-4} if nonzero_only else v

    def push_undo(self):
        self.history.append(self.bl.call("get_parameters"))

    def set_parameters(self, values, record=True):
        if record: self.push_undo()
        return self.bl.call("set_parameters", values=values)

    def undo(self):
        if not self.history: return dict(undone=False)
        vals = self.history.pop()
        self.bl.call("set_parameters", values=vals)
        return dict(undone=True, remaining=len(self.history))

    def save_state(self, name="default", path=None):
        vals = self.bl.call("get_parameters")
        self.saved[name] = vals
        path = path or os.path.join(self.workdir, f"state_{name}.json")
        json.dump(dict(parameters=vals, reference=self.ref_image), open(path, "w"), indent=1)
        return dict(name=name, path=path)

    def load_state(self, name="default", path=None):
        vals = self.saved.get(name) if path is None else json.load(open(path))["parameters"]
        if vals is None: raise KeyError(name)
        return self.set_parameters(vals)

    # ---------------- error / fit
    def model2d(self):
        P = self.bl.call("project_vertices", vertex_ids=self.vids)
        return np.array(P)

    def get_landmark_error(self):
        e = F.landmark_error(self.model2d(), self.ref2d(), self.weights, self.ref_iod())
        e.pop("per_point_px"); return e

    def auto_fit(self, params=None, reg=0.005, style=None, locked=None, iters=15):
        """Fit sliders to the reference. style: offsets added after fitting (e.g. {'eye-scale-decr-incr': 0.2})."""
        params = params or F.DEFAULT_FIT_PARAMS
        avail = {p["name"] for p in self.list_parameters()}
        params = [p for p in params if p in avail]
        self.push_undo()
        before = self.get_landmark_error()
        lm = self.bl.call("linear_model", vertex_ids=self.vids, param_names=params)
        model = F.LinearFaceModel(lm)
        cur = self.bl.call("get_parameters", names=params)
        t = time.time()
        vals, hist = F.solve(model, self.ref2d(), self.weights, w0=model.w_from_values(cur), reg=reg,
                             iters=iters, locked=locked)
        for k, dv in (style or {}).items():
            vals[k] = float(np.clip(vals.get(k, 0.0) + dv, -1, 1))
        self.bl.call("set_parameters", values=vals)
        after = self.get_landmark_error()
        return dict(error_before=before, error_after=after, solve_seconds=round(time.time() - t, 2),
                    iterations=len(hist), objective_history=hist,
                    parameters={k: v for k, v in vals.items() if abs(v) > 1e-3})

    # ---------------- render
    def render_view(self, path=None, overlay=False, markers=False, res=768, opacity=0.5):
        import cv2
        path = os.path.abspath(path or os.path.join(self.workdir, f"render_{int(time.time()*1000)}.png"))
        raw = path if not (overlay or markers) else path.replace(".png", "_raw.png")
        self.bl.call("render", path=raw, res=res)
        if not (overlay or markers):
            return dict(path=path)
        img = cv2.imread(raw)
        m2d = np.array(self.bl.call("project_vertices", vertex_ids=self.vids, res=res))
        if overlay and self.ref is not None:
            ref = cv2.imread(self.ref_image)
            T = F.umeyama2d(m2d, self.ref2d(), self.weights)
            warped = cv2.warpAffine(img, T, (ref.shape[1], ref.shape[0]), borderValue=(64, 64, 64))
            img = cv2.addWeighted(ref, 1 - opacity, warped, opacity, 0)
            m2d = F.apply(T, m2d)
            if markers:
                for p in self.ref2d(): cv2.circle(img, tuple(int(v) for v in p), 3, (0, 255, 0), -1)
        if markers:
            for p in m2d: cv2.circle(img, tuple(int(v) for v in p), 3, (0, 0, 255), -1)
        cv2.imwrite(path, img)
        return dict(path=path, raw=raw, legend="green=reference landmarks, red=model landmarks")

    def mediapipe_validation(self, render_path):
        """Independent check: run MediaPipe on a model render and compare to reference (same subset)."""
        r = detect(render_path)
        if r is None: return None
        p = np.array(r["landmarks_px"])[self.mp_ids, :2]
        e = F.landmark_error(p, self.ref2d(), self.weights, self.ref_iod()); e.pop("per_point_px"); e.pop("T_model_to_ref")
        return e

    def export_model(self, path, fmt=None):
        path = os.path.abspath(path); fmt = fmt or os.path.splitext(path)[1][1:]
        return dict(path=self.bl.call("export_model", path=path, fmt=fmt))
