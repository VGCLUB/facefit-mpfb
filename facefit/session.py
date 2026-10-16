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
        blend = path.replace(".json", ".blend")          # full scene incl. macros, refine/style keys, materials
        self.bl.call("save_blend", path=blend)
        return dict(name=name, path=path, blend=blend)

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

    def ref_z(self):
        return np.array(self.ref["landmarks_px"])[self.mp_ids, 2]

    def depth_mask(self):
        # jaw = silhouette (MediaPipe z unreliable); eyelid vertices sit on the eye-hole rim (depth noisy)
        from .landmark_sets import JAW, EYES
        return np.array([0.0 if (i in JAW or i in EYES) else 1.0 for i in self.mp_ids])

    def get_depth_error(self):
        """RMS depth error (% model IOD) of mapped vertices vs MediaPipe z (scale from 2D alignment, free offset)."""
        if self.ref is None: self.detect_landmarks()
        m2 = self.model2d(); V = np.array(self.bl.call("vertex_coords", vertex_ids=self.vids))
        P = F.project(V)
        T = F.umeyama2d(self.ref2d(), P, self.weights); s = np.sqrt(abs(np.linalg.det(T[:, :2])))
        iod = np.linalg.norm(P[self.mp_ids.index(F.L_EYE_OUT)] - P[self.mp_ids.index(F.R_EYE_OUT)])
        dm = self.depth_mask()
        return F.depth_error(-V[:, 1], -self.ref_z(), s, self.weights * dm, iod) if dm.any() else None

    def auto_fit(self, params=None, reg=0.005, style=None, locked=None, iters=15, depth_weight=0.0):
        """Fit sliders to the reference. style: offsets added after fitting (e.g. {'eye-scale-decr-incr': 0.2})."""
        params = params or (F.DEFAULT_FIT_PARAMS + (F.DEPTH_FIT_PARAMS if depth_weight > 0 else []))
        avail = {p["name"] for p in self.list_parameters()}
        params = [p for p in params if p in avail]
        self.push_undo()
        before = self.get_landmark_error(); before["depth_rms_iod_pct"] = self.get_depth_error()
        lm = self.bl.call("linear_model", vertex_ids=self.vids, param_names=params)
        model = F.LinearFaceModel(lm)
        cur = self.bl.call("get_parameters", names=params)
        t = time.time()
        vals, hist = F.solve(model, self.ref2d(), self.weights, w0=model.w_from_values(cur), reg=reg,
                             iters=iters, locked=locked, ref_z=self.ref_z(), depth_weight=depth_weight,
                             depth_mask=self.depth_mask())
        for k, dv in (style or {}).items():
            vals[k] = float(np.clip(vals.get(k, 0.0) + dv, -1, 1))
        self.bl.call("set_parameters", values=vals)
        after = self.get_landmark_error(); after["depth_rms_iod_pct"] = self.get_depth_error()
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

    def project_texture(self, outdir=None, res=768, add_eyes=True):
        """Project the reference onto MPFB UVs; writes albedo + region masks; applies material in Blender."""
        import cv2
        from . import texture, parsing
        if self.ref is None: self.detect_landmarks()
        outdir = os.path.abspath(outdir or os.path.join(self.workdir, "textures"))
        os.makedirs(outdir, exist_ok=True)
        eyes = None
        if add_eyes:
            try: eyes = self.bl.call("add_eyes")
            except Exception as e: eyes = f"failed: {str(e).splitlines()[0]}"
        npz = os.path.join(outdir, "uvproj.npz")
        self.bl.call("uv_projection_data", path=npz, res=res)
        m2d = np.array(self.bl.call("project_vertices", vertex_ids=self.vids, res=res))
        T = F.umeyama2d(m2d, self.ref2d(), self.weights)
        ref = cv2.imread(self.ref_image)
        labels = parsing.parse(ref, self.ref["landmarks_px"])
        cv2.imwrite(os.path.join(outdir, "ref_parsing.png"), cv2.addWeighted(ref, 0.4, parsing.colorize(labels), 0.6, 0))
        res_files = texture.project(npz, ref, T, labels, outdir)
        # eyes: the photo's iris covers too few pixels for direct projection -> recolour the MakeHuman eye texture
        tpl = os.path.join(os.path.expanduser("~"), ".config/blender/4.2/extensions/.user/user_default/mpfb/data/eyes/materials/brown_eye.png")
        tpl = os.environ.get("FACEFIT_EYE_TEMPLATE", tpl)
        eye_tex = None
        if os.path.exists(tpl):
            eye = texture.recolor_eye_texture(tpl, ref, labels, os.path.join(outdir, "eyes_iris_recolored.png"))
            res_files["eyes_recolored"] = eye; eye_tex = eye["path"]
        self.bl.call("apply_textures", albedo=res_files["human"]["albedo"],
                     eyes_albedo=eye_tex or res_files.get("eyes", {}).get("albedo"))
        return dict(eyes=eyes, files=res_files)

    def render_textured(self, path, yaw_deg=0.0, res=768, samples=48):
        return dict(path=self.bl.call("render_textured", path=os.path.abspath(path), res=res, yaw_deg=yaw_deg, samples=samples))

    # ---------------- phase 3
    def dump(self):
        path = os.path.join(self.workdir, "mesh_dump.npz")
        self.bl.call("mesh_dump", path=path); return dict(np.load(path))

    def T_model_to_ref(self):
        V = np.array(self.bl.call("vertex_coords", vertex_ids=self.vids))
        return F.umeyama2d(F.project(V), self.ref2d(), self.weights)

    def labels(self):
        if getattr(self, "_labels", None) is None or self._labels_for != self.ref_image:
            import cv2
            from . import parsing
            self._labels = parsing.parse(cv2.imread(self.ref_image), self.ref["landmarks_px"]); self._labels_for = self.ref_image
        return self._labels

    def fit_macros(self, candidates=None, depth_weight=0.3):
        """Try a few MPFB macro presets (female/young first), fit sliders for each, keep the best 2D error."""
        candidates = candidates or [dict(gender=0.0, age=0.5, asian=a, proportions=p, weight=w)
                                    for a in (0.5, 1.0) for p in (0.5, 1.0) for w in (0.5,)]
        res = []
        for c in candidates:
            mac = dict(c); a = mac.pop("asian", None)
            if a is not None: mac.update(asian=a, caucasian=(1 - a) / 2, african=(1 - a) / 2)
            self.bl.call("set_macros", values=mac); self.bl.call("reset_parameters")
            r = self.auto_fit(depth_weight=depth_weight)
            res.append((r["error_after"]["rms_iod_pct"], c))
        best = min(res, key=lambda x: x[0])[1]
        mac = dict(best); a = mac.pop("asian", None)
        if a is not None: mac.update(asian=a, caucasian=(1 - a) / 2, african=(1 - a) / 2)
        self.bl.call("set_macros", values=mac); self.bl.call("reset_parameters")
        fit = self.auto_fit(depth_weight=depth_weight)
        return dict(best=best, tried=[dict(err=e, **c) for e, c in res], fit_error=fit["error_after"])

    def vertex_refine(self, passes=2, strength=1.0, contour=True):
        """Per-vertex smooth symmetric deformation (shape key 'facefit_refine') to match landmarks + parsed jaw contour."""
        from . import refine as R
        self.push_undo()
        self.bl.call("remove_keys", prefix="facefit_refine")
        before = self.get_landmark_error()
        total = None; infos = []
        npy = os.path.join(self.workdir, "refine_delta.npy")
        for k in range(passes):
            d = self.dump()
            delta, info = R.compute_refine(d, self.vids, self.ref2d(), self.T_model_to_ref(), self.labels(),
                                           np.array(self.ref["landmarks_px"]), strength=strength, contour=contour)
            total = delta if total is None else total + delta
            np.save(npy, total); self.bl.call("set_delta_key", name="facefit_refine", npy_path=npy, value=1.0)
            infos.append(info)
        after = self.get_landmark_error()
        return dict(error_before=before, error_after=after, passes=infos)

    def add_style_keys(self):
        d = self.dump(); co = d["basis"]
        ids = {m: v for m, v in zip(self.mp_ids, self.vids)}
        c = lambda *ms: co[[ids[m] for m in ms]].mean(0)
        sd = __import__("facefit.v3", fromlist=["x"]).style_deltas(
            d, [c(33, 133, 160, 144), c(263, 362, 387, 373)], c(61, 291, 0, 17), c(1, 98, 327), co[ids[152]])
        for name, delta in sd.items():
            p = os.path.join(self.workdir, f"{name}.npy"); np.save(p, delta)
            self.bl.call("set_delta_key", name=name, npy_path=p, value=0.0)
        return list(sd)

    def set_style(self, values):
        """values: {style_big_eyes, style_small_nose, style_full_lips, style_v_chin: 0..1.5}"""
        return self.bl.call("set_key_values", values=values)

    def project_face(self, outdir=None, hair_cap=True, delight=0.3):
        """Phase-3 texture: cleaned photo camera-projected onto the fitted mesh (per-vertex UV), body = skin colour."""
        import cv2
        from . import v3, texture
        outdir = os.path.abspath(outdir or os.path.join(self.workdir, "textures_v3")); os.makedirs(outdir, exist_ok=True)
        try: self.bl.call("add_eyes")
        except Exception as e: pass
        ref = cv2.imread(self.ref_image); labels = self.labels()
        clean, filled = v3.clean_photo(ref, labels, os.path.join(outdir, "photo_clean.png"), delight=delight)
        d = self.dump()
        npz, info = v3.projection_npz(d, self.T_model_to_ref(), labels, filled, ref.shape, os.path.join(outdir, "proj.npz"), ref)
        self.bl.call("setup_face_projection", npz_path=npz, photo_path=clean)
        tpl = os.path.join(os.path.expanduser("~"), ".config/blender/4.2/extensions/.user/user_default/mpfb/data/eyes/materials/brown_eye.png")
        if os.path.exists(tpl):
            eye = texture.recolor_eye_texture(tpl, ref, labels, os.path.join(outdir, "eyes_iris_recolored.png"))
            self.bl.call("apply_eye_texture", path=eye["path"])
        if hair_cap: info["hair_cap"] = self.bl.call("make_hair_cap", npz_path=npz)
        info["photo_clean"] = clean
        return info

    def export_model(self, path, fmt=None):
        path = os.path.abspath(path); fmt = fmt or os.path.splitext(path)[1][1:]
        return dict(path=self.bl.call("export_model", path=path, fmt=fmt))
