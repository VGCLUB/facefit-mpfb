"""Bounded least-squares fitting of MPFB slider values to 2D landmarks (runs outside Blender).

Model: shape keys blend linearly, so mapped vertex positions are
    V(w) = R + sum_t w_t * D_t,   w_t in [0, 1]   (each signed slider = incr target - decr target)
We project orthographically to the front plane (x, -z), and alternate:
  1. similarity-align (Umeyama: scale+rotation+translation) the reference landmarks into the model plane
  2. solve a bounded, ridge-regularised linear least squares for w (scipy lsq_linear)
Reported errors are in reference-image pixels and as % of inter-ocular distance (IOD).
"""
import numpy as np
from scipy.optimize import lsq_linear

L_EYE_OUT, R_EYE_OUT = 33, 263

# sliders that are visible from the front and meaningful for face shape
DEFAULT_FIT_PARAMS = [
    "head-scale-vert-decr-incr", "head-scale-horiz-decr-incr", "head-fat-decr-incr", "head-oval", "head-round",
    "head-invertedtriangular", "head-triangular", "head-square", "head-diamond", "head-rectangular",
    "chin-height-decr-incr", "chin-width-decr-incr", "chin-triangle", "chin-jaw-drop-decr-incr", "chin-bones-decr-incr",
    "eye-scale-decr-incr", "eye-trans-in-out", "eye-trans-down-up", "eye-height1-decr-incr", "eye-height2-decr-incr",
    "eye-height3-decr-incr", "eye-corner1-down-up", "eye-corner2-down-up", "eye-epicanthus-in-out",
    "eyebrows-angle-down-up", "eyebrows-trans-down-up",
    "nose-scale-vert-decr-incr", "nose-scale-horiz-decr-incr", "nose-trans-down-up", "nose-nostrils-width-decr-incr",
    "nose-point-width-decr-incr", "nose-width1-decr-incr", "nose-width2-decr-incr",
    "mouth-scale-horiz-decr-incr", "mouth-scale-vert-decr-incr", "mouth-trans-down-up", "mouth-upperlip-height-decr-incr",
    "mouth-lowerlip-height-decr-incr", "mouth-upperlip-width-decr-incr", "mouth-lowerlip-width-decr-incr",
    "mouth-angles-down-up", "mouth-cupidsbow-decr-incr",
    "cheek-volume-decr-incr", "cheek-bones-decr-incr", "cheek-inner-decr-incr", "forehead-scale-vert-decr-incr",
    "forehead-temple-decr-incr", "neck-scale-horiz-decr-incr",
]


def umeyama2d(src, dst, w=None):
    """Similarity T (s,R,t) minimising sum w|s R src + t - dst|^2. Returns 2x3 matrix."""
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    S, D = src - ms, dst - md
    cov = (w[:, None] * D).T @ S
    U, sig, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, d])
    R = U @ E @ Vt
    var = (w * (S ** 2).sum(1)).sum()
    s = (sig * np.diag(E)).sum() / var
    t = md - s * R @ ms
    return np.hstack([s * R, t[:, None]])


def apply(T, p):
    return p @ T[:, :2].T + T[:, 2]


def project(V):
    V = np.asarray(V)
    return np.stack([V[..., 0], -V[..., 2]], -1)


def landmark_error(model2d, ref2d, weights=None, ref_iod=None):
    """Align model->ref by similarity, return dict(rms_px, mean_px, rms_iod_pct, per_point)."""
    T = umeyama2d(model2d, ref2d, weights)
    d = np.linalg.norm(apply(T, model2d) - ref2d, axis=1)
    out = dict(rms_px=float(np.sqrt((d ** 2).mean())), mean_px=float(d.mean()), max_px=float(d.max()))
    if ref_iod:
        out["rms_iod_pct"] = 100 * out["rms_px"] / ref_iod
    out["per_point_px"] = d.tolist()
    out["T_model_to_ref"] = T.tolist()
    return out


class LinearFaceModel:
    def __init__(self, lm):
        self.rest = np.asarray(lm["rest"], float)            # (N,3)
        self.params = lm["params"]
        cols, self.cols_meta = [], []
        for p in self.params:
            d = lm["deltas"][p["name"]]
            if d["pos"]:
                cols.append(np.sum(d["pos"], 0)); self.cols_meta.append((p["name"], +1))
            if d["neg"]:
                cols.append(np.sum(d["neg"], 0)); self.cols_meta.append((p["name"], -1))
        self.D = np.stack(cols, -1) if cols else np.zeros(self.rest.shape + (0,))  # (N,3,K)
        self.P0 = project(self.rest)                         # (N,2)
        self.A = np.stack([self.D[:, 0, :], -self.D[:, 2, :]], 1)  # (N,2,K)

    def w_from_values(self, values):
        w = np.zeros(len(self.cols_meta))
        for k, (n, sgn) in enumerate(self.cols_meta):
            v = values.get(n, 0.0)
            w[k] = max(v, 0) if sgn > 0 else max(-v, 0)
        return w

    def values_from_w(self, w):
        vals = {p["name"]: 0.0 for p in self.params}
        for k, (n, sgn) in enumerate(self.cols_meta):
            vals[n] += sgn * w[k]
        return {k: float(np.round(v, 4)) for k, v in vals.items()}

    def points2d(self, w):
        return self.P0 + np.einsum("ndk,k->nd", self.A, w)


def solve(model, ref2d, weights, w0=None, reg=0.02, iters=15, locked=None, max_abs=1.0):
    """Returns (values, history). ref2d in reference pixels; weights per point."""
    K = len(model.cols_meta)
    w = np.zeros(K) if w0 is None else np.asarray(w0, float).copy()
    locked = locked or {}
    lock_mask = np.array([n in locked for n, _ in model.cols_meta])
    sw = np.sqrt(np.repeat(np.asarray(weights, float), 2))
    hist = []
    for it in range(iters):
        P = model.points2d(w)
        T = umeyama2d(ref2d, P, weights)          # ref -> model plane
        tgt = apply(T, ref2d)
        scale = np.sqrt(abs(np.linalg.det(T[:, :2])))
        # normalise residuals by model-plane IOD so reg is scale-free
        iod = np.linalg.norm(tgt[model_idx(L_EYE_OUT)] - tgt[model_idx(R_EYE_OUT)]) if _IDX else 1.0
        A = model.A.reshape(-1, K) * sw[:, None] / iod
        b = (tgt - model.P0).reshape(-1) * sw / iod
        lb = np.zeros(K); ub = np.full(K, max_abs)
        for k, (n, sgn) in enumerate(model.cols_meta):
            if n in locked:
                v = locked[n]; val = max(v, 0) if sgn > 0 else max(-v, 0)
                lb[k], ub[k] = val - 1e-9, val + 1e-9
        Areg = np.vstack([A, np.sqrt(reg) * np.eye(K)])
        breg = np.concatenate([b, np.zeros(K)])
        res = lsq_linear(Areg, breg, bounds=(lb, ub), method="bvls")
        w_new = res.x
        hist.append(float(np.sqrt(np.mean(((model.points2d(w_new) - tgt) ** 2).sum(1))) / iod))
        if np.abs(w_new - w).max() < 1e-4:
            w = w_new; break
        w = w_new
    return model.values_from_w(w), hist


_IDX = {}
def set_index(mp_ids):
    _IDX.clear(); _IDX.update({m: k for k, m in enumerate(mp_ids)})
def model_idx(mp):
    return _IDX[mp]
