"""Phase 3: landmark + silhouette driven per-vertex refinement (smooth, symmetric RBF in the front plane).

Controls (model plane x,-z):
  * mapped MediaPipe vertices -> reference landmarks (similarity-mapped into the model plane)
  * lower-face silhouette vertices -> left/right boundary of the face-parsing skin mask (per row)
  * zero-displacement anchors on the neck / back of the head
The 2D displacement field is interpolated with a thin-plate RBF over head vertices, symmetrised with a
mirror map, faded out towards the neck, and stored as a Blender shape key ('facefit_refine').
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from . import fit as F

FACE_LABELS = [1, 2, 3, 4, 5, 10, 11, 12, 13]


def mirror_map(co, tol=2e-3):
    m = co.copy(); m[:, 0] *= -1
    d, idx = cKDTree(co).query(m)
    idx[d > tol] = -1
    return idx


def face_boundary(labels):
    """Per-row left/right x of the (hole-filled) parsed face mask, plus lowest row (chin)."""
    import cv2
    mask = np.isin(labels, FACE_LABELS).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    big = max(cnts, key=cv2.contourArea); filled = np.zeros_like(mask); cv2.drawContours(filled, [big], -1, 1, -1)
    rows = {}
    for y in range(filled.shape[0]):
        xs = np.flatnonzero(filled[y])
        if len(xs): rows[y] = (xs[0], xs[-1])
    return rows, filled


def compute_refine(dump, vids, ref2d, T_model_to_ref, labels, ref_lm_all, strength=1.0, contour=True,
                   contour_from_lm=61, smooth=1e-4):
    co = dump["co"]; nrm = dump["normals"]; helper = dump["helper"]; top = float(dump["top"])
    P = F.project(co)                                   # (N,2) model plane
    Tm2r = np.asarray(T_model_to_ref)
    A = Tm2r[:, :2]; t = Tm2r[:, 2]; Ainv = np.linalg.inv(A)
    to_model = lambda q: (np.asarray(q) - t) @ Ainv.T
    to_ref = lambda p: np.asarray(p) @ A.T + t
    ctrl_v, ctrl_d = list(vids), list(to_model(ref2d) - P[vids])
    info = {"landmark_controls": len(vids), "contour_controls": 0}
    if contour:
        fb = face_boundary(labels)
        if fb:
            rows, _ = fb
            y_start = ref_lm_all[contour_from_lm][1]              # from mouth-corner height ...
            y_chin = max(rows)                                    # ... down to the chin
            head = (co[:, 2] > top - 0.36) & ~helper
            front = head & (co[:, 1] < -0.02) & (np.abs(nrm[:, 1]) < 0.55)
            q = to_ref(P)
            cand = np.flatnonzero(front & (q[:, 1] >= y_start) & (q[:, 1] <= y_chin - 3))
            bins = np.round(q[cand, 1] / 6).astype(int)
            for side in (-1, 1):
                sel = cand[(P[cand, 0] * side) > 0]
                if not len(sel): continue
                bsel = np.round(q[sel, 1] / 6).astype(int)
                for b in np.unique(bsel):
                    grp = sel[bsel == b]
                    v = grp[np.argmax(P[grp, 0] * side)]              # outermost vertex in this row band
                    yy = int(round(q[v, 1]))
                    if yy not in rows: continue
                    # model x positive = image right? check via mapping of a known point
                    xl, xr = rows[yy]
                    xt = xr if q[v, 0] > to_ref(np.array([0.0, P[v, 1]]))[0] else xl
                    tgt = to_model(np.array([xt, q[v, 1]]))
                    ctrl_v.append(v); ctrl_d.append(np.array([tgt[0] - P[v, 0], 0.0]))
                    info["contour_controls"] += 1
    ctrl_v = np.array(ctrl_v); ctrl_d = np.array(ctrl_d) * strength
    # anchors (zero displacement): neck below the jaw and back of the head
    rng = np.random.RandomState(0)
    anchor = np.flatnonzero(((co[:, 2] < top - 0.27) & (co[:, 2] > top - 0.45)) |      # neck
                            ((co[:, 1] > 0.03) & (co[:, 2] > top - 0.3)) |                # back of head
                            (co[:, 2] > top - 0.045))                                      # skull top
    anchor = anchor[~helper[anchor]]
    anchor = rng.choice(anchor, min(500, len(anchor)), replace=False)
    X = np.vstack([co[ctrl_v], co[anchor]])
    Y = np.vstack([ctrl_d, np.zeros((len(anchor), 2))])
    # dedupe identical points
    _, uniq = np.unique(np.round(X, 6), axis=0, return_index=True); X, Y = X[uniq], Y[uniq]
    rbf = RBFInterpolator(X, Y, kernel="thin_plate_spline", smoothing=smooth)
    region = np.flatnonzero(co[:, 2] > top - 0.48)
    D2 = np.zeros((len(co), 2)); D2[region] = rbf(co[region])
    # fade out towards the neck
    fade = np.clip((co[:, 2] - (top - 0.30)) / 0.05, 0, 1); D2 *= fade[:, None]
    # symmetrise
    mm = mirror_map(dump["basis"])
    ok = mm >= 0
    Ds = D2.copy()
    Ds[ok, 0] = 0.5 * (D2[ok, 0] - D2[mm[ok], 0]); Ds[ok, 1] = 0.5 * (D2[ok, 1] + D2[mm[ok], 1])
    delta = np.zeros_like(co); delta[:, 0] = Ds[:, 0]; delta[:, 2] = -Ds[:, 1]
    info["max_disp_mm"] = float(np.abs(delta).max() * 1000)
    info["mirror_coverage"] = float(ok.mean())
    return delta, info


def local_scale_delta(co, center, radius, scale_xyz, falloff=2.0):
    """Smooth local scaling around a centre (for stylisation shape keys)."""
    d = co - center; r = np.linalg.norm(d, axis=1) / radius
    w = np.clip(1 - r, 0, 1) ** falloff
    return d * (np.asarray(scale_xyz) - 1) * w[:, None]
