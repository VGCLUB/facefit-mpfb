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
HEAD_LABELS = FACE_LABELS + [7, 8, 9]          # + ears (bald / visible-skull references)


def mirror_map(co, tol=2e-3):
    m = co.copy(); m[:, 0] *= -1
    d, idx = cKDTree(co).query(m)
    idx[d > tol] = -1
    return idx


def face_boundary(labels, label_set=None):
    """Per-row left/right x of the (hole-filled) parsed face mask, plus lowest row (chin)."""
    import cv2
    mask = np.isin(labels, label_set or FACE_LABELS).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    big = max(cnts, key=cv2.contourArea); filled = np.zeros_like(mask); cv2.drawContours(filled, [big], -1, 1, -1)
    rows = {}
    for y in range(filled.shape[0]):
        xs = np.flatnonzero(filled[y])
        if len(xs): rows[y] = (xs[0], xs[-1])
    return rows, filled


def compute_refine(dump, vids, ref2d, T_model_to_ref, labels, ref_lm_all, strength=1.0, contour=True,
                   contour_from_lm=61, smooth=1e-4, full_head=False):
    co = dump["co"]; nrm = dump["normals"]; helper = dump["helper"]; top = float(dump["top"])
    P = F.project(co)                                   # (N,2) model plane
    Tm2r = np.asarray(T_model_to_ref)
    A = Tm2r[:, :2]; t = Tm2r[:, 2]; Ainv = np.linalg.inv(A)
    to_model = lambda q: (np.asarray(q) - t) @ Ainv.T
    to_ref = lambda p: np.asarray(p) @ A.T + t
    ctrl_v, ctrl_d = list(vids), list(to_model(ref2d) - P[vids])
    info = {"landmark_controls": len(vids), "contour_controls": 0}
    if contour and full_head:
        import cv2
        fb = face_boundary(labels, HEAD_LABELS)
        if fb:
            rows, filled = fb
            Hh, Ww = filled.shape
            q = to_ref(P)
            chin_y = ref_lm_all[152][1]
            c_ref = (ref_lm_all[33][:2] + ref_lm_all[263][:2]) / 2
            c_mod = to_model(c_ref)
            s_ = np.sqrt(abs(np.linalg.det(A)))                     # ref px per model unit
            zc = P[vids[[i for i in range(len(vids))]]] if False else None
            head = ~helper & (q[:, 1] < chin_y + 2) & (co[:, 2] > top - 0.32)
            hv = np.flatnonzero(head)
            rel = P[hv] - c_mod
            ang = np.degrees(np.arctan2(rel[:, 0], -rel[:, 1]))      # 0 = up, +-180 = down
            rad = np.linalg.norm(rel, axis=1)
            step = 3.0
            for th in np.arange(-180, 180, step):
                d = np.array([np.sin(np.radians(th)), -np.cos(np.radians(th))])   # model plane (x, -z): up = -1 in plane y
                # ref ray (image y down == model plane y)
                rr = 0.0
                for t in np.arange(1, max(Hh, Ww), 1.0):
                    x, y = c_ref + d * t
                    if x < 0 or y < 0 or x >= Ww or y >= Hh: rr = None; break
                    if filled[int(y), int(x)]: rr = t
                    elif t > rr + 25: break
                if not rr: continue
                if (c_ref + d * rr)[1] > chin_y + 2: continue          # below the chin -> neck, skip
                sel = np.abs(((ang - th + 180) % 360) - 180) < step / 2
                if not sel.any(): continue
                k = hv[sel][np.argmax(rad[sel])]
                cur = np.linalg.norm(P[k] - c_mod)
                ctrl_v.append(k); ctrl_d.append(d * (rr / s_ - cur)); info["contour_controls"] += 1
            # neck width rows
            neck = (labels == 14).astype(np.uint8)
            iod = np.linalg.norm(ref_lm_all[33][:2] - ref_lm_all[263][:2])
            nv = np.flatnonzero(~helper & (q[:, 1] > chin_y + 0.15 * iod) & (q[:, 1] < min(chin_y + 0.7 * iod, Hh - 5)))
            info["neck_controls"] = 0
            for yy in np.arange(chin_y + 0.2 * iod, min(chin_y + 0.65 * iod, Hh - 5), 0.08 * iod):
                xs = np.flatnonzero(neck[int(yy)])
                if len(xs) < 2: continue
                g = nv[np.abs(q[nv, 1] - yy) < 0.03 * iod]
                for side, xt in ((-1, xs[0]), (1, xs[-1])):
                    gg = g[(P[g, 0] - c_mod[0]) * side > 0]
                    if not len(gg): continue
                    v = gg[np.argmax((P[gg, 0] - c_mod[0]) * side)]
                    # which image side does this model side map to?
                    img_side_right = q[v, 0] > c_ref[0]
                    xt = xs[-1] if img_side_right else xs[0]
                    tgt = to_model(np.array([xt, q[v, 1]]))
                    ctrl_v.append(v); ctrl_d.append(np.array([tgt[0] - P[v, 0], 0.0])); info["neck_controls"] += 1
    elif contour:
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
    rng = np.random.RandomState(0)
    region = np.flatnonzero(co[:, 2] > top - 0.56)
    D2 = np.zeros((len(co), 2))
    if full_head and len(ctrl_v) > len(vids) + 10:
        # stage 1: smooth large-scale 2D (x,z) field from the silhouette, applied to every head vertex
        anchor = np.flatnonzero((co[:, 2] < top - 0.42) & (co[:, 2] > top - 0.55) & ~helper)
        anchor = rng.choice(anchor, min(300, len(anchor)), replace=False)
        nl = len(vids); cv_, cd_ = ctrl_v[nl:], ctrl_d[nl:]
        X2 = np.vstack([P[cv_], P[anchor]]); Y2 = np.vstack([cd_, np.zeros((len(anchor), 2))])
        _, u2 = np.unique(np.round(X2, 6), axis=0, return_index=True)
        rbf2 = RBFInterpolator(X2[u2], Y2[u2], kernel="thin_plate_spline", smoothing=1e-3)
        D2[region] = rbf2(P[region])
        # stage 2: residual landmark field (3D, local), zero at silhouette/back/neck
        res = ctrl_d[:nl] - D2[ctrl_v[:nl]]
        zero = np.concatenate([cv_, anchor, rng.choice(np.flatnonzero((co[:, 1] > 0.0) & (co[:, 2] > top - 0.42) & ~helper), 300, replace=False)])
        X = np.vstack([co[ctrl_v[:nl]], co[zero]]); Y = np.vstack([res, np.zeros((len(zero), 2))])
        _, uniq = np.unique(np.round(X, 6), axis=0, return_index=True)
        rbf = RBFInterpolator(X[uniq], Y[uniq], kernel="thin_plate_spline", smoothing=smooth)
        D2[region] += rbf(co[region])
    else:
        # anchors (zero displacement): neck below the jaw and back of the head
        anchor = np.flatnonzero(((co[:, 2] < top - 0.27) & (co[:, 2] > top - 0.45)) |      # neck
                                ((co[:, 1] > 0.03) & (co[:, 2] > top - 0.3)) |                # back of head
                                (co[:, 2] > top - 0.045))                                      # skull top
        anchor = anchor[~helper[anchor]]
        anchor = rng.choice(anchor, min(500, len(anchor)), replace=False)
        X = np.vstack([co[ctrl_v], co[anchor]])
        Y = np.vstack([ctrl_d, np.zeros((len(anchor), 2))])
        _, uniq = np.unique(np.round(X, 6), axis=0, return_index=True); X, Y = X[uniq], Y[uniq]
        rbf = RBFInterpolator(X, Y, kernel="thin_plate_spline", smoothing=smooth)
        D2[region] = rbf(co[region])
    # fade out towards the neck
    fade = np.clip((co[:, 2] - ((top - 0.44) if full_head else (top - 0.30))) / 0.05, 0, 1); D2 *= fade[:, None]
    # symmetrise
    mm = mirror_map(dump["basis"])
    ok = mm >= 0
    Ds = D2.copy()
    Ds[ok, 0] = 0.5 * (D2[ok, 0] - D2[mm[ok], 0]); Ds[ok, 1] = 0.5 * (D2[ok, 1] + D2[mm[ok], 1])
    delta = np.zeros_like(co); delta[:, 0] = Ds[:, 0]; delta[:, 2] = -Ds[:, 1]
    if full_head:
        # depth is unobserved: give the cranium the same relative vertical stretch in Y so the dome stays round
        ear_z = top - 0.13
        cran = (co[:, 2] > ear_z) & ~helper
        if cran.any():
            k_top = np.argmax(np.where(cran, co[:, 2], -1))
            sv = delta[k_top, 2] / max(top - ear_z, 1e-3)
            yc = np.median(co[cran, 1])
            t = np.clip((co[:, 2] - ear_z) / (top - ear_z), 0, 1) * cran
            backw = np.clip((co[:, 1] - (yc - 0.03)) / 0.06, 0, 1)       # only the back/top, not the face
            delta[:, 1] += (co[:, 1] - yc) * sv * t * backw
            info["cranium_depth_stretch"] = float(sv)
    info["max_disp_mm"] = float(np.abs(delta).max() * 1000)
    info["mirror_coverage"] = float(ok.mean())
    return delta, info


def local_scale_delta(co, center, radius, scale_xyz, falloff=2.0):
    """Smooth local scaling around a centre (for stylisation shape keys)."""
    d = co - center; r = np.linalg.norm(d, axis=1) / radius
    w = np.clip(1 - r, 0, 1) ** falloff
    return d * (np.asarray(scale_xyz) - 1) * w[:, None]


def silhouette_iou(dump, T_model_to_ref, labels, chin_y, out_png=None, ref_bgr=None):
    """IoU between the projected head mesh (above the chin line) and the parsed head mask (face+ears)."""
    import cv2
    co, tris, helper, top = dump["co"], dump["tris"], dump["helper"], float(dump["top"])
    T = np.asarray(T_model_to_ref); q = F.project(co) @ T[:, :2].T + T[:, 2]
    keep = ~helper[tris].any(1) & (co[tris][:, :, 2].min(1) > top - 0.45)
    H, W = labels.shape
    m = np.zeros((H, W), np.uint8)
    for t in tris[keep]:
        cv2.fillConvexPoly(m, np.round(q[t] * 4).astype(np.int32), 1, shift=2)
    fb = face_boundary(labels, HEAD_LABELS)
    ref = fb[1] if fb else np.isin(labels, HEAD_LABELS).astype(np.uint8)
    cut = np.arange(H)[:, None] <= chin_y
    a, b = (m > 0) & cut, (ref > 0) & cut
    iou = float((a & b).sum() / max((a | b).sum(), 1))
    if out_png is not None and ref_bgr is not None:
        vis = ref_bgr.copy()
        cnt_m, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_r, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnt_r, -1, (0, 255, 0), 3); cv2.drawContours(vis, cnt_m, -1, (0, 0, 255), 3)
        cv2.imwrite(out_png, vis)
    return iou
