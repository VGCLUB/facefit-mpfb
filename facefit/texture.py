"""Phase 2: project the reference photo onto MPFB UVs (front orthographic camera, similarity-aligned).

Per-triangle affine warps from reference-image space to UV space, with z-buffer occlusion,
normal-based weights and face-parsing validity (hair/background never projected). Unseen texels
fall back to a median skin colour (eyes: sclera colour) with a feathered transition.
"""
import os
import numpy as np, cv2
from . import parsing

TEX = 4096
VALID = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]       # skin, brows, eyes, ears, nose, mouth, lips (not neck: hair shadows)


def _zbuffer(tri_q, tri_depth, shape):
    """Painter's algorithm triangle-id buffer in reference space."""
    buf = np.zeros(shape, np.int32)
    order = np.argsort(tri_depth)                     # far (small depth toward camera) first
    for i in order:
        cv2.fillConvexPoly(buf, np.round(tri_q[i] * 4).astype(np.int32), int(i + 1), lineType=cv2.LINE_8, shift=2)
    return buf


def _warp_tri(src_img, src_tri, dst_tri, out_shape, interp):
    x, y, w, h = cv2.boundingRect(np.float32(dst_tri))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, out_shape[1]), min(y + h, out_shape[0])
    if x1 <= x0 or y1 <= y0: return None
    M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri - [x0, y0]))
    patch = cv2.warpAffine(src_img, M, (x1 - x0, y1 - y0), flags=interp, borderMode=cv2.BORDER_REFLECT)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, np.round((dst_tri - [x0, y0]) * 4).astype(np.int32), 1, shift=2)
    return (y0, y1, x0, x1), patch, m


def project(npz_path, ref_bgr, T_render_to_ref, labels, outdir, tex=TEX):
    d = np.load(npz_path)
    H, W = ref_bgr.shape[:2]
    objs = [k[:-3] for k in d.files if k.endswith("_tv")]
    # gather triangles of all objects in reference space
    tris = []
    for tag in objs:
        tv, px = d[f"{tag}_tv"], d[f"{tag}_px"]
        q = px @ np.asarray(T_render_to_ref)[:, :2].T + np.asarray(T_render_to_ref)[:, 2]
        tris.append(dict(tag=tag, tv=tv, q=q[tv], uv=d[f"{tag}_uv"], depth=d[f"{tag}_depth"][tv].mean(1),
                         cos=d[f"{tag}_cos"][tv].min(1)))
    allq = np.concatenate([t["q"] for t in tris]); alld = np.concatenate([t["depth"] for t in tris])
    buf = _zbuffer(allq, alld, (H, W))
    vis_ids = np.bincount(buf.ravel(), minlength=len(allq) + 1)[1:]   # visible pixel count per triangle
    hair = cv2.dilate(np.isin(labels, [0, 17, 18]).astype(np.uint8), np.ones((15, 15), np.uint8))  # hair/bg + shadow band
    valid = (np.isin(labels, VALID) & (hair == 0)).astype(np.float32)
    lab3 = labels.astype(np.uint8)
    results = {}
    off = 0
    for t in tris:
        n = len(t["tv"]); vis = vis_ids[off:off + n]; off += n
        acc = np.zeros((tex, tex, 3), np.float32); wsum = np.zeros((tex, tex), np.float32)
        uvlab = np.zeros((tex, tex), np.uint8)
        dst_all = np.stack([t["uv"][..., 0] * tex, (1 - t["uv"][..., 1]) * tex], -1)
        for i in range(n):
            if t["cos"][i] < 0.05 or vis[i] == 0:
                continue
            src = t["q"][i]
            if (src[:, 0] < 0).any() or (src[:, 1] < 0).any() or (src[:, 0] >= W).any() or (src[:, 1] >= H).any():
                continue
            r = _warp_tri(ref_bgr, src, dst_all[i], (tex, tex), cv2.INTER_LINEAR)
            if r is None: continue
            (y0, y1, x0, x1), patch, m = r
            vpatch = _warp_tri(valid, src, dst_all[i], (tex, tex), cv2.INTER_LINEAR)[1]
            lpatch = _warp_tri(lab3, src, dst_all[i], (tex, tex), cv2.INTER_NEAREST)[1]
            wgt = m * vpatch * float(np.clip(t["cos"][i], 0, 1)) ** 1.5
            acc[y0:y1, x0:x1] += patch.astype(np.float32) * wgt[..., None]
            wsum[y0:y1, x0:x1] += wgt
            uvlab[y0:y1, x0:x1] = np.where(m > 0, lpatch, uvlab[y0:y1, x0:x1])
        proj = acc / np.maximum(wsum, 1e-6)[..., None]
        if t["tag"] == "eyes":
            base = np.array([200, 205, 215], np.float32)     # sclera (BGR)
        else:
            skin = ref_bgr[(labels == 1)]
            base = np.median(skin, 0).astype(np.float32) if len(skin) else np.array([170, 185, 220], np.float32)
        cover = (wsum > 0.3).astype(np.uint8)
        alpha = cv2.erode(cover, np.ones((25, 25), np.uint8)).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 12) * cover
        if t["tag"] != "eyes" and cover.any():   # match fill colour to the projected border to hide the seam
            band = (cover > 0) & (alpha < 0.5) & (alpha > 0.05)
            if band.sum() > 50: base = np.median(proj[band], 0).astype(np.float32)
        # inpaint thin gaps inside the projected region before blending
        holes = ((wsum <= 0.3) & (cv2.dilate(cover, np.ones((9, 9), np.uint8)) > 0)).astype(np.uint8)
        filled = np.where(cover[..., None] > 0, proj, base).astype(np.uint8)
        filled = cv2.inpaint(filled, holes, 5, cv2.INPAINT_TELEA)
        albedo = (filled.astype(np.float32) * alpha[..., None] + base * (1 - alpha[..., None])).astype(np.uint8)
        os.makedirs(outdir, exist_ok=True)
        files = {}
        p = os.path.join(outdir, f"{t['tag']}_albedo.png"); cv2.imwrite(p, albedo); files["albedo"] = p
        p = os.path.join(outdir, f"{t['tag']}_coverage.png"); cv2.imwrite(p, (alpha * 255).astype(np.uint8)); files["coverage"] = p
        if t["tag"] == "human":
            for reg, ids in {"skin": [1, 7, 8, 10], "brows": [2, 3], "lips": [12, 13], "eyes": [4, 5], "mouth": [11]}.items():
                mk = (np.isin(uvlab, ids) & (cover > 0)).astype(np.uint8) * 255
                p = os.path.join(outdir, f"mask_{reg}.png"); cv2.imwrite(p, mk); files[f"mask_{reg}"] = p
                rgba = np.dstack([albedo, mk])
                p = os.path.join(outdir, f"tex_{reg}.png"); cv2.imwrite(p, rgba); files[f"tex_{reg}"] = p
        files["visible_triangles"] = int((vis > 0).sum()); files["triangles"] = int(n)
        results[t["tag"]] = files
    return results


def iris_color(ref_bgr, labels):
    """Median colour of saturated pixels inside the parsed eye regions (the iris)."""
    hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
    m = np.isin(labels, [4, 5]) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
    if m.sum() < 10:
        return None
    return np.median(hsv[m], 0)


def recolor_eye_texture(template_path, ref_bgr, labels, out_path, iris_scale=1.0):
    """Recolour the MakeHuman eye texture's iris to the reference iris colour (hue/sat, keep detail)."""
    tpl = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    bgr = tpl[..., :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    c = iris_color(ref_bgr, labels)
    if c is None:
        cv2.imwrite(out_path, bgr); return dict(path=out_path, iris_hsv=None)
    iris = (hsv[..., 1] > 90) & (hsv[..., 2] > 25)
    iris = cv2.GaussianBlur(iris.astype(np.float32), (0, 0), 2)
    new = hsv.copy()
    new[..., 0] = c[0]
    new[..., 1] = np.clip(hsv[..., 1] * (1.35 * c[1] / max(np.median(hsv[..., 1][iris > 0.5]), 1)), 0, 255)
    new[..., 2] = np.clip(hsv[..., 2] * (c[2] / max(np.median(hsv[..., 2][iris > 0.5]), 1)), 0, 255)
    out = hsv * (1 - iris[..., None]) + new * iris[..., None]
    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if tpl.shape[2] == 4:   # keep alpha: the cornea maps to a transparent texel area
        out = np.dstack([out, tpl[..., 3]])
    cv2.imwrite(out_path, out)
    return dict(path=out_path, iris_hsv=[float(x) for x in c])


def _template_irises(tpl_bgr):
    hsv = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] > 90) & (hsv[..., 2] > 25)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, cc, st, cen = cv2.connectedComponentsWithStats(m)
    idx = np.argsort(st[1:, 4])[::-1][:2] + 1
    return [(cen[i], np.sqrt(st[i, 4] / np.pi)) for i in idx]


def _photo_iris_disc(ref, lm, c_id, ring, labels, out_r):
    c = lm[c_id, :2]; r = float(np.mean([np.linalg.norm(lm[k, :2] - c) for k in ring]))
    R = int(np.ceil(r * 1.08))
    x0, y0 = int(c[0]) - R, int(c[1]) - R
    crop = ref[y0:y0 + 2 * R, x0:x0 + 2 * R].astype(np.float32)
    lab = labels[y0:y0 + 2 * R, x0:x0 + 2 * R]
    yy, xx = np.mgrid[0:2 * R, 0:2 * R]; d = np.hypot(xx - (c[0] - x0), yy - (c[1] - y0))
    L = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0]
    vis = (d <= r * 1.02) & np.isin(lab, [4, 5])
    vis &= ~((L < np.percentile(L[vis], 8)) & (d > r * 0.45)) if vis.sum() > 20 else vis   # drop lash shadows outside pupil
    # polar fill: missing angles take the per-radius median of visible samples
    A = 360; RR = 2 * R
    pol = cv2.warpPolar(crop, (RR, A), (c[0] - x0, c[1] - y0), R, cv2.WARP_POLAR_LINEAR)
    pm = cv2.warpPolar(vis.astype(np.float32), (RR, A), (c[0] - x0, c[1] - y0), R, cv2.WARP_POLAR_LINEAR) > 0.5
    for j in range(RR):
        ok = pm[:, j]
        if ok.sum() >= 3:
            med = np.median(pol[ok, j], 0); pol[~ok, j] = med
        elif j > 0:
            pol[:, j] = pol[:, j - 1]
    # light angular smoothing of filled area
    disc = cv2.warpPolar(pol, (2 * R, 2 * R), (R, R), R, cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP)
    s = out_r / r
    disc = cv2.resize(disc, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return disc, r, float(vis.mean())


def photo_iris_texture(template_path, ref, lm, labels, out_path):
    """Replace the template irises with the photo's irises (eyelid-occluded parts filled by polar symmetry)."""
    tpl = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    bgr = tpl[..., :3].copy().astype(np.float32)
    irs = _template_irises(tpl[..., :3])
    info = {}
    for k, ((tc, tr), (cid, ring)) in enumerate(zip(irs, ((468, range(469, 473)), (473, range(474, 478))))):
        out_r = tr * 1.12
        disc, r, vis = _photo_iris_disc(ref, lm, cid, ring, labels, out_r)
        h_, w_ = disc.shape[:2]
        x0, y0 = int(round(tc[0] - w_ / 2)), int(round(tc[1] - h_ / 2))
        yy, xx = np.mgrid[0:h_, 0:w_]; d = np.hypot(xx - w_ / 2, yy - h_ / 2)
        a = np.clip((out_r - d) / (0.08 * out_r), 0, 1)[..., None]
        sub = bgr[y0:y0 + h_, x0:x0 + w_]
        bgr[y0:y0 + h_, x0:x0 + w_] = sub * (1 - a) + disc[:sub.shape[0], :sub.shape[1]] * a
        info[f"iris{k}"] = dict(photo_radius_px=r, visible_fraction=vis, template_radius=float(tr))
    out = bgr.astype(np.uint8)
    if tpl.shape[2] == 4: out = np.dstack([out, tpl[..., 3]])
    cv2.imwrite(out_path, out)
    info["path"] = out_path
    return info


def lash_mask(ref, lm, labels, out_path):
    """Photo-space soft mask of dark eyelash pixels around the eyes (composited onto eyeballs)."""
    L = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    eye = np.isin(labels, [4, 5]).astype(np.uint8)
    iod = np.linalg.norm(lm[33, :2] - lm[263, :2])
    k = max(3, int(iod * 0.04))
    zone = cv2.dilate(eye, np.ones((3 * k, 3 * k), np.uint8)) - cv2.erode(eye, np.ones((k, k), np.uint8))
    iris = np.zeros_like(eye)
    for c, ring in ((468, range(469, 473)), (473, range(474, 478))):
        r = np.mean([np.linalg.norm(lm[j, :2] - lm[c, :2]) for j in ring])
        cv2.circle(iris, tuple(int(v) for v in lm[c, :2]), int(r * 1.05), 1, -1)
    bh = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_BLACKHAT, np.ones((k, k), np.uint8)).astype(np.float32)
    Lskin = np.median(L[labels == 1]) if (labels == 1).any() else 180
    m = np.clip((0.55 * Lskin - L) / 20, 0, 1) * (zone > 0) * (iris == 0)
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    cv2.imwrite(out_path, (m * 255).astype(np.uint8))
    return out_path
