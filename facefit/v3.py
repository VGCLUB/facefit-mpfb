"""Phase 3 pipeline pieces (likeness): macro search, vertex refine, stylisation keys, photo cleaning,
camera-projected face material, hair cap."""
import os, json
import numpy as np, cv2
from . import fit as F, refine as R, parsing

FACE_LABELS = R.FACE_LABELS


def srgb_to_linear(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def clean_photo(ref, labels, out_path, delight=0.3, band=25):
    """Inpaint hair strands / earrings crossing the face, extend skin beyond the face edge, mild de-lighting."""
    rows_filled = R.face_boundary(labels)
    face = np.isin(labels, FACE_LABELS).astype(np.uint8)
    filled = rows_filled[1] if rows_filled else face
    strands = ((filled > 0) & (face == 0)).astype(np.uint8)          # non-face pixels inside the face hull
    strands = cv2.dilate(strands, np.ones((5, 5), np.uint8))
    # inner band along the face outline is where loose hair strands lie -> re-synthesise it from the interior
    ring = cv2.dilate(filled, np.ones((61, 61), np.uint8)) - cv2.erode(filled, np.ones((band, band), np.uint8))
    outside = (ring > 0).astype(np.uint8)
    # thin dark hair strands that parsing labelled as skin: black-hat on luminance, away from eyes/brows/lips/nose
    L = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[..., 0]
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    keep_out = cv2.dilate(np.isin(labels, [2, 3, 4, 5, 10, 11, 12, 13]).astype(np.uint8), np.ones((15, 15), np.uint8))
    thin = ((bh > 7) & (filled > 0) & (keep_out == 0)).astype(np.uint8)
    pass
    n, cc, stats, _ = cv2.connectedComponentsWithStats(thin)
    elong = np.zeros_like(thin)
    for i in range(1, n):                       # keep elongated components (strands), not pores/freckles
        w, h_, a = stats[i, 2], stats[i, 3], stats[i, 4]
        if max(w, h_) >= 15: elong[cc == i] = 1
    strands = np.maximum(strands, cv2.dilate(elong, np.ones((5, 5), np.uint8)))
    hole = np.maximum(strands, outside)
    img = cv2.inpaint(ref, hole, 9, cv2.INPAINT_TELEA)
    if delight > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0]; m = filled > 0
        low = cv2.GaussianBlur(L, (0, 0), 25)
        L2 = L - delight * (low - np.median(L[m]))
        w = cv2.GaussianBlur(cv2.dilate(filled, np.ones((31, 31), np.uint8)).astype(np.float32), (0, 0), 10)
        lab[..., 0] = L * (1 - w) + L2 * w
        img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    cv2.imwrite(out_path, img)
    return out_path, filled


def projection_npz(dump, T_model_to_ref, labels, filled, ref_shape, out_path, ref_bgr):
    co, nrm, top = dump["co"], dump["normals"], float(dump["top"])
    H, W = ref_shape[:2]
    P = F.project(co); Tm = np.asarray(T_model_to_ref)
    q = P @ Tm[:, :2].T + Tm[:, 2]
    uv = np.stack([q[:, 0] / W, 1 - q[:, 1] / H], 1)
    # feathered face mask in image space (slightly shrunk), sampled at each vertex
    fm = cv2.erode(filled, np.ones((5, 5), np.uint8)).astype(np.float32)
    fm = cv2.GaussianBlur(fm, (0, 0), 6)
    qi = np.clip(np.round(q).astype(int), 0, [W - 1, H - 1])
    m_img = fm[qi[:, 1], qi[:, 0]]
    cosv = -nrm[:, 1]
    m = m_img * np.clip((cosv - 0.15) / 0.35, 0, 1) * (co[:, 2] > top - 0.40) * ~dump["helper"]
    skin = ref_bgr[labels == 1]
    skin_rgb = srgb_to_linear(np.median(skin, 0)[::-1]) if len(skin) else np.array([0.6, 0.45, 0.4])
    # hair: non-face head vertices (top/back/sides) + front vertices that land on parsed hair
    hair_lab = np.isin(labels, [17, 18])[qi[:, 1], qi[:, 0]]
    head = (co[:, 2] > top - 0.24) & ~dump["helper"]
    face_any = m_img > 0.02
    ears = (np.abs(co[:, 0]) > 0.055) & (co[:, 1] > -0.04) & (co[:, 2] > top - 0.21) & (co[:, 2] < top - 0.07)
    hair = head & ~face_any & ((co[:, 1] > -0.02) | (co[:, 2] > top - 0.09) |
                               (hair_lab & (cosv > 0) & (co[:, 2] > top - 0.16)) | ears)
    # smooth the hair scalar over the mesh (removes jagged hairline), then threshold for the cap
    tris = dump["tris"]
    import scipy.sparse as sp
    n = len(co); r = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2], tris[:, 1], tris[:, 2], tris[:, 0]])
    c = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0], tris[:, 0], tris[:, 1], tris[:, 2]])
    Adj = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n)); Adj.data[:] = 1
    deg = np.asarray(Adj.sum(1)).ravel() + 1e-9
    hv = hair.astype(float)
    for _ in range(12): hv = 0.5 * hv + 0.5 * (Adj @ hv) / deg
    hv *= ~dump["helper"]
    hair = hv
    np.savez_compressed(out_path, proj_uv=uv, mask=m.astype(np.float32), skin_rgb=skin_rgb, hair=hair,
                        normals=nrm, co=co, tris=dump["tris"])
    return out_path, dict(face_vertices=int((m > 0.5).sum()), hair_vertices=int((hair > 0.5).sum()))


def style_deltas(dump, eye_centers, mouth_center, nose_center, chin_point):
    co = dump["basis"].astype(float)
    out = {}
    d = np.zeros_like(co)
    for c in eye_centers: d += R.local_scale_delta(co, c, 0.03, (1.18, 1.0, 1.18))
    out["style_big_eyes"] = d
    out["style_small_nose"] = R.local_scale_delta(co, nose_center, 0.035, (0.85, 0.92, 0.95))
    out["style_full_lips"] = R.local_scale_delta(co, mouth_center, 0.03, (0.97, 1.05, 1.2))
    # V-chin: pull the lower jaw sides inwards, more strongly near the chin
    w = np.clip((chin_point[2] + 0.07 - co[:, 2]) / 0.07, 0, 1) * (co[:, 2] > chin_point[2] - 0.02) * (co[:, 1] < 0.03)
    vd = np.zeros_like(co); vd[:, 0] = -co[:, 0] * 0.15 * w ** 1.2
    out["style_v_chin"] = vd
    return out
