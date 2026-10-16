"""FaceFit Blender-side core: runs INSIDE Blender (>=4.2) with the MPFB2 extension enabled.

Responsibilities: create/locate the MPFB human, expose face sliders as signed parameters,
evaluate them, render a front orthographic view, project mapped vertices, provide the
linear (shape-key) model used by the external least-squares solver, and export.
Heavy numerics (landmarks, solving) live outside Blender in the MCP server process.
"""
import importlib, json, os, sys, math
import bpy
import numpy as np

FACE_SECTIONS = ["head", "chin", "eyes", "eyebrows", "nose", "mouth", "cheek", "forehead", "neck", "ears"]
CAM_NAME = "FaceFitCam"


def _imp(path, key):
    for m in list(sys.modules):
        if m.endswith(path):
            mod = importlib.import_module(m)
            if hasattr(mod, key):
                return getattr(mod, key)
    raise RuntimeError(f"MPFB module {path} not loaded - is the MPFB extension enabled?")


def mpfb():
    return dict(
        HumanService=_imp("mpfb.services.humanservice", "HumanService"),
        TargetService=_imp("mpfb.services.targetservice", "TargetService"),
        LocationService=_imp("mpfb.services.locationservice", "LocationService"),
        ObjectService=_imp("mpfb.services.objectservice", "ObjectService"),
        HumanObjectProperties=_imp("mpfb.entities.objectproperties", "HumanObjectProperties"),
    )


# ---------------------------------------------------------------- parameters
_PARAMS = None


def parameter_registry():
    """Build signed face parameters from MPFB's bundled target.json.

    Each MPFB 'category' becomes one parameter. Two-sided categories map to [-1, 1]
    (negative -> 'decr' target(s), positive -> 'incr' target(s)); left/right targets are
    driven symmetrically. Single-target categories map to [0, 1].
    """
    global _PARAMS
    if _PARAMS is not None:
        return _PARAMS
    LS = mpfb()["LocationService"]
    with open(os.path.join(LS.get_mpfb_data("targets"), "target.json")) as f:
        meta = json.load(f)
    params = []
    for sec in FACE_SECTIONS:
        for cat in meta.get(sec, {}).get("categories", []):
            op = cat.get("opposites") or {}
            if op:
                neg = [t for t in (op.get("negative-unsided"), op.get("negative-left"), op.get("negative-right")) if t]
                pos = [t for t in (op.get("positive-unsided"), op.get("positive-left"), op.get("positive-right")) if t]
                params.append(dict(name=cat["name"], section=sec, min=-1.0, max=1.0, neg=neg, pos=pos))
            else:
                params.append(dict(name=cat["name"], section=sec, min=0.0, max=1.0, neg=[], pos=list(cat["targets"])))
    _PARAMS = {p["name"]: p for p in params}
    return _PARAMS


# ---------------------------------------------------------------- human
def get_human(create=True):
    for o in bpy.data.objects:
        if o.type == "MESH" and o.get("facefit_human"):
            return o
    if not create:
        return None
    return create_human()


def create_human(gender=0.5, age=0.5, asian=None):
    M = mpfb()
    for o in list(bpy.data.objects):
        if o.get("facefit_human"):
            bpy.data.objects.remove(o)
    h = M["HumanService"].create_human()
    h["facefit_human"] = True
    HOP = M["HumanObjectProperties"]
    HOP.set_value("gender", float(gender), entity_reference=h)
    HOP.set_value("age", float(age), entity_reference=h)
    if asian is not None:
        a = float(asian)
        HOP.set_value("asian", a, entity_reference=h)
        HOP.set_value("caucasian", (1 - a) / 2, entity_reference=h)
        HOP.set_value("african", (1 - a) / 2, entity_reference=h)
    M["TargetService"].reapply_macro_details(h)
    return h


def _key_blocks(h):
    return {k.name: k for k in h.data.shape_keys.key_blocks}


def ensure_targets(h, names):
    """Load MPFB target files as shape keys (weight 0) if not present. Returns {target: shapekey_name}."""
    TS = mpfb()["TargetService"]
    kb = _key_blocks(h)
    out = {}
    for t in names:
        path = TS.target_full_path(t)
        if not path:
            continue
        skn = TS.filename_to_shapekey_name(path)
        if skn not in kb:
            sk = TS.load_target(h, path, weight=0.0, name=skn)
            kb[skn] = sk
        out[t] = skn
    return out


def ensure_all_param_targets(h, param_names=None):
    reg = parameter_registry()
    names = param_names or list(reg)
    tl = sorted({t for n in names for t in reg[n]["neg"] + reg[n]["pos"]})
    return ensure_targets(h, tl)


def list_parameters():
    return [dict(name=p["name"], section=p["section"], min=p["min"], max=p["max"]) for p in parameter_registry().values()]


def get_parameters(h=None, names=None):
    h = h or get_human()
    reg = parameter_registry()
    kb = _key_blocks(h)
    TS = mpfb()["TargetService"]
    vals = {}
    for n in (names or reg):
        p = reg[n]
        def v(ts):
            for t in ts:
                path = TS.target_full_path(t)
                skn = TS.filename_to_shapekey_name(path) if path else t
                if skn in kb:
                    return kb[skn].value
            return 0.0
        vals[n] = round(v(p["pos"]) - v(p["neg"]), 5)
    return vals


def set_parameters(values, h=None):
    h = h or get_human()
    reg = parameter_registry()
    unknown = [n for n in values if n not in reg]
    names = [n for n in values if n in reg]
    m = ensure_all_param_targets(h, names)
    kb = _key_blocks(h)
    for n in names:
        p = reg[n]
        x = max(p["min"], min(p["max"], float(values[n])))
        for t in p["pos"]:
            if t in m: kb[m[t]].value = max(x, 0.0)
        for t in p["neg"]:
            if t in m: kb[m[t]].value = max(-x, 0.0)
    h.data.update()
    return dict(set=len(names), unknown=unknown)


def reset_parameters(h=None):
    h = h or get_human()
    reg = parameter_registry()
    return set_parameters({n: 0.0 for n in reg}, h)


# ---------------------------------------------------------------- linear model for the solver
def linear_model(vertex_ids, param_names, h=None):
    """Return rest positions (all non-fit shape keys at current values) and per-target deltas."""
    h = h or get_human()
    reg = parameter_registry()
    m = ensure_all_param_targets(h, param_names)
    kbs = h.data.shape_keys.key_blocks
    n = len(h.data.vertices)
    idx = np.asarray(vertex_ids, dtype=np.int64)
    def co(k):
        a = np.empty(n * 3, dtype=np.float32); k.data.foreach_get("co", a); return a.reshape(-1, 3)[idx].astype(np.float64)
    basis = co(kbs[0])
    fit_keys = {m[t] for pn in param_names for t in reg[pn]["neg"] + reg[pn]["pos"] if t in m}
    rest = basis.copy()
    for k in kbs[1:]:
        if k.name in fit_keys or k.mute or k.value == 0.0:
            continue
        rest += k.value * (co(k) - basis)
    deltas = {}
    for pn in param_names:
        p = reg[pn]
        deltas[pn] = {"neg": [(co(kbs[m[t]]) - basis).tolist() for t in p["neg"] if t in m],
                      "pos": [(co(kbs[m[t]]) - basis).tolist() for t in p["pos"] if t in m]}
    return dict(rest=rest.tolist(), deltas=deltas,
                params=[dict(name=pn, min=reg[pn]["min"], max=reg[pn]["max"]) for pn in param_names])


def evaluated_coords(h=None, vertex_ids=None):
    h = h or get_human()
    dg = bpy.context.evaluated_depsgraph_get()
    # shape keys only (no modifiers) to keep vertex indices stable
    kbs = h.data.shape_keys.key_blocks
    n = len(h.data.vertices)
    basis = np.empty(n * 3, np.float32); kbs[0].data.foreach_get("co", basis); basis = basis.reshape(-1, 3)
    out = basis.astype(np.float64).copy()
    for k in kbs[1:]:
        if k.mute or k.value == 0.0: continue
        a = np.empty(n * 3, np.float32); k.data.foreach_get("co", a)
        out += k.value * (a.reshape(-1, 3) - basis)
    mw = np.array(h.matrix_world)
    out = out @ mw[:3, :3].T + mw[:3, 3]
    return out if vertex_ids is None else out[np.asarray(vertex_ids)]


# ---------------------------------------------------------------- camera / render
def setup_camera(h=None, res=768, ortho_scale=0.32, center_offset=0.13):
    h = h or get_human()
    sc = bpy.context.scene
    cam = bpy.data.objects.get(CAM_NAME)
    if cam is None:
        cd = bpy.data.cameras.new(CAM_NAME)
        cam = bpy.data.objects.new(CAM_NAME, cd)
        sc.collection.objects.link(cam)
    if "facefit_top" not in h:  # freeze framing on first use so before/after are comparable
        h["facefit_top"] = float(evaluated_coords(h)[:, 2].max())
    top = h["facefit_top"]
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ortho_scale
    cam.location = (h.matrix_world.translation.x, -3.0, top - center_offset)
    cam.rotation_euler = (math.pi / 2, 0, 0)
    sc.camera = cam
    sc.render.resolution_x = sc.render.resolution_y = int(res)
    sc.render.resolution_percentage = 100
    return cam


def render(path, res=768, engine="BLENDER_WORKBENCH"):
    h = get_human()
    setup_camera(h, res=res)
    sc = bpy.context.scene
    sc.render.engine = engine
    if engine == "BLENDER_WORKBENCH":
        sh = sc.display.shading
        sh.light = "STUDIO"; sh.color_type = "SINGLE"; sh.single_color = (0.85, 0.72, 0.65)
        sh.show_cavity = True; sh.cavity_type = "WORLD"
        sc.display.render_aa = "8"
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def project_vertices(vertex_ids, h=None):
    """Pixel coords (x right, y down) of vertices in the FaceFitCam render."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    h = h or get_human()
    cam = setup_camera(h, res=bpy.context.scene.render.resolution_x)
    sc = bpy.context.scene
    W, H = sc.render.resolution_x, sc.render.resolution_y
    pts = evaluated_coords(h, vertex_ids)
    out = []
    for p in pts:
        c = world_to_camera_view(sc, cam, Vector(p))
        out.append([c.x * W, (1 - c.y) * H])
    return out


def raycast_pixels(pixels, res):
    """For each render pixel, cast a ray from the ortho camera; return nearest mesh vertex id (or -1)."""
    h = get_human()
    cam = setup_camera(h, res=res)
    dg = bpy.context.evaluated_depsgraph_get()
    from mathutils import Vector
    s = cam.data.ortho_scale
    out = []
    mesh = h.data
    for (px, py) in pixels:
        x = cam.location.x + (px / res - 0.5) * s
        z = cam.location.z - (py / res - 0.5) * s
        hit, loc, nrm, fidx, obj, mat = bpy.context.scene.ray_cast(dg, Vector((x, -3.0, z)), Vector((0, 1, 0)))
        if not hit or obj.name != h.name:
            out.append(-1); continue
        # evaluated mesh (mask modifier) may renumber faces: use nearest original vertex to hit point
        out.append(int(loc_to_vertex(h, loc)))
    return out


_KD = None
def loc_to_vertex(h, loc):
    global _KD
    from mathutils.kdtree import KDTree
    if _KD is None or _KD[0] != h.name:
        co = evaluated_coords(h)
        # exclude helper geometry vertices
        helper = set()
        g = h.vertex_groups.get("HelperGeometry")
        if g:
            gi = g.index
            for v in h.data.vertices:
                if any(e.group == gi for e in v.groups): helper.add(v.index)
        kd = KDTree(len(co))
        for i, c in enumerate(co):
            if i not in helper: kd.insert(c, i)
        kd.balance()
        _KD = (h.name, kd)
    return _KD[1].find(loc)[1]


# ---------------------------------------------------------------- state / export
def save_blend(path):
    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    return path


def export_model(path, fmt):
    h = get_human()
    bpy.ops.object.select_all(action="DESELECT")
    h.select_set(True)
    bpy.context.view_layer.objects.active = h
    fmt = fmt.lower()
    if fmt == "glb":
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True)
    elif fmt == "obj":
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True, apply_modifiers=True, export_materials=False)
    elif fmt == "stl":
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, apply_modifiers=True)
    else:
        raise ValueError("fmt must be glb/obj/stl")
    return path
