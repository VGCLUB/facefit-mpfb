"""FaceFit MCP server (stdio). Talks to Blender running the facefit_bridge add-on on FACEFIT_PORT (9876).

Run:  python -m facefit.mcp_server
"""
import json, os
try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
from .session import FaceFitSession

mcp = _Server("facefit")
_S = None


def S():
    global _S
    if _S is None:
        _S = FaceFitSession(host=os.environ.get("FACEFIT_HOST", "127.0.0.1"),
                            port=int(os.environ.get("FACEFIT_PORT", "9876")),
                            workdir=os.environ.get("FACEFIT_OUT"))
    return _S


@mcp.tool()
def load_reference_image(path: str) -> dict:
    """Load a reference face photo (front-facing works best)."""
    return S().load_reference_image(path)


@mcp.tool()
def detect_landmarks(path: str | None = None) -> dict:
    """Run MediaPipe Face Landmarker (CPU, 478 pts) on the reference image."""
    return S().detect_landmarks(path)


@mcp.tool()
def new_human(gender: float = 0.5, age: float = 0.5, asian: float | None = None) -> dict:
    """Replace the MPFB human with a fresh one (macro values 0..1; gender 0=female, 1=male)."""
    return {"object": S().bl.call("create_human", gender=gender, age=age, asian=asian)}


@mcp.tool()
def auto_fit(reg: float = 0.005, params: list[str] | None = None, style: dict[str, float] | None = None,
             locked: dict[str, float] | None = None, depth_weight: float = 0.3, fit_macros: bool = False,
             vertex_refine: bool = False, refine_passes: int = 2, full_head: bool | None = None) -> dict:
    """Solve MPFB face sliders (bounded least squares) to match reference landmarks.
    reg: ridge strength (higher = closer to default face). params: subset of slider names (default: ~48 frontal face sliders).
    style: offsets applied after fitting, e.g. {"eye-scale-decr-incr": 0.2, "chin-triangle": 0.4} for a more stylised look.
    locked: sliders held fixed at given values during the solve.
    depth_weight: weight of MediaPipe z depth residuals (0 = 2D only); enables nose/chin/mouth depth sliders.
    fit_macros: first search female/young MPFB macro presets (gender/age/ethnicity) and keep the best.
    vertex_refine: after sliders, add a smooth symmetric per-vertex deformation (shape key 'facefit_refine') that
    matches landmarks and the face-parsing jaw contour exactly (for likeness beyond slider range).
    full_head: also fit skull/forehead/ear outline + neck width (auto when no hair is parsed, e.g. bald references)."""
    out = {}
    if fit_macros:
        out["macros"] = S().fit_macros(depth_weight=depth_weight)
    out["sliders"] = S().auto_fit(params=params, reg=reg, style=style, locked=locked, depth_weight=depth_weight)
    if vertex_refine:
        r = S().vertex_refine(passes=refine_passes, full_head=full_head)
        out["vertex_refine"] = {"error_after": r["error_after"], "passes": r["passes"]}
        out["style_keys"] = S().add_style_keys()
    return out


@mcp.tool()
def list_parameters(section: str | None = None) -> list:
    """List MPFB face sliders (name, section, min, max). Sections: head, chin, eyes, eyebrows, nose, mouth, cheek, forehead, neck, ears."""
    p = S().list_parameters()
    return [x for x in p if section is None or x["section"] == section]


@mcp.tool()
def get_parameters(nonzero_only: bool = True) -> dict:
    """Current slider values."""
    return S().get_parameters(nonzero_only)


@mcp.tool()
def set_parameters(values: dict[str, float]) -> dict:
    """Set slider values, e.g. {"eye-scale-decr-incr": 0.3}. Pushes an undo step."""
    return S().set_parameters(values)


@mcp.tool()
def render_view(path: str | None = None, overlay: bool = False, markers: bool = False, res: int = 768) -> dict:
    """Render the head (front orthographic, Workbench/CPU). overlay=True warps the render onto the reference at 50% opacity;
    markers=True draws landmarks (green=reference, red=model). Returns PNG path."""
    return S().render_view(path, overlay=overlay, markers=markers, res=res)


@mcp.tool()
def get_landmark_error() -> dict:
    """2D landmark error after similarity alignment, in reference pixels and % of inter-ocular distance."""
    return S().get_landmark_error()


@mcp.tool()
def undo() -> dict:
    """Revert the last set_parameters / auto_fit."""
    return S().undo()


@mcp.tool()
def save_state(name: str = "default") -> dict:
    """Save current slider values (in memory and as out/state_<name>.json)."""
    return S().save_state(name)


@mcp.tool()
def load_state(name: str = "default", path: str | None = None) -> dict:
    """Restore a saved state by name or JSON path."""
    return S().load_state(name, path)


@mcp.tool()
def export_model(path: str, fmt: str | None = None) -> dict:
    """Export the fitted human to glb / obj / stl (format from extension if fmt omitted)."""
    return S().export_model(path, fmt)


@mcp.tool()
def project_texture(outdir: str | None = None, add_eyes: bool = True) -> dict:
    """Project the reference photo onto the fitted head's MPFB UVs. Face parsing (BiSeNet, CPU) splits skin/brows/lips/eyes
    masks; hair/background are never projected; unseen texels are filled with a border-matched skin colour.
    Also (re)fits MakeHuman eyes and recolours the iris to the reference. Applies the material in Blender. Run after auto_fit."""
    return S().project_texture(outdir=outdir, add_eyes=add_eyes)


@mcp.tool()
def project_face_photo(hair_cap: bool = True, delight: float = 0.3) -> dict:
    """hair_cap=False for bald references."""
    return S().project_face(hair_cap=hair_cap, delight=delight)


@mcp.tool()
def calibrate_eyes(rounds: int = 2) -> dict:
    """Scale/move the eyeballs so iris size and centre in a render match the photo (MediaPipe iris landmarks)."""
    return {"history": S().calibrate_eyes(rounds=rounds)}


@mcp.tool()
def silhouette_iou() -> dict:
    """IoU of projected head mesh vs parsed head mask (face+ears) above the chin line."""
    return {"iou": S().silhouette_iou()}


@mcp.tool()
def set_style(values: dict[str, float]) -> dict:
    """Stylisation shape keys (created by auto_fit(vertex_refine=True)): style_big_eyes, style_small_nose,
    style_full_lips, style_v_chin. Values ~0..1.5."""
    return S().set_style(values)


@mcp.tool()
def render_textured(path: str, yaw_deg: float = 0.0, res: int = 768, samples: int = 48) -> dict:
    """Cycles CPU render of the textured head. yaw_deg=35 gives a 3/4 view."""
    return S().render_textured(path, yaw_deg=yaw_deg, res=res, samples=samples)


@mcp.tool()
def get_depth_error() -> dict:
    """RMS depth error (% IOD) of mapped vertices vs MediaPipe z (free depth scale/offset; jaw+eyelids excluded)."""
    return {"depth_rms_iod_pct": S().get_depth_error()}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
