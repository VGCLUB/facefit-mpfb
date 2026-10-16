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
             locked: dict[str, float] | None = None, depth_weight: float = 0.3) -> dict:
    """Solve MPFB face sliders (bounded least squares) to match reference landmarks.
    reg: ridge strength (higher = closer to default face). params: subset of slider names (default: ~48 frontal face sliders).
    style: offsets applied after fitting, e.g. {"eye-scale-decr-incr": 0.2, "chin-triangle": 0.4} for a more stylised look.
    locked: sliders held fixed at given values during the solve.
    depth_weight: weight of MediaPipe z depth residuals (0 = 2D only); enables nose/chin/mouth depth sliders."""
    return S().auto_fit(params=params, reg=reg, style=style, locked=locked, depth_weight=depth_weight)


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
