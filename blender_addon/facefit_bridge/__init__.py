bl_info = {
    "name": "FaceFit Bridge (MPFB2 face fitting via MCP)",
    "author": "FaceFit contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "category": "Interface",
    "description": "Socket bridge so an MCP server can fit an MPFB2 head to a reference photo",
}
import bpy
from . import server

_SRV = None


class FACEFIT_OT_start(bpy.types.Operator):
    bl_idname = "facefit.start_server"; bl_label = "Start FaceFit server"
    def execute(self, ctx):
        global _SRV
        if _SRV is None:
            _SRV = server.TimerServer(port=ctx.scene.facefit_port); _SRV.start()
        self.report({"INFO"}, f"FaceFit listening on {_SRV.port}"); return {"FINISHED"}


class FACEFIT_OT_stop(bpy.types.Operator):
    bl_idname = "facefit.stop_server"; bl_label = "Stop FaceFit server"
    def execute(self, ctx):
        global _SRV
        if _SRV: _SRV.stop(); _SRV = None
        return {"FINISHED"}


class FACEFIT_PT_panel(bpy.types.Panel):
    bl_label = "FaceFit"; bl_space_type = "VIEW_3D"; bl_region_type = "UI"; bl_category = "FaceFit"
    def draw(self, ctx):
        l = self.layout; l.prop(ctx.scene, "facefit_port")
        l.label(text="running" if _SRV else "stopped")
        l.operator("facefit.start_server"); l.operator("facefit.stop_server")


CLASSES = (FACEFIT_OT_start, FACEFIT_OT_stop, FACEFIT_PT_panel)


def register():
    bpy.types.Scene.facefit_port = bpy.props.IntProperty(name="Port", default=9876)
    for c in CLASSES: bpy.utils.register_class(c)


def unregister():
    if _SRV: _SRV.stop()
    for c in reversed(CLASSES): bpy.utils.unregister_class(c)
    del bpy.types.Scene.facefit_port
