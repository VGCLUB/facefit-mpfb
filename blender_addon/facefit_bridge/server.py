"""Tiny JSON-over-TCP command server (blender-mcp style). One JSON object per line.
Request: {"cmd": "...", "args": {...}}   Response: {"ok": true, "result": ...} | {"ok": false, "error": "..."}
"""
import json, socket, traceback
from . import core

HOST, PORT = "127.0.0.1", 9876


def _ping(): return {"pong": True}

COMMANDS = {
    "ping": _ping,
    "create_human": lambda gender=0.5, age=0.5, asian=None: (core.create_human(gender, age, asian).name),
    "list_parameters": core.list_parameters,
    "get_parameters": lambda names=None: core.get_parameters(names=names),
    "set_parameters": lambda values: core.set_parameters(values),
    "reset_parameters": lambda: core.reset_parameters(),
    "linear_model": lambda vertex_ids, param_names: core.linear_model(vertex_ids, param_names),
    "render": lambda path, res=768, engine="BLENDER_WORKBENCH": core.render(path, res, engine),
    "project_vertices": lambda vertex_ids, res=768: (core.setup_camera(res=res), core.project_vertices(vertex_ids))[1],
    "raycast_pixels": lambda pixels, res=768: core.raycast_pixels(pixels, res),
    "save_blend": lambda path: core.save_blend(path),
    "vertex_coords": lambda vertex_ids: core.evaluated_coords(vertex_ids=vertex_ids).tolist(),
    "add_eyes": lambda mhclo=None: core.add_eyes(mhclo),
    "uv_projection_data": lambda path, res=768: (core.setup_camera(res=res), core.uv_projection_data(path))[1],
    "apply_textures": lambda albedo, eyes_albedo=None: core.apply_textures(albedo, eyes_albedo),
    "render_textured": lambda path, res=768, yaw_deg=0.0, samples=48: core.render_textured(path, res, yaw_deg, samples),
    "export_model": lambda path, fmt="glb": core.export_model(path, fmt),
}


def handle(line):
    try:
        req = json.loads(line)
        fn = COMMANDS[req["cmd"]]
        return {"ok": True, "result": fn(**(req.get("args") or {}))}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}


def serve_blocking(host=HOST, port=PORT):
    """Headless mode (blender -b): serve forever in the main thread."""
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port)); srv.listen(4)
    print(f"[facefit] listening on {host}:{port}", flush=True)
    while True:
        conn, _ = srv.accept()
        with conn, conn.makefile("rwb") as f:
            for line in f:
                if line.strip() == b"__shutdown__":
                    print("[facefit] shutdown", flush=True); return
                f.write((json.dumps(handle(line)) + "\n").encode()); f.flush()


class TimerServer:
    """GUI mode: non-blocking socket polled by bpy.app.timers so bpy runs on the main thread."""
    def __init__(self, host=HOST, port=PORT):
        self.host, self.port, self.srv, self.clients = host, port, None, {}

    def start(self):
        import bpy
        self.srv = socket.socket(); self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((self.host, self.port)); self.srv.listen(4); self.srv.setblocking(False)
        bpy.app.timers.register(self._tick, persistent=True)

    def stop(self):
        import bpy
        if bpy.app.timers.is_registered(self._tick): bpy.app.timers.unregister(self._tick)
        for c in list(self.clients): c.close()
        self.clients.clear()
        if self.srv: self.srv.close(); self.srv = None

    def _tick(self):
        try:
            while True:
                c, _ = self.srv.accept(); c.setblocking(False); self.clients[c] = b""
        except BlockingIOError:
            pass
        for c in list(self.clients):
            try:
                data = c.recv(1 << 20)
                if not data: c.close(); del self.clients[c]; continue
                self.clients[c] += data
                while b"\n" in self.clients[c]:
                    line, self.clients[c] = self.clients[c].split(b"\n", 1)
                    c.setblocking(True); c.sendall((json.dumps(handle(line)) + "\n").encode()); c.setblocking(False)
            except BlockingIOError:
                pass
            except OSError:
                c.close(); self.clients.pop(c, None)
        return 0.05
