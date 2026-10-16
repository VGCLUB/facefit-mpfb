"""Run inside Blender:  blender -b --python scripts/blender_headless_server.py -- [--port 9876]
Imports the add-on package from the repo (no install needed) and serves until '__shutdown__'."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blender_addon"))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 9876
from facefit_bridge import server, core
import bpy
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
core.create_human()
server.serve_blocking(port=port)
