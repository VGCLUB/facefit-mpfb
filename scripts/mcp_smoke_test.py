"""Drive the MCP server over stdio like Codex would (official mcp client)."""
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

async def main():
    p = StdioServerParameters(command=sys.executable, args=["-m", "facefit.mcp_server"], cwd=ROOT)
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools(); print("tools:", [t.name for t in tools.tools])
            async def call(n, **a):
                res = await s.call_tool(n, a); txt = res.content[0].text if res.content else ""
                print(f"{n} ->", txt[:300].replace("\n", " ")); return txt
            await call("load_reference_image", path=os.environ.get("FACEFIT_TEST_IMAGE", os.path.join(ROOT, "testdata/guoman_face.png")))
            await call("detect_landmarks")
            await call("set_parameters", values={"eye-scale-decr-incr": 0.0})
            await call("get_landmark_error")
            await call("auto_fit", style={"eye-scale-decr-incr": 0.15, "chin-triangle": 0.3})
            await call("get_landmark_error")
            await call("render_view", path=os.path.join(ROOT, "out/mcp_stylized_overlay.png"), overlay=True, markers=True)
            await call("undo"); await call("get_landmark_error")
            await call("list_parameters", section="chin")
            await call("project_texture")
asyncio.run(main())
