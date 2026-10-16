# FaceFit — MPFB2 head ↔ reference-photo fitting via MCP

> 状态 / Status: **prototype (v0.1)**。在无 GPU 的 Linux 上用 Blender 4.2.3 LTS（Workbench 软件渲染）+ MPFB2 2.0.18 + MediaPipe（CPU）测试通过。

## 中文简介
FaceFit 是一个开源的 Blender 插件加 MCP 服务器。它把 MPFB2（MakeHuman for Blender）人头的面部滑杆拟合到一张正脸参考图上（目标是国漫 CG 风格的半写实脸），Codex 之类的外部 agent 可以通过 MCP 驱动整个拟合流程。

流程：MediaPipe Face Landmarker 检测 478 个点 → 取 57 个稳定点（眼角、眼睑、眉、鼻、嘴、下颌轮廓）→ 对应到 MPFB 基础网格顶点（`data/mp_to_mpfb.json`，由 `scripts/calibrate_correspondences.py` 自动标定）→ 正交正视投影 + 相似变换对齐 → 用 scipy 有界线性最小二乘（`lsq_linear`，带岭正则）交替求解约 48 个面部滑杆。

## English
Blender add-on (`blender_addon/facefit_bridge`, a JSON socket bridge in the style of blender-mcp) plus a Python MCP server (`facefit/mcp_server.py`, official `mcp` SDK, stdio). Landmark detection and solving run in the MCP server process. Blender handles the MPFB mesh, shape keys, rendering and export.

## Install
```bash
# 1. Blender >= 4.2 (tested 4.2.3 LTS, headless OK, no GPU needed for Workbench)
curl -LO https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz && tar xf blender-4.2.3-linux-x64.tar.xz
# Debian/Ubuntu headless libs: sudo apt install libegl1 libgl1 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6
# 2. MPFB2 as a Blender extension (bundled base mesh + targets are enough for fitting)
git clone --depth 1 https://github.com/makehumancommunity/mpfb2
(cd mpfb2/src && zip -r ../../mpfb.zip mpfb)
blender -b --command extension install-file -r user_default -e mpfb.zip
# 3. Python env for MCP server
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
curl -L -o testdata/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
Optional MPFB "system assets" pack (eyes/eyebrows/skins): <https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html>. Fitting does not need it. On our test box the 280 MB download ran at about 12 KB/s, so we skipped it and render untextured clay.

## Run
```bash
BLENDER=/path/to/blender scripts/run_blender_headless.sh &      # headless bridge on 127.0.0.1:9876
.venv/bin/python scripts/e2e_test.py testdata/your_face.png     # -> out/before.png after.png overlay.png report.json fitted.{glb,obj,stl}
.venv/bin/python scripts/mcp_smoke_test.py                      # drives the MCP server over stdio
```
GUI alternative: install `blender_addon/facefit_bridge` as an add-on/extension, then open the N-panel ▸ FaceFit ▸ Start server.

## Codex MCP config (`~/.codex/config.toml`)
```toml
[mcp_servers.facefit]
command = "/path/to/facefit/.venv/bin/python"
args = ["-m", "facefit.mcp_server"]
cwd = "/path/to/facefit"
env = { FACEFIT_PORT = "9876", FACEFIT_OUT = "/path/to/facefit/out" }
```
(Equivalent: `codex mcp add facefit -- /path/to/facefit/.venv/bin/python -m facefit.mcp_server`, which must run with the repo as cwd.)

## MCP tools
`load_reference_image`, `detect_landmarks`, `new_human`, `auto_fit(reg, params, style, locked)`, `list_parameters`, `get_parameters`, `set_parameters`, `render_view(overlay, markers)`, `get_landmark_error`, `undo`, `save_state`, `load_state`, `export_model(glb|obj|stl)`, `project_texture` (**stub, phase 2**).

Stylisation: `auto_fit(style={"eye-scale-decr-incr": 0.2, "chin-triangle": 0.4})` adds offsets after the fit (bigger eyes, pointier chin).

## Result on our private test image (front-facing guoman CG face, not distributed)
| metric (57 pts, similarity-aligned) | before | after |
|---|---|---|
| vertex-correspondence RMS (px / %IOD) | 8.85 / 5.49% | 4.95 / 3.07% |
| independent check: MediaPipe re-run on render | 8.64 / 5.36% | 5.48 / 3.40% |

## Limitations / TODO
- 2D frontal only. Depth sliders (nose projection, chin prominence) are not constrained. Next step: add a profile view or MediaPipe z.
- Correspondences come from MediaPipe run on a clay render of the default MPFB head (about 4 px reprojection error at 1024 px). Jaw-contour points are silhouette-dependent and only approximate.
- MPFB slider ranges cannot reach extreme anime proportions (very large eyes). The residual error is mostly from the eyes and jaw.
- Texture projection, hair, eyes, eyebrow assets: phase 2.
- Licenses: code GPL-3.0-or-later (it drives MPFB, which is GPL). MPFB assets are CC0. The MediaPipe model is Apache-2.0.
