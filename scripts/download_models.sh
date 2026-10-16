#!/usr/bin/env bash
# Downloads third-party models/assets (not redistributed in this repo).
set -e
cd "$(dirname "$0")/.."
mkdir -p models testdata
curl -fL -o testdata/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task   # Apache-2.0
curl -fL -o models/bisenet_resnet_34.onnx https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx   # face parsing
# MakeHuman eyes (CC0) from the MakeHuman GitHub repo -> MPFB user data (workaround when the asset-pack server is slow)
MPFB_DATA=${MPFB_DATA:-$HOME/.config/blender/4.2/extensions/.user/user_default/mpfb/data}
tmp=$(mktemp -d); git clone -q --depth 1 --filter=blob:none --sparse https://github.com/makehumancommunity/makehuman "$tmp/mh"
(cd "$tmp/mh" && git sparse-checkout set makehuman/data/eyes)
mkdir -p "$MPFB_DATA" && cp -r "$tmp/mh/makehuman/data/eyes" "$MPFB_DATA/" && rm -rf "$tmp"
echo "done"
