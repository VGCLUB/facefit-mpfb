#!/usr/bin/env bash
# Start Blender headless with the FaceFit bridge (MPFB extension must be installed in Blender's user extensions).
BLENDER=${BLENDER:-blender}
exec "$BLENDER" -b -noaudio --python "$(dirname "$0")/blender_headless_server.py" -- --port "${FACEFIT_PORT:-9876}"
