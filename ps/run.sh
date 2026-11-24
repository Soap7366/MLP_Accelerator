#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${PYNQ_PYTHON:-/usr/local/share/pynq-venv/bin/python3}
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
if [ "$(id -u)" != 0 ]; then
    exec sudo -E "$PYTHON" -u demo_modes.py "$@"
fi
exec "$PYTHON" -u demo_modes.py "$@"
