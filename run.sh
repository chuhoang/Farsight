#!/usr/bin/env bash
# Run anything in the project env (WSL; Windows wheels are blocked by the corporate gateway).
# Usage from Windows:  wsl -d Ubuntu-22.04 -- bash run.sh pytest -q farsight
cd "$(dirname "$0")" && export PYTHONPATH="$PWD" && exec ~/fsenv/bin/python -m "$@"
