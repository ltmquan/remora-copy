#!/usr/bin/env bash
# Run on the rented Linux host from the uploaded experimental/scaling directory.
set -euo pipefail
cd -- "$(dirname -- "$0")"
python_bin="${PYTHON_BIN:-python3.13}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,13), "Use Python 3.13 for the comparison environment"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install tinygrad==0.14.0 numpy==2.5.3
.venv/bin/python -m pip freeze > requirements-resolved.txt
.venv/bin/python preflight.py --output results/preflight.json
