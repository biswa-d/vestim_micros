#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
    echo "[1/2] Creating virtual environment (one-time setup)..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install pandas numpy matplotlib scikit-learn scipy joblib
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo "Setup done."
    echo ""
fi

echo "[2/2] Running inference..."
.venv/bin/python run_offline_inference.py --output-dir inference_output "$@"

echo ""
echo "Done. Results in inference_output/"

