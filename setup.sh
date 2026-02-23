#!/usr/bin/env bash
# setup.sh — bootstrap the development environment for chest-xray-sim
set -euo pipefail

# ── 1. Python version check ────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.11+ first." >&2
    exit 1
fi

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    version=$(python3 --version 2>&1)
    echo "Error: Python 3.11+ required (found $version)." >&2
    exit 1
fi

echo "Python: $(python3 --version)"

# ── 2. uv install (if missing) ────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing via official script..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv: $(uv --version)"

# ── 3. Install dependencies ───────────────────────────────────────────────
echo "Running uv sync..."
uv sync

# ── 4. Register Jupyter kernel ────────────────────────────────────────────
echo "Registering Jupyter kernel 'chest-xray-sim'..."
uv run python -m ipykernel install --user \
    --name chest-xray-sim \
    --display-name "Python (chest-xray-sim)"

echo ""
echo "Setup complete."
echo "  Start JupyterLab:  make jupyter"
echo "  Run tests:         make test"
