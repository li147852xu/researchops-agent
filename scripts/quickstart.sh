#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== ResearchOps Agent — Quick Start ==="
echo ""

# 1. Find a suitable Python >= 3.11
_check_python() {
    local py="$1"
    local ver
    ver=$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || return 1
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 11 ]]; then
        echo "$ver"
        return 0
    fi
    return 1
}

PYTHON=""
PY_VERSION=""

# Try user-specified, then versioned executables, then generic
for candidate in "${PYTHON:-}" python3.13 python3.12 python3.11 python3 python; do
    [[ -z "$candidate" ]] && continue
    if ver=$(_check_python "$candidate"); then
        PYTHON="$candidate"
        PY_VERSION="$ver"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.11+ is required but no suitable interpreter was found."
    echo ""
    echo "Tried: python3.13, python3.12, python3.11, python3, python"
    echo "Set the PYTHON env var to point to a Python 3.11+ executable, e.g.:"
    echo "  PYTHON=/path/to/python3.11 make quickstart"
    if command -v conda &>/dev/null; then
        echo ""
        echo "Conda detected — you can create a compatible environment:"
        echo "  conda create -n researchops python=3.11 -y"
        echo "  conda activate researchops"
        echo "  make quickstart"
    fi
    exit 1
fi

echo "[1/5] Python $PY_VERSION ($PYTHON) — OK"

# 2. Create virtualenv if not present
if [[ ! -d ".venv" ]]; then
    echo "[2/5] Creating virtual environment..."
    $PYTHON -m venv .venv
else
    echo "[2/5] Virtual environment exists — OK"
fi
source .venv/bin/activate

# 3. Install the package with dev + embeddings extras
echo "[3/5] Installing researchops with dev + embeddings extras..."
pip install -q -e ".[dev,embeddings]"

# 4. Copy .env.example if .env does not exist
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    echo "[4/5] Created .env from .env.example — edit it to add your API keys"
else
    echo "[4/5] .env already exists — OK"
fi

# 5. Run tests to verify setup
echo "[5/5] Running test suite..."
if pytest -q --tb=short; then
    echo ""
    echo "=== All tests passed! ==="
else
    echo ""
    echo "WARNING: Some tests failed. Check the output above."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick demo (offline, no API key needed):"
echo "  source .venv/bin/activate"
echo "  make demo"
echo ""
echo "Full LLM-powered run (requires API key in .env):"
echo "  source .venv/bin/activate"
echo "  researchops run \"your research topic\" \\"
echo "    --llm openai_compat \\"
echo "    --llm-base-url https://api.deepseek.com/v1 \\"
echo "    --llm-model deepseek-chat \\"
echo "    --llm-api-key \$DEEPSEEK_API_KEY \\"
echo "    --sources hybrid --retrieval hybrid --graph --judge"
echo ""
