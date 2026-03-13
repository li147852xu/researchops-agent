#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== ResearchOps Agent — Quick Start ==="
echo ""

# 1. Check Python version
PYTHON="${PYTHON:-python3}"
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
if [[ -z "$PY_VERSION" ]]; then
    echo "ERROR: Python 3.11+ is required but python3 was not found."
    exit 1
fi
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
    echo "ERROR: Python 3.11+ is required (found $PY_VERSION)."
    exit 1
fi
echo "[1/5] Python $PY_VERSION — OK"

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
