#!/usr/bin/env bash
set -euo pipefail

echo "=== ResearchOps Web UI ==="
echo "Installing web dependencies..."
pip install -e ".[web]" --quiet 2>/dev/null || pip install -e ".[web]"

echo "Launching at http://0.0.0.0:7860 ..."
researchops web
