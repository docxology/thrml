#!/bin/bash
# Setup script for active_inference using uv

set -e

echo "=== Setting up active_inference development environment ==="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo ""
echo "1. Creating virtual environment with uv..."
uv venv

echo ""
echo "2. Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "3. Installing package with dependencies..."
uv pip install -e ".[all]"

echo ""
echo "4. Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
else
    echo "pre-commit not found, skipping hook installation"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate your environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Run tests with:"
echo "  pytest tests/ -v"
echo ""
echo "Run checks with:"
echo "  ./scripts/check.sh"
