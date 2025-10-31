#!/bin/bash
# Format code script for active_inference

set -e

echo "=== Formatting code ==="

echo ""
echo "1. Running black..."
black src/ tests/

echo ""
echo "2. Running isort..."
isort src/ tests/

echo ""
echo "3. Running ruff --fix..."
ruff check --fix src/ tests/

echo ""
echo "=== Formatting complete! ==="
