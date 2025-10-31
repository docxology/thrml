#!/bin/bash
# Development check script for active_inference

set -e

echo "=== Running development checks ==="

echo ""
echo "1. Running pytest..."
pytest tests/ -v --cov=active_inference --cov-report=term-missing

echo ""
echo "2. Running black formatter..."
black --check src/ tests/

echo ""
echo "3. Running isort..."
isort --check-only src/ tests/

echo ""
echo "4. Running ruff linter..."
ruff check src/ tests/

echo ""
echo "5. Running type checker..."
pyright src/

echo ""
echo "=== All checks passed! ==="
