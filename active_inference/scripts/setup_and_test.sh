#!/bin/bash
# Complete setup and test validation script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd "${PROJECT_DIR}"

echo "=== Complete Setup and Test Validation ==="
echo "Project directory: ${PROJECT_DIR}"
echo ""

# 1. Run setup
echo "Step 1: Running setup..."
echo "=========================="
./scripts/setup.sh

# 2. Run tests
echo ""
echo "Step 2: Running comprehensive test suite..."
echo "============================================"
./scripts/run_tests.sh

echo ""
echo "=== Setup and validation complete ==="
echo "Check test_results/latest/ for detailed results"
