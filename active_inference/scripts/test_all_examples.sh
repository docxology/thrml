#!/bin/bash
# Run all examples and generate comprehensive test report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/../examples" && pwd)"
OUTPUT_DIR="$(cd "$SCRIPT_DIR/../output" && pwd)"

echo "============================================================"
echo "  COMPREHENSIVE EXAMPLE TEST SUITE"
echo "============================================================"
echo ""
echo "Examples directory: $EXAMPLES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Find all example files
EXAMPLES=($(ls $EXAMPLES_DIR/[0-9][0-9]_*.py | sort))
TOTAL=${#EXAMPLES[@]}

echo "Found $TOTAL examples to test"
echo ""

# Track results
PASSED=0
FAILED=0
FAILED_EXAMPLES=()

# Run each example
for EXAMPLE in "${EXAMPLES[@]}"; do
    EXAMPLE_NAME=$(basename "$EXAMPLE" .py)
    echo "────────────────────────────────────────────────────────────"
    echo "Testing: $EXAMPLE_NAME"
    echo "────────────────────────────────────────────────────────────"

    if python3 "$EXAMPLE" > /dev/null 2>&1; then
        echo "✓ PASSED: $EXAMPLE_NAME"
        ((PASSED++))
    else
        echo "✗ FAILED: $EXAMPLE_NAME"
        ((FAILED++))
        FAILED_EXAMPLES+=("$EXAMPLE_NAME")
    fi
    echo ""
done

# Summary
echo "============================================================"
echo "  TEST SUMMARY"
echo "============================================================"
echo ""
echo "Total examples:  $TOTAL"
echo "Passed:          $PASSED"
echo "Failed:          $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ ALL TESTS PASSED"
    exit 0
else
    echo "✗ SOME TESTS FAILED:"
    for FAILED_EXAMPLE in "${FAILED_EXAMPLES[@]}"; do
        echo "  - $FAILED_EXAMPLE"
    done
    exit 1
fi
