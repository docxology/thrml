#!/bin/bash
# Comprehensive test runner with logging and reporting

set -e
set -o pipefail  # Catch errors in piped commands

# Configuration
TEST_OUTPUT_DIR="test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${TEST_OUTPUT_DIR}/run_${TIMESTAMP}"

echo "=== Active Inference Test Suite Runner ==="
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root for all test operations
cd "${PROJECT_ROOT}"

# Create output directories first
mkdir -p "${PROJECT_ROOT}/${RUN_DIR}/logs"
mkdir -p "${PROJECT_ROOT}/${RUN_DIR}/coverage"
mkdir -p "${PROJECT_ROOT}/${RUN_DIR}/reports"

# Log file
MAIN_LOG="${PROJECT_ROOT}/${RUN_DIR}/logs/test_run.log"

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MAIN_LOG}"
}

# Install package in editable mode for testing
log "Checking active_inference package installation..."

# Check if package is already importable
if python -c "import active_inference" 2>/dev/null; then
    log "✅ Package already installed and importable"
else
    log "Installing active_inference package in editable mode..."
    # Use uv pip for fast package installation
    uv pip install -e . --quiet 2>&1 | tee -a "${MAIN_LOG}"
    INSTALL_EXIT_CODE=$?
    if [ ${INSTALL_EXIT_CODE} -ne 0 ]; then
        log "❌ Package installation failed with exit code: ${INSTALL_EXIT_CODE}"
        exit ${INSTALL_EXIT_CODE}
    fi
    log "✅ Package installed successfully"
fi
log ""

log "Created output directory: ${RUN_DIR}"
log "Starting test suite..."

# 1. Run active_inference tests
log ""
log "=== Running Active Inference Tests ==="
set +e  # Temporarily disable exit on error to capture exit code
pytest tests/ -v \
    --cov=active_inference \
    --cov-report=html:"${PROJECT_ROOT}/${RUN_DIR}/coverage/active_inference" \
    --cov-report=term \
    --junitxml="${PROJECT_ROOT}/${RUN_DIR}/reports/active_inference_junit.xml" \
    2>&1 | tee "${PROJECT_ROOT}/${RUN_DIR}/logs/active_inference_tests.log"

AI_EXIT_CODE=${PIPESTATUS[0]}  # Get pytest exit code, not tee exit code
set -e  # Re-enable exit on error
log "Active Inference tests exit code: ${AI_EXIT_CODE}"

# 2. Run THRML integration tests
log ""
log "=== Running THRML Integration Tests ==="
set +e  # Temporarily disable exit on error to capture exit code
pytest tests/test_thrml_integration.py -v \
    --junitxml="${PROJECT_ROOT}/${RUN_DIR}/reports/thrml_integration_junit.xml" \
    2>&1 | tee "${PROJECT_ROOT}/${RUN_DIR}/logs/thrml_integration_tests.log"

THRML_EXIT_CODE=${PIPESTATUS[0]}  # Get pytest exit code, not tee exit code
set -e  # Re-enable exit on error
log "THRML integration tests exit code: ${THRML_EXIT_CODE}"

# 3. Run parent THRML tests (if they exist and we want to validate)
log ""
log "=== Checking Parent THRML Tests ==="
if [ -d "../tests" ] && [ -f "../tests/test_readme.py" ]; then
    log "Parent THRML tests found, running subset..."

    # Run from parent directory using subshell to avoid cd issues
    set +e  # Temporarily disable exit on error
    (cd .. && pytest tests/test_readme.py -v \
        --junitxml="${PROJECT_ROOT}/${RUN_DIR}/reports/parent_thrml_junit.xml" \
        2>&1 | tee "${PROJECT_ROOT}/${RUN_DIR}/logs/parent_thrml_tests.log")

    PARENT_EXIT_CODE=$?
    set -e  # Re-enable exit on error
    log "Parent THRML tests exit code: ${PARENT_EXIT_CODE}"
else
    log "Parent THRML tests not found or skipped"
    PARENT_EXIT_CODE=0
fi

# 4. Generate summary report
log ""
log "=== Generating Summary Report ==="

SUMMARY_FILE="${PROJECT_ROOT}/${RUN_DIR}/TEST_SUMMARY.md"

cat > "${SUMMARY_FILE}" << EOF
# Test Suite Summary Report

**Run Date:** $(date)
**Run ID:** ${TIMESTAMP}

## Test Results

### Active Inference Tests
- Exit Code: ${AI_EXIT_CODE}
- Status: $([ ${AI_EXIT_CODE} -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")
- Log: logs/active_inference_tests.log
- Coverage: coverage/active_inference/index.html
- JUnit: reports/active_inference_junit.xml

### THRML Integration Tests
- Exit Code: ${THRML_EXIT_CODE}
- Status: $([ ${THRML_EXIT_CODE} -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")
- Log: logs/thrml_integration_tests.log
- JUnit: reports/thrml_integration_junit.xml

### Parent THRML Validation
- Exit Code: ${PARENT_EXIT_CODE}
- Status: $([ ${PARENT_EXIT_CODE} -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")
- Log: logs/parent_thrml_tests.log
- JUnit: reports/parent_thrml_junit.xml

## Overall Status

$([ ${AI_EXIT_CODE} -eq 0 ] && [ ${THRML_EXIT_CODE} -eq 0 ] && [ ${PARENT_EXIT_CODE} -eq 0 ] && echo "✅ **ALL TESTS PASSED**" || echo "❌ **SOME TESTS FAILED**")

## Files Generated

- Main log: logs/test_run.log
- Coverage report: coverage/active_inference/index.html
- JUnit XML reports: reports/*.xml

## Next Steps

$([ ${AI_EXIT_CODE} -eq 0 ] && [ ${THRML_EXIT_CODE} -eq 0 ] && [ ${PARENT_EXIT_CODE} -eq 0 ] && echo "All tests passed! The active_inference package is ready for use." || echo "Review failed tests in the log files above.")

---
Generated by: scripts/run_tests.sh
EOF

# Display summary
log ""
log "=== Test Summary ==="
cat "${SUMMARY_FILE}" | tee -a "${MAIN_LOG}"

# 5. Create a symlink to latest results
rm -f "${PROJECT_ROOT}/${TEST_OUTPUT_DIR}/latest"
ln -s "run_${TIMESTAMP}" "${PROJECT_ROOT}/${TEST_OUTPUT_DIR}/latest"

log ""
log "=== Test suite complete ==="
log "Results saved to: ${RUN_DIR}"
log "View summary: ${SUMMARY_FILE}"
log "View coverage: ${RUN_DIR}/coverage/active_inference/index.html"
log ""
log "Quick access via: ${TEST_OUTPUT_DIR}/latest/"

# Exit with appropriate code
if [ ${AI_EXIT_CODE} -eq 0 ] && [ ${THRML_EXIT_CODE} -eq 0 ] && [ ${PARENT_EXIT_CODE} -eq 0 ]; then
    log "✅ All tests passed!"
    exit 0
else
    log "❌ Some tests failed. Check logs for details."
    exit 1
fi
