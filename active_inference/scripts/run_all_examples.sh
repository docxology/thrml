#!/usr/bin/env bash
#
# Run all example scripts with logging and error handling
#
# Usage: ./scripts/run_all_examples.sh [--clean]
#
# Options:
#   --clean    Clean output directory before running

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="${PROJECT_ROOT}/examples"
OUTPUT_DIR="${PROJECT_ROOT}/output"
LOG_FILE="${OUTPUT_DIR}/run_all_examples.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to print section header
print_section() {
    echo ""
    print_status "${BLUE}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_status "${BLUE}" "$1"
    print_status "${BLUE}" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Check if clean option is specified
CLEAN_OUTPUT=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN_OUTPUT=true
fi

# Clean output directory if requested
if [[ "$CLEAN_OUTPUT" == "true" ]]; then
    print_status "${YELLOW}" "Cleaning output directory..."
    if [[ -d "$OUTPUT_DIR" ]]; then
        # Keep the directory structure but remove timestamped runs
        find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d -exec rm -rf {} + 2>/dev/null || true
        print_status "${GREEN}" "Output directory cleaned"
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "=== Active Inference Examples Run ===" > "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "Project: $PROJECT_ROOT" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

print_section "Running Active Inference Examples"

# Print system information
if command -v python3 &> /dev/null; then
    python3 "${SCRIPT_DIR}/resource_tracker.py" banner 2>/dev/null || true
    python3 "${SCRIPT_DIR}/resource_tracker.py" estimate-banner 13 2>/dev/null || true
fi

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [[ ! -d ".venv" ]]; then
    print_status "${RED}" "Error: Virtual environment not found at .venv"
    print_status "${YELLOW}" "Please run ./scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
print_status "${YELLOW}" "Activating virtual environment..."
source .venv/bin/activate

# Check if active_inference package is installed
if ! python3 -c "import active_inference" 2>/dev/null; then
    print_status "${RED}" "Error: active_inference package not installed"
    print_status "${YELLOW}" "Please run: uv pip install -e ."
    exit 1
fi

# List of examples to run (16 examples total)
EXAMPLES=(
    "00_probabilistic_computing.py"
    "01_all_of_thrml.py"
    "02_spin_models.py"
    "03_precision_control.py"
    "04_mdp_example.py"
    "05_pomdp_example.py"
    "06_coin_flip_inference.py"
    "07_matrix_performance.py"
    "08_signal_processing.py"
    "09_control_theory.py"
    "10_active_inference_fundamentals.py"
    "11_thrml_comprehensive.py"
    "12_statistical_validation_demo.py"
    "13_coin_flip_meta_analysis.py"
    "14_basic_inference.py"
    "15_grid_world_agent.py"
)

# Track results
TOTAL_EXAMPLES=${#EXAMPLES[@]}
SUCCESSFUL=0
FAILED=0
FAILED_EXAMPLES=()
START_TIME=$(date +%s)

# Run each example
for example in "${EXAMPLES[@]}"; do
    example_path="${EXAMPLES_DIR}/${example}"
    example_name=$(basename "$example" .py)

    print_section "Running: $example_name"

    if [[ ! -f "$example_path" ]]; then
        print_status "${RED}" "✗ Example not found: $example"
        ((FAILED++))
        FAILED_EXAMPLES+=("$example")
        echo "FAILED: $example (not found)" >> "$LOG_FILE"
        continue
    fi

    print_status "${YELLOW}" "Executing: python3 $example_path"
    echo "" >> "$LOG_FILE"
    echo "=== $example ===" >> "$LOG_FILE"

    # Time the example execution
    example_start=$(date +%s)

    # Run the example
    if python3 "$example_path" >> "$LOG_FILE" 2>&1; then
        example_end=$(date +%s)
        example_duration=$((example_end - example_start))

        print_status "${GREEN}" "✓ Success: $example_name (${example_duration}s)"
        ((SUCCESSFUL++))
        echo "SUCCESS: $example (${example_duration}s)" >> "$LOG_FILE"

        # Show output summary
        example_output="${OUTPUT_DIR}/${example_name}"
        if [[ -d "$example_output" ]]; then
            # Check if this is a timestamped structure or simple structure
            latest_run=$(ls -t "$example_output" | head -1)
            if [[ -n "$latest_run" && -d "${example_output}/${latest_run}" ]]; then
                run_dir="${example_output}/${latest_run}"
            else
                # Simple structure (e.g., examples 00-02)
                run_dir="$example_output"
            fi

            # Count files by type (handle both structures)
            n_configs=$(find "$run_dir" -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -path "*/config/*" -type f 2>/dev/null | wc -l)
            n_data=$(find "$run_dir" -path "*/data/*" -type f 2>/dev/null | wc -l)
            n_plots=$(find "$run_dir" -name "*.png" -o -name "*.pdf" -o -name "*.svg" -o -path "*/plots/*" -type f 2>/dev/null | wc -l)
            n_logs=$(find "$run_dir" -path "*/logs/*" -o -name "*.log" -type f 2>/dev/null | wc -l)

            # Calculate total size
            total_size=$(du -sh "$run_dir" 2>/dev/null | cut -f1)

            print_status "${BLUE}" "  Duration: ${example_duration}s"
            if [[ "$n_configs" -gt 0 || "$n_data" -gt 0 || "$n_plots" -gt 0 || "$n_logs" -gt 0 ]]; then
                print_status "${BLUE}" "  Output:   $n_configs configs, $n_data data files, $n_plots plots, $n_logs logs (${total_size})"
            else
                # For simple examples with just image files
                n_files=$(find "$run_dir" -type f 2>/dev/null | wc -l)
                print_status "${BLUE}" "  Output:   $n_files files (${total_size})"
            fi
            print_status "${BLUE}" "  Location: $run_dir"
        fi
    else
        example_end=$(date +%s)
        example_duration=$((example_end - example_start))
        print_status "${RED}" "✗ Failed: $example_name (${example_duration}s)"
        ((FAILED++))
        FAILED_EXAMPLES+=("$example")
        echo "FAILED: $example" >> "$LOG_FILE"
    fi
done

# Print summary
print_section "Execution Summary"

# Calculate total execution time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
DURATION_MINUTES=$((TOTAL_DURATION / 60))
DURATION_SECONDS=$((TOTAL_DURATION % 60))

echo ""
print_status "${BLUE}" "Total examples:    $TOTAL_EXAMPLES"
print_status "${GREEN}" "Successful:        $SUCCESSFUL"
print_status "${RED}" "Failed:            $FAILED"
print_status "${YELLOW}" "Total time:        ${DURATION_MINUTES}m ${DURATION_SECONDS}s"
echo ""

# Calculate overall statistics
if [[ $SUCCESSFUL -gt 0 ]]; then
    total_configs=0
    total_data=0
    total_plots=0
    total_logs=0

    for example in "${EXAMPLES[@]}"; do
        example_name=$(basename "$example" .py)
        example_output="${OUTPUT_DIR}/${example_name}"

        if [[ -d "$example_output" ]]; then
            # Check if this is a timestamped structure or simple structure
            latest_run=$(ls -t "$example_output" | head -1)
            if [[ -n "$latest_run" && -d "${example_output}/${latest_run}" ]]; then
                run_dir="${example_output}/${latest_run}"
            else
                # Simple structure (e.g., examples 00-02)
                run_dir="$example_output"
            fi

            total_configs=$((total_configs + $(find "$run_dir" -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -path "*/config/*" -type f 2>/dev/null | wc -l)))
            total_data=$((total_data + $(find "$run_dir" -path "*/data/*" -type f 2>/dev/null | wc -l)))
            total_plots=$((total_plots + $(find "$run_dir" -name "*.png" -o -name "*.pdf" -o -name "*.svg" -o -path "*/plots/*" -type f 2>/dev/null | wc -l)))
            total_logs=$((total_logs + $(find "$run_dir" -path "*/logs/*" -o -name "*.log" -type f 2>/dev/null | wc -l)))
        fi
    done

    print_status "${BLUE}" "Total outputs:"
    print_status "${BLUE}" "  • Configuration files: $total_configs"
    print_status "${BLUE}" "  • Data files: $total_data"
    print_status "${BLUE}" "  • Plots: $total_plots"
    print_status "${BLUE}" "  • Logs: $total_logs"
    echo ""

    # Calculate total disk usage
    total_disk=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    print_status "${BLUE}" "  • Total disk usage: $total_disk"
    echo ""
fi

if [[ $FAILED -gt 0 ]]; then
    print_status "${RED}" "Failed examples:"
    for failed_example in "${FAILED_EXAMPLES[@]}"; do
        print_status "${RED}" "  - $failed_example"
    done
    echo ""
fi

# Print output location
print_status "${BLUE}" "Results saved to: $OUTPUT_DIR"
print_status "${BLUE}" "Log file: $LOG_FILE"
echo ""

# Generate summary report
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
{
    echo "═══════════════════════════════════════════════════════════"
    echo "   Active Inference Examples - Execution Summary"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Total: $TOTAL_EXAMPLES | Success: $SUCCESSFUL | Failed: $FAILED"
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "EXAMPLES EXECUTED:"
    echo "───────────────────────────────────────────────────────────"

    for example in "${EXAMPLES[@]}"; do
        example_name=$(basename "$example" .py)
        example_output="${OUTPUT_DIR}/${example_name}"

        if [[ -d "$example_output" ]]; then
            latest_run=$(ls -t "$example_output" | head -1)
            if [[ -n "$latest_run" ]]; then
                run_dir="${example_output}/${latest_run}"
                summary_json="${run_dir}/summary.json"
                manifest_json="${run_dir}/manifest.json"

                echo ""
                echo "✓ $example_name"
                echo "  Run: $latest_run"

                if [[ -f "$summary_json" ]]; then
                    # Extract key metrics if jq is available
                    if command -v jq &> /dev/null; then
                        duration=$(jq -r '.duration_seconds' "$summary_json" 2>/dev/null || echo "N/A")
                        total_files=$(jq -r '.output_summary.total_files' "$summary_json" 2>/dev/null || echo "N/A")
                        total_size=$(jq -r '.output_summary.total_size_formatted' "$summary_json" 2>/dev/null || echo "N/A")

                        echo "  Duration: ${duration}s"
                        echo "  Outputs: $total_files files ($total_size)"
                    else
                        echo "  Summary: available (install jq for details)"
                    fi
                fi

                # Count file types
                n_configs=$(find "$run_dir/config" -type f 2>/dev/null | wc -l | tr -d ' ')
                n_data=$(find "$run_dir/data" -type f 2>/dev/null | wc -l | tr -d ' ')
                n_plots=$(find "$run_dir/plots" -type f 2>/dev/null | wc -l | tr -d ' ')

                echo "  Files: $n_configs configs, $n_data data, $n_plots plots"
                echo "  Path: $run_dir"
            else
                echo ""
                echo "? $example_name (no runs found)"
            fi
        else
            echo ""
            echo "✗ $example_name (FAILED)"
        fi
    done

    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "GLOBAL STATISTICS:"
    echo "───────────────────────────────────────────────────────────"
    echo ""

    # Calculate totals if successful examples exist
    if [[ $SUCCESSFUL -gt 0 ]]; then
        total_configs=0
        total_data=0
        total_plots=0
        total_duration=0.0

        for example in "${EXAMPLES[@]}"; do
            example_name=$(basename "$example" .py)
            example_output="${OUTPUT_DIR}/${example_name}"

            if [[ -d "$example_output" ]]; then
                # Check if this is a timestamped structure or simple structure
                latest_run=$(ls -t "$example_output" | head -1)
                if [[ -n "$latest_run" && -d "${example_output}/${latest_run}" ]]; then
                    run_dir="${example_output}/${latest_run}"
                else
                    # Simple structure (e.g., examples 00-02)
                    run_dir="$example_output"
                fi

                total_configs=$((total_configs + $(find "$run_dir" -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -path "*/config/*" -type f 2>/dev/null | wc -l)))
                total_data=$((total_data + $(find "$run_dir" -path "*/data/*" -type f 2>/dev/null | wc -l)))
                total_plots=$((total_plots + $(find "$run_dir" -name "*.png" -o -name "*.pdf" -o -name "*.svg" -o -path "*/plots/*" -type f 2>/dev/null | wc -l)))

                # Get duration if jq is available
                if command -v jq &> /dev/null; then
                    summary_json="${run_dir}/summary.json"
                    if [[ -f "$summary_json" ]]; then
                        duration=$(jq -r '.duration_seconds' "$summary_json" 2>/dev/null)
                        if [[ "$duration" != "null" && "$duration" != "N/A" ]]; then
                            total_duration=$(echo "$total_duration + $duration" | bc 2>/dev/null || echo "$total_duration")
                        fi
                    fi
                fi
            fi
        done

        echo "Total Configuration Files: $total_configs"
        echo "Total Data Files: $total_data"
        echo "Total Plot Files: $total_plots"
        echo "Total Files Generated: $((total_configs + total_data + total_plots))"

        if command -v jq &> /dev/null && [[ $(echo "$total_duration > 0" | bc) -eq 1 ]]; then
            echo "Total Execution Time: ${total_duration}s"
        fi

        # Calculate disk usage
        total_disk=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
        echo "Total Disk Usage: $total_disk"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "Output directory: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo "═══════════════════════════════════════════════════════════"
} > "$SUMMARY_FILE"

print_status "${GREEN}" "Summary report: $SUMMARY_FILE"
echo ""

# Exit with appropriate code
if [[ $FAILED -gt 0 ]]; then
    print_status "${RED}" "Some examples failed. Check logs for details."
    exit 1
else
    print_status "${GREEN}" "All examples completed successfully! ✓"
    exit 0
fi
