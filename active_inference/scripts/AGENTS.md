# Development Scripts: Automation and Utilities

## Script Functions

### `setup.sh`
**Purpose**: Initialize development environment.

**Actions**:
1. Check if `uv` is available, install if needed
2. Create virtual environment (`.venv/`) with `uv venv`
3. Install package in editable mode (`uv pip install -e .`)
4. Install all dependencies (core + development + testing)
5. Set up pre-commit hooks (if available)

**Usage**:
```bash
./scripts/setup.sh
```

**Requirements**:
- Python 3.10+
- `uv` (installed automatically if not present)

---

### `setup_and_test.sh`
**Purpose**: Combined setup and validation.

**Actions**:
1. Run `setup.sh` to configure environment
2. Execute full test suite
3. Generate coverage report
4. Display results

**Usage**:
```bash
./scripts/setup_and_test.sh
```

**Exit Code**: 0 if all tests pass, non-zero otherwise.

**Use Case**: CI/CD initialization, fresh environment validation.

---

### `run_tests.sh`
**Purpose**: Execute test suite with coverage tracking.

**Algorithm**:
1. Run pytest on `tests/` directory
2. Track code coverage
3. Generate HTML report in `htmlcov/`
4. Check coverage threshold (90%)

**Usage**:
```bash
./scripts/run_tests.sh              # All tests
./scripts/run_tests.sh -v           # Verbose
./scripts/run_tests.sh -k test_core # Specific pattern
./scripts/run_tests.sh --no-cov     # Without coverage
```

**Output**:
- Test results to console
- Coverage report in `htmlcov/index.html`
- Coverage summary

**Threshold**: Fails if coverage < 90%

---

### `test_all_examples.sh`
**Purpose**: Validate all example scripts.

**Algorithm**:
1. Iterate through examples (01-13)
2. Run each example with Python
3. Capture stdout/stderr
4. Check for errors
5. Verify output generation

**Usage**:
```bash
./scripts/test_all_examples.sh
```

**Checks**:
- ✅ Script runs without Python errors
- ✅ Imports succeed
- ✅ Output directory created
- ✅ Plots generated
- ✅ Data files saved

**Output**: Summary report with pass/fail for each example.

**Results Saved To**: `scripts/test_results/`

---

### `run_all_examples.sh`
**Purpose**: Execute all examples with full output generation.

**Algorithm**:
1. Optionally clean previous outputs (with `--clean`)
2. Run each example in sequence
3. Save outputs to timestamped directories
4. Generate overall summary

**Usage**:
```bash
./scripts/run_all_examples.sh         # Keep existing outputs
./scripts/run_all_examples.sh --clean # Clean first
```

**Output Structure**:
```
output/
├── 01_basic_inference/
│   └── 20251030_123456/
│       ├── config/
│       ├── data/
│       ├── logs/
│       ├── plots/
│       └── summary.json
├── 02_grid_world_agent/
│   └── ...
└── summary.txt  # Overall summary
```

**Use Case**: Generate complete output suite for review, documentation, or validation.

---

### `validate_examples.py`
**Purpose**: Comprehensive example validation.

**Algorithm**:
```python
for example in examples:
    # 1. Check file exists
    validate_file_exists(example)

    # 2. Check imports
    validate_imports(example)

    # 3. Run example
    result = run_example(example)

    # 4. Validate outputs
    validate_plots_created(example)
    validate_data_saved(example)
    validate_logs_written(example)

    # 5. Check for errors
    validate_no_errors(result)
```

**Usage**:
```bash
python3 scripts/validate_examples.py
python3 scripts/validate_examples.py --example 06  # Specific example
```

**Validation Checks**:
- File structure
- Import validity
- Execution success
- Output generation
- Error absence
- Config saved
- Metrics recorded

**Output**: Detailed validation report (text + JSON)

---

### `check.sh`
**Purpose**: Run all code quality checks.

**Checks**:
1. **Linting**: `ruff check src/ tests/` - Style and quality issues
2. **Type Checking**: `pyright src/` - Static type validation
3. **Tests**: `pytest tests/ -v` - Test suite execution
4. **Coverage**: Coverage threshold check (90%)
5. **Examples**: Basic example validation

**Usage**:
```bash
./scripts/check.sh
```

**Exit Code**: 0 if all checks pass, non-zero on any failure.

**Use Case**: Pre-commit validation, CI/CD quality gates.

---

### `format.sh`
**Purpose**: Auto-format code to style standards.

**Formatters**:
1. **Black**: Python code formatting (line length 100)
2. **isort**: Import sorting and organization
3. **Ruff**: Auto-fix safe linting issues

**Usage**:
```bash
./scripts/format.sh              # Format in place
./scripts/format.sh --check      # Check only (no modifications)
```

**Configuration**: See `pyproject.toml` for formatting rules.

**Use Case**: Pre-commit hook, code cleanup after development.

---

### `resource_tracker.py`
**Purpose**: Monitor resource usage during execution.

**Monitors**:
- CPU usage (%)
- Memory usage (MB)
- GPU memory (MB, if available)
- Execution time (seconds)
- Peak memory usage

**Usage**:
```python
from scripts.resource_tracker import ResourceTracker

tracker = ResourceTracker()
tracker.start()

# Your code here
run_experiment()

metrics = tracker.stop()
tracker.save_report("output/resources.json")
```

**Context Manager**:
```python
with track_resources("experiment_name"):
    run_experiment()
# Automatically saves metrics
```

**Output**:
- JSON file with metrics
- Plots of usage over time
- Summary statistics

**Use Case**: Performance profiling, resource optimization, benchmarking.

---

## Usage Patterns

### Development Workflow

```bash
# 1. Setup (first time)
./scripts/setup.sh

# 2. Work on code
# ... edit files ...

# 3. Format
./scripts/format.sh

# 4. Check quality
./scripts/check.sh

# 5. Run tests
./scripts/run_tests.sh

# 6. Test examples
./scripts/test_all_examples.sh
```

### Pre-Commit Checks

```bash
# Quick validation before commit
./scripts/format.sh && ./scripts/check.sh
```

### CI/CD Pipeline

```bash
# Full validation pipeline
./scripts/setup_and_test.sh
./scripts/check.sh
./scripts/test_all_examples.sh
```

### Example Development

```bash
# 1. Create new example
touch examples/14_new_example.py

# 2. Develop example
# ... write code ...

# 3. Test specific example
python3 examples/14_new_example.py

# 4. Validate
python3 scripts/validate_examples.py --example 14

# 5. Test all examples
./scripts/test_all_examples.sh
```

### Batch Example Execution

```bash
# Run all examples with fresh output
./scripts/run_all_examples.sh --clean

# Generate documentation screenshots
for ex in examples/*.py; do
    python3 "$ex"
done
```

## Script Dependencies

### Required

- `bash` (shell scripts)
- `python3 >= 3.10`
- `pytest >= 7.2.0`
- `pytest-cov >= 4.0.0`

### Optional (Development)

- `ruff` - Linting
- `black` - Code formatting
- `isort` - Import sorting
- `pyright` - Type checking
- `psutil` - Resource monitoring

Install all:
```bash
uv pip install -e ".[development]"
```

## Configuration Files

### `pyproject.toml`

Scripts use configuration from `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=active_inference"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 100
```

### `.pre-commit-config.yaml`

Pre-commit hooks (if using):
```yaml
repos:
  - repo: local
    hooks:
      - id: format
        name: Format code
        entry: ./scripts/format.sh
        language: system
        pass_filenames: false

      - id: check
        name: Quality checks
        entry: ./scripts/check.sh
        language: system
        pass_filenames: false
```

## Output Structure

Scripts create/use:

- **`htmlcov/`**: HTML coverage reports
- **`output/`**: Example outputs
- **`scripts/test_results/`**: Example test results
- **`.pytest_cache/`**: Pytest cache
- **`.ruff_cache/`**: Ruff linter cache
- **`.coverage`**: Coverage data file

## Best Practices

### 1. Run Checks Before Committing

```bash
./scripts/format.sh
./scripts/check.sh
git add .
git commit -m "Feature: ..."
```

### 2. Validate Examples After Changes

```bash
# Quick check
./scripts/test_all_examples.sh

# Full validation
python3 scripts/validate_examples.py
```

### 3. Monitor Resource Usage

```python
# For expensive operations
from scripts.resource_tracker import track_resources

with track_resources("my_experiment"):
    run_expensive_computation()
```

### 4. Keep Environment Updated

```bash
# Recreate environment after dependency changes
rm -rf .venv
./scripts/setup.sh
```

## Troubleshooting

### Permission Errors

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Virtual Environment Issues

```bash
# Clean and rebuild
rm -rf .venv .pytest_cache .ruff_cache
./scripts/setup.sh
```

### Test Failures

```bash
# Verbose test run
./scripts/run_tests.sh -v

# Specific test
pytest tests/test_core.py::test_specific -v

# With output
pytest -v -s
```

### Example Failures

```bash
# Run example directly
python3 examples/06_coin_flip_inference.py

# Check logs
cat output/06_coin_flip_inference/latest/logs/execution.log

# Validate specific example
python3 scripts/validate_examples.py --example 06
```

### Coverage Issues

```bash
# Generate detailed coverage report
pytest tests/ --cov=active_inference --cov-report=term-missing

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Integration Points

### CI/CD

GitHub Actions example:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup and Test
        run: ./scripts/setup_and_test.sh
      - name: Quality Checks
        run: ./scripts/check.sh
      - name: Validate Examples
        run: ./scripts/test_all_examples.sh
```

### Pre-commit Hooks

```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Future Enhancements

1. **Parallel Testing**: Run examples in parallel
2. **Benchmark Suite**: Performance benchmarking
3. **Documentation Generation**: Auto-generate API docs
4. **Coverage Reporting**: Upload to codecov/coveralls
5. **Performance Regression**: Track performance over time

## Related Documentation

- **Development**: [CONTRIBUTING.md](../CONTRIBUTING.md) (if exists)
- **Testing**: [tests/README.md](../tests/README.md)
- **Examples**: [examples/README.md](../examples/README.md)
- **Main README**: [README.md](../README.md)
