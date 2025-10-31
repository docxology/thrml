# Development Scripts

## Overview

The `scripts/` directory contains development automation scripts for setup, testing, validation, and code quality checks. These scripts streamline the development workflow and ensure code quality.

## Scripts

### `setup.sh`
**Purpose**: Initial package setup and environment configuration.

**Usage**:
```bash
./scripts/setup.sh
```

**Actions**:
- Creates virtual environment (if needed)
- Installs package in development mode
- Installs all dependencies (including development dependencies)
- Runs initial tests to verify setup

**Requirements**: `uv` or `pip`

---

### `setup_and_test.sh`
**Purpose**: Combined setup and test execution.

**Usage**:
```bash
./scripts/setup_and_test.sh
```

**Actions**:
- Runs `setup.sh`
- Executes full test suite
- Generates coverage report

**Use Case**: CI/CD pipelines, fresh checkouts

---

### `run_tests.sh`
**Purpose**: Execute test suite with coverage.

**Usage**:
```bash
./scripts/run_tests.sh
```

**Actions**:
- Runs pytest with coverage tracking
- Generates HTML coverage report in `htmlcov/`
- Displays coverage summary
- Fails if coverage < 90% (configurable)

**Options**:
```bash
./scripts/run_tests.sh -v              # Verbose output
./scripts/run_tests.sh -k test_core    # Run specific tests
./scripts/run_tests.sh --no-cov        # Skip coverage
```

---

### `test_all_examples.sh`
**Purpose**: Test all example scripts for errors.

**Usage**:
```bash
./scripts/test_all_examples.sh
```

**Actions**:
- Runs each example script (01-13)
- Captures output and errors
- Reports success/failure for each
- Saves results to `scripts/test_results/`

**Validation**:
- Checks for Python errors
- Verifies output generation
- Confirms plots are created

---

### `run_all_examples.sh`
**Purpose**: Execute all example scripts with output.

**Usage**:
```bash
./scripts/run_all_examples.sh [--clean]
```

**Actions**:
- Runs all examples in order (01-13)
- Saves outputs to `output/<example>/`
- Generates summary report
- Optionally cleans previous outputs

**Options**:
- `--clean`: Remove previous output directories before running

**Output**: Timestamped directories with plots, data, logs, metrics.

---

### `validate_examples.py`
**Purpose**: Validate example scripts and outputs.

**Usage**:
```bash
python3 scripts/validate_examples.py
```

**Validation Checks**:
- ✅ Example file exists and is readable
- ✅ Example imports are valid
- ✅ Example runs without errors
- ✅ Expected outputs are generated
- ✅ Plots are created
- ✅ Data is saved
- ✅ Logs are written

**Output**: Detailed validation report with pass/fail for each example.

---

### `check.sh`
**Purpose**: Run all code quality checks.

**Usage**:
```bash
./scripts/check.sh
```

**Checks**:
1. **Linting**: `ruff check src/ tests/`
2. **Type Checking**: `pyright src/`
3. **Tests**: `pytest tests/ -v`
4. **Coverage**: Coverage report and threshold check
5. **Examples**: Validate all examples run

**Exit Code**: 0 if all checks pass, non-zero otherwise.

**Use Case**: Pre-commit checks, CI/CD validation.

---

### `format.sh`
**Purpose**: Format code according to style guidelines.

**Usage**:
```bash
./scripts/format.sh
```

**Actions**:
- **Black**: Format Python code
- **isort**: Sort and organize imports
- **Ruff**: Auto-fix linting issues (safe fixes only)

**Options**:
```bash
./scripts/format.sh --check    # Check only, don't modify
```

**Use Case**: Pre-commit hook, code cleanup.

---

### `resource_tracker.py`
**Purpose**: Monitor resource usage during example execution.

**Usage**:
```python
from scripts.resource_tracker import track_resources

with track_resources("my_experiment"):
    # Your code here
    run_experiment()

# Saves resource metrics to output/
```

**Monitors**:
- CPU usage (%)
- Memory usage (MB)
- GPU memory (if available)
- Execution time
- Peak memory

**Output**: JSON file with resource metrics, plots of usage over time.

---

## Workflow Examples

### Development Workflow

```bash
# 1. Initial setup
./scripts/setup.sh

# 2. Make changes to code
# ... edit files ...

# 3. Format code
./scripts/format.sh

# 4. Run checks
./scripts/check.sh

# 5. Run tests
./scripts/run_tests.sh

# 6. Test examples
./scripts/test_all_examples.sh
```

### CI/CD Workflow

```bash
# Full validation pipeline
./scripts/setup_and_test.sh && \
./scripts/check.sh && \
./scripts/test_all_examples.sh
```

### Example Development

```bash
# Run specific example
python3 examples/06_coin_flip_inference.py

# Validate example
python3 scripts/validate_examples.py

# Test all examples
./scripts/test_all_examples.sh

# Run all examples with fresh output
./scripts/run_all_examples.sh --clean
```

## Configuration

### Test Configuration

Tests are configured in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --cov=active_inference --cov-report=html"
```

### Coverage Requirements

Minimum coverage: 90% (configured in `pyproject.toml`)

### Linting Configuration

Ruff configuration in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 100
target-version = "py310"
```

## Output Directories

Scripts create/use the following directories:

- `htmlcov/`: HTML coverage reports
- `output/`: Example outputs
- `scripts/test_results/`: Example test results
- `.pytest_cache/`: Pytest cache
- `.ruff_cache/`: Ruff cache

## Dependencies

Scripts require:

**Core**:
- `bash` (for shell scripts)
- `python3 >= 3.10`
- `pytest`
- `pytest-cov`

**Optional**:
- `ruff` (linting)
- `black` (formatting)
- `isort` (import sorting)
- `pyright` (type checking)

Install development dependencies:
```bash
uv pip install -e ".[development]"
```

## Continuous Integration

These scripts are designed for CI/CD integration:

**GitHub Actions Example**:
```yaml
- name: Setup and Test
  run: ./scripts/setup_and_test.sh

- name: Code Quality
  run: ./scripts/check.sh

- name: Validate Examples
  run: ./scripts/test_all_examples.sh
```

## Maintenance

### Adding New Scripts

When adding new scripts:

1. Make executable: `chmod +x scripts/new_script.sh`
2. Add shebang: `#!/bin/bash`
3. Add usage documentation to this README
4. Test in clean environment
5. Add to CI/CD pipeline if appropriate

### Updating Scripts

When updating scripts:

1. Test changes locally
2. Update this README
3. Verify CI/CD compatibility
4. Update version in comments if appropriate

## Troubleshooting

### Script Permission Errors

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf .venv
./scripts/setup.sh
```

### Test Failures

```bash
# Run tests verbosely
./scripts/run_tests.sh -v

# Run specific test
pytest tests/test_core.py::test_specific -v
```

### Example Failures

```bash
# Check example validation
python3 scripts/validate_examples.py

# Run example with full output
python3 examples/06_coin_flip_inference.py

# Check example logs
cat output/06_coin_flip_inference/latest/logs/execution.log
```

## Best Practices

1. **Run checks before commit**: `./scripts/check.sh`
2. **Format code regularly**: `./scripts/format.sh`
3. **Validate examples after changes**: `./scripts/test_all_examples.sh`
4. **Monitor resource usage**: Use `resource_tracker.py` for expensive operations
5. **Keep scripts updated**: Update scripts when adding new features

## References

- **Testing**: [pytest documentation](https://docs.pytest.org/)
- **Coverage**: [coverage.py documentation](https://coverage.readthedocs.io/)
- **Linting**: [Ruff documentation](https://docs.astral.sh/ruff/)
- **Formatting**: [Black documentation](https://black.readthedocs.io/)

## Related Documentation

- **Development**: [CONTRIBUTING.md](../CONTRIBUTING.md) (if exists)
- **Testing**: [tests/README.md](../tests/README.md)
- **Examples**: [examples/README.md](../examples/README.md)
