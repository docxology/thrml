# UV-Only Setup - Quick Guide

The active_inference project now uses **UV exclusively** for all package management.

## One-Line Setup

```bash
cd active_inference && ./scripts/setup.sh
```

That's it! The setup script:
- ✅ Installs UV automatically if not present
- ✅ Creates virtual environment with `uv venv`
- ✅ Installs THRML from parent directory (if available) or PyPI
- ✅ Installs all dependencies with `uv pip`
- ✅ Sets up pre-commit hooks (if available)

## Manual Setup

If you prefer manual setup:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
cd active_inference
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package
uv pip install -e ".[all]"
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[development]"

# Run tests
./scripts/run_tests.sh

# Run examples
./scripts/run_all_examples.sh

# Format code
./scripts/format.sh

# Run all checks
./scripts/check.sh
```

## Why UV Only?

- ⚡ **10-100x faster** than pip
- 🔒 **Better dependency resolution**
- 🎯 **Single, simple path**
- 🚀 **Modern tooling**
- 🤝 **Consistent with THRML**

## Requirements

- Python 3.10+
- UV (installed by setup.sh)

## Complete Documentation

See [`UV_ONLY_MIGRATION.md`](UV_ONLY_MIGRATION.md) for full migration details.
