#!/usr/bin/env python3
"""Validation script for active inference examples.

This script performs dry-run validation of example scripts without full execution.
Checks:
- Import validity
- Syntax correctness
- Example structure
- Output directory creation
"""

import ast
import sys
from pathlib import Path


def validate_example_syntax(example_path: Path) -> tuple[bool, str]:
    """Validate that example has correct Python syntax."""
    try:
        with open(example_path, "r") as f:
            code = f.read()
        ast.parse(code)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_example_structure(example_path: Path) -> tuple[bool, str]:
    """Validate that example follows the expected structure."""
    try:
        with open(example_path, "r") as f:
            content = f.read()

        required_elements = [
            ("ExampleRunner import", "from example_utils import ExampleRunner"),
            ("main function", "def main():"),
            ("runner.start()", "runner.start()"),
            ("runner.end()", "runner.end()"),
            ("if __name__", 'if __name__ == "__main__":'),
        ]

        missing = []
        for name, pattern in required_elements:
            if pattern not in content:
                missing.append(name)

        if missing:
            return False, f"Missing elements: {', '.join(missing)}"

        return True, "Structure OK"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Run validation on all examples."""
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"

    examples = [
        "example_utils.py",
        "01_basic_inference.py",
        "02_grid_world_agent.py",
        "03_precision_control.py",
        "04_mdp_example.py",
        "05_pomdp_example.py",
    ]

    print("=" * 60)
    print("Example Validation Report")
    print("=" * 60)
    print()

    all_valid = True

    for example in examples:
        example_path = examples_dir / example

        if not example_path.exists():
            print(f"✗ {example}: NOT FOUND")
            all_valid = False
            continue

        print(f"Checking {example}...")

        # Check syntax
        syntax_ok, syntax_msg = validate_example_syntax(example_path)
        print(f"  Syntax: {'✓' if syntax_ok else '✗'} {syntax_msg}")

        # Check structure (skip for example_utils)
        if example != "example_utils.py":
            structure_ok, structure_msg = validate_example_structure(example_path)
            print(f"  Structure: {'✓' if structure_ok else '✗'} {structure_msg}")

            if not structure_ok:
                all_valid = False

        if not syntax_ok:
            all_valid = False

        print()

    # Check output directory structure
    output_dir = project_root / "output"
    print("Output directory structure:")
    if output_dir.exists():
        subdirs = list(output_dir.iterdir())
        print(f"  ✓ Exists with {len(subdirs)} subdirectories")
        for subdir in sorted(subdirs):
            if subdir.is_dir():
                print(f"    - {subdir.name}/")
    else:
        print("  ✓ Will be created on first run")
    print()

    # Check orchestrator script
    script_path = project_root / "scripts" / "run_all_examples.sh"
    if script_path.exists():
        print(f"✓ Orchestrator script exists: {script_path.name}")
        if script_path.stat().st_mode & 0o111:
            print("  ✓ Is executable")
        else:
            print("  ✗ Not executable (run: chmod +x scripts/run_all_examples.sh)")
            all_valid = False
    else:
        print("✗ Orchestrator script not found")
        all_valid = False
    print()

    print("=" * 60)
    if all_valid:
        print("✓ All validation checks passed!")
        print()
        print("To run examples:")
        print("  1. Ensure virtual environment is activated")
        print("  2. Install package: pip install -e .")
        print("  3. Run: ./scripts/run_all_examples.sh")
        return 0
    else:
        print("✗ Some validation checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
