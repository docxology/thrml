"""Data validation and comprehensive reporting utilities.

Provides robust data validation, quality checks, and report generation:
- Data integrity validation
- Distribution validation
- Numerical stability checks
- Comprehensive HTML/text reports
- Quality metrics

All validation uses real checks, not mocks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass
class ValidationResult:
    """Result from a validation check.

    Attributes:
        check_name: Name of the validation check
        passed: Whether check passed
        message: Descriptive message
        severity: 'info', 'warning', or 'error'
        details: Additional details

    """

    check_name: str
    passed: bool
    message: str
    severity: str = "info"  # 'info', 'warning', 'error'
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        symbol = "✓" if self.passed else "✗"
        return f"[{self.severity.upper()}] {symbol} {self.check_name}: {self.message}"


class DataValidator:
    """Comprehensive data validator for active inference experiments.

    Performs validation checks on:
    - Data integrity (NaN, Inf, shape consistency)
    - Distribution properties (normalization, valid probabilities)
    - Numerical stability
    - Model consistency

    **Example:**
    ```python
    validator = DataValidator()

    # Validate beliefs
    result = validator.validate_distribution(beliefs, "belief_distribution")
    if not result.passed:
        print(result)

    # Validate model matrices
    results = validator.validate_generative_model(model)
    validator.print_report(results)
    ```
    """

    def __init__(self, tolerance: float = 1e-6):
        """Initialize validator.

        **Arguments:**

        - `tolerance`: Numerical tolerance for checks
        """
        self.tolerance = tolerance
        self.results: List[ValidationResult] = []

    def validate_array(
        self,
        array: Float[Array, "..."],
        name: str,
        expected_shape: Optional[Tuple] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> ValidationResult:
        """Validate array properties.

        **Arguments:**

        - `array`: Array to validate
        - `name`: Name for reporting
        - `expected_shape`: Expected shape (None to skip)
        - `min_val`: Minimum expected value (None to skip)
        - `max_val`: Maximum expected value (None to skip)

        **Returns:**

        - ValidationResult with check outcome
        """
        issues = []

        # Check for NaN
        if jnp.any(jnp.isnan(array)):
            n_nan = int(jnp.sum(jnp.isnan(array)))
            issues.append(f"{n_nan} NaN values")

        # Check for Inf
        if jnp.any(jnp.isinf(array)):
            n_inf = int(jnp.sum(jnp.isinf(array)))
            issues.append(f"{n_inf} Inf values")

        # Check shape
        if expected_shape is not None and array.shape != expected_shape:
            issues.append(f"Shape mismatch: got {array.shape}, expected {expected_shape}")

        # Check value range
        if min_val is not None and jnp.min(array) < min_val:
            issues.append(f"Values below minimum: min={float(jnp.min(array)):.4f} < {min_val}")

        if max_val is not None and jnp.max(array) > max_val:
            issues.append(f"Values above maximum: max={float(jnp.max(array)):.4f} > {max_val}")

        if issues:
            result = ValidationResult(
                check_name=f"array_validation_{name}",
                passed=False,
                message="; ".join(issues),
                severity="error",
                details={"shape": array.shape, "dtype": str(array.dtype)},
            )
        else:
            result = ValidationResult(
                check_name=f"array_validation_{name}",
                passed=True,
                message=f"Array valid: shape={array.shape}",
                severity="info",
                details={"shape": array.shape, "dtype": str(array.dtype)},
            )

        self.results.append(result)
        return result

    def validate_distribution(
        self, dist: Float[Array, "n"], name: str, check_normalization: bool = True
    ) -> ValidationResult:
        """Validate probability distribution.

        **Arguments:**

        - `dist`: Probability distribution
        - `name`: Name for reporting
        - `check_normalization`: Whether to check if sums to 1

        **Returns:**

        - ValidationResult
        """
        issues = []

        # Check basic array validity
        if jnp.any(jnp.isnan(dist)):
            issues.append("Contains NaN values")

        if jnp.any(jnp.isinf(dist)):
            issues.append("Contains Inf values")

        # Check non-negativity
        if jnp.any(dist < 0):
            n_negative = int(jnp.sum(dist < 0))
            issues.append(f"{n_negative} negative probability values")

        # Check normalization
        if check_normalization:
            total = float(jnp.sum(dist))
            if not jnp.isclose(total, 1.0, atol=self.tolerance):
                issues.append(f"Not normalized: sum={total:.6f} (expected 1.0)")

        # Check for zero probability
        if jnp.all(dist == 0):
            issues.append("All probabilities are zero")

        if issues:
            result = ValidationResult(
                check_name=f"distribution_{name}",
                passed=False,
                message="; ".join(issues),
                severity="error",
                details={"sum": float(jnp.sum(dist)), "min": float(jnp.min(dist)), "max": float(jnp.max(dist))},
            )
        else:
            result = ValidationResult(
                check_name=f"distribution_{name}",
                passed=True,
                message=f"Valid probability distribution (sum={float(jnp.sum(dist)):.6f})",
                severity="info",
                details={"sum": float(jnp.sum(dist)), "entropy": float(-jnp.sum(dist * jnp.log(dist + 1e-16)))},
            )

        self.results.append(result)
        return result

    def validate_generative_model(self, model: Any) -> List[ValidationResult]:
        """Validate generative model matrices.

        **Arguments:**

        - `model`: GenerativeModel instance

        **Returns:**

        - List of ValidationResults
        """
        results = []

        # Validate A (observation model)
        result = self.validate_array(model.A, "observation_model_A", min_val=0.0, max_val=1.0)
        results.append(result)

        # Check A normalization (each column should sum to 1)
        a_col_sums = jnp.sum(model.A, axis=0)
        if not jnp.allclose(a_col_sums, 1.0, atol=self.tolerance):
            result = ValidationResult(
                check_name="observation_model_normalization",
                passed=False,
                message=f"A matrix columns don't sum to 1: range=[{float(jnp.min(a_col_sums)):.4f}, {float(jnp.max(a_col_sums)):.4f}]",
                severity="error",
            )
        else:
            result = ValidationResult(
                check_name="observation_model_normalization",
                passed=True,
                message="A matrix properly normalized",
                severity="info",
            )
        results.append(result)
        self.results.append(result)

        # Validate B (transition model)
        result = self.validate_array(model.B, "transition_model_B", min_val=0.0, max_val=1.0)
        results.append(result)

        # Check B normalization (each matrix should be row-stochastic)
        for action in range(model.B.shape[0]):
            b_action = model.B[action]
            row_sums = jnp.sum(b_action, axis=1)
            if not jnp.allclose(row_sums, 1.0, atol=self.tolerance):
                result = ValidationResult(
                    check_name=f"transition_model_action_{action}",
                    passed=False,
                    message=f"B[{action}] rows don't sum to 1",
                    severity="error",
                )
            else:
                result = ValidationResult(
                    check_name=f"transition_model_action_{action}",
                    passed=True,
                    message=f"B[{action}] properly normalized",
                    severity="info",
                )
            results.append(result)
            self.results.append(result)

        # Validate D (prior)
        result = self.validate_distribution(model.D, "prior_D")
        results.append(result)

        # Validate C (preferences)
        result = self.validate_array(model.C, "preferences_C")
        results.append(result)

        return results

    def validate_trajectory(
        self,
        beliefs: List[Float[Array, "n_states"]],
        actions: Optional[List[int]] = None,
        observations: Optional[List[int]] = None,
    ) -> List[ValidationResult]:
        """Validate agent trajectory data.

        **Arguments:**

        - `beliefs`: List of belief distributions
        - `actions`: List of actions (optional)
        - `observations`: List of observations (optional)

        **Returns:**

        - List of ValidationResults
        """
        results = []

        # Check trajectory length consistency
        if actions is not None and len(beliefs) != len(actions) + 1:
            result = ValidationResult(
                check_name="trajectory_length_consistency",
                passed=False,
                message=f"Length mismatch: {len(beliefs)} beliefs, {len(actions)} actions",
                severity="warning",
            )
        else:
            result = ValidationResult(
                check_name="trajectory_length_consistency",
                passed=True,
                message=f"Trajectory length consistent: {len(beliefs)} steps",
                severity="info",
            )
        results.append(result)
        self.results.append(result)

        # Validate each belief
        for i, belief in enumerate(beliefs):
            result = self.validate_distribution(belief, f"belief_step_{i}")
            if not result.passed:
                results.append(result)

        # Check for belief collapse
        belief_entropies = [float(-jnp.sum(b * jnp.log(b + 1e-16))) for b in beliefs]
        if any(e < 0.1 for e in belief_entropies):
            result = ValidationResult(
                check_name="belief_collapse_check",
                passed=False,
                message=f"Low entropy detected: min={min(belief_entropies):.4f}",
                severity="warning",
                details={"entropies": belief_entropies},
            )
        else:
            result = ValidationResult(
                check_name="belief_collapse_check",
                passed=True,
                message=f"Entropy range: [{min(belief_entropies):.4f}, {max(belief_entropies):.4f}]",
                severity="info",
                details={"entropies": belief_entropies},
            )
        results.append(result)
        self.results.append(result)

        return results

    def print_report(self, results: Optional[List[ValidationResult]] = None):
        """Print validation report.

        **Arguments:**

        - `results`: Results to print (None to use all stored results)
        """
        if results is None:
            results = self.results

        print("\n" + "=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)

        n_passed = sum(1 for r in results if r.passed)
        n_failed = sum(1 for r in results if not r.passed)
        n_errors = sum(1 for r in results if not r.passed and r.severity == "error")
        n_warnings = sum(1 for r in results if not r.passed and r.severity == "warning")

        print(f"\nTotal checks: {len(results)}")
        print(f"Passed: {n_passed} ✓")
        print(f"Failed: {n_failed} ✗")
        print(f"  - Errors: {n_errors}")
        print(f"  - Warnings: {n_warnings}")
        print()

        # Print failed checks
        if n_failed > 0:
            print("-" * 70)
            print("FAILED CHECKS:")
            print("-" * 70)
            for result in results:
                if not result.passed:
                    print(result)

        print("=" * 70 + "\n")

    def generate_html_report(self, output_path: Path, title: str = "Validation Report"):
        """Generate HTML validation report.

        **Arguments:**

        - `output_path`: Path to save HTML report
        - `title`: Report title
        """
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append(f"<title>{title}</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1 { color: #333; }")
        html.append(".passed { color: green; }")
        html.append(".failed { color: red; }")
        html.append(".warning { color: orange; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        html.append(f"<h1>{title}</h1>")
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

        # Summary
        n_passed = sum(1 for r in self.results if r.passed)
        n_failed = sum(1 for r in self.results if not r.passed)

        html.append("<h2>Summary</h2>")
        html.append(f"<p>Total checks: {len(self.results)}</p>")
        html.append(f"<p class='passed'>Passed: {n_passed}</p>")
        html.append(f"<p class='failed'>Failed: {n_failed}</p>")

        # Results table
        html.append("<h2>Validation Results</h2>")
        html.append("<table>")
        html.append("<tr><th>Check</th><th>Status</th><th>Message</th><th>Severity</th></tr>")

        for result in self.results:
            status_class = "passed" if result.passed else "failed"
            status = "✓ Pass" if result.passed else "✗ Fail"
            html.append(f"<tr class='{status_class}'>")
            html.append(f"<td>{result.check_name}</td>")
            html.append(f"<td>{status}</td>")
            html.append(f"<td>{result.message}</td>")
            html.append(f"<td>{result.severity.upper()}</td>")
            html.append("</tr>")

        html.append("</table>")
        html.append("</body>")
        html.append("</html>")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(html))

        print(f"📋 HTML validation report saved to {output_path}")

    def clear_results(self):
        """Clear stored validation results."""
        self.results = []


def validate_experiment_outputs(output_dir: Path, required_files: Optional[List[str]] = None) -> List[ValidationResult]:
    """Validate experiment output files.

    **Arguments:**

    - `output_dir`: Output directory to validate
    - `required_files`: List of required file patterns

    **Returns:**

    - List of ValidationResults
    """
    results = []

    # Check directory exists
    if not output_dir.exists():
        result = ValidationResult(
            check_name="output_directory_exists",
            passed=False,
            message=f"Output directory not found: {output_dir}",
            severity="error",
        )
        results.append(result)
        return results

    # Check required subdirectories
    required_dirs = ["data", "logs", "config"]
    for dir_name in required_dirs:
        dir_path = output_dir / dir_name
        if not dir_path.exists():
            result = ValidationResult(
                check_name=f"required_directory_{dir_name}",
                passed=False,
                message=f"Missing required directory: {dir_name}/",
                severity="warning",
            )
        else:
            n_files = len(list(dir_path.glob("*")))
            result = ValidationResult(
                check_name=f"required_directory_{dir_name}",
                passed=True,
                message=f"Directory exists with {n_files} files",
                severity="info",
            )
        results.append(result)

    # Check for required files
    if required_files:
        for file_pattern in required_files:
            matching_files = list(output_dir.rglob(file_pattern))
            if not matching_files:
                result = ValidationResult(
                    check_name=f"required_file_{file_pattern}",
                    passed=False,
                    message=f"Required file not found: {file_pattern}",
                    severity="warning",
                )
            else:
                result = ValidationResult(
                    check_name=f"required_file_{file_pattern}",
                    passed=True,
                    message=f"Found {len(matching_files)} matching files",
                    severity="info",
                )
            results.append(result)

    return results
