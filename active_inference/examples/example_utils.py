"""Utilities for example scripts: logging, saving outputs, configuration management.

Enhanced orchestrator with:
- Comprehensive logging with context managers
- Real-time metrics tracking and validation
- THRML integration support
- Advanced visualization utilities
- Data validation and reporting
- Performance profiling

**THRML Integration Support**:
- ExampleRunner is THRML-aware and works seamlessly with THRML inference methods
- **Examples 00-06, 09-15 use THRML sampling-based inference**
- **Examples 00-02**: Direct notebook translations with complete THRML methods
- **Examples 03-06, 09-10, 12, 14-15**: Use THRML via `ThrmlInferenceEngine`
- Real THRML methods used: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`, `FactorSamplingProgram`, `sample_states`
- Examples load THRML parameters (n_samples, n_warmup, steps_per_sample) from config
- See Example 11 (`11_thrml_comprehensive.py`) for comprehensive direct THRML usage
- THRML methods available via `active_inference.inference.ThrmlInferenceEngine`
- All THRML usage is verified and uses real methods (no mocks)
"""

import json
import logging
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Use non-interactive backend for server environments
matplotlib.use("Agg")


def load_examples_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load centralized examples configuration.

    Args:
        config_path: Path to config file. If None, looks for examples_config.yaml
                    in the examples directory.

    Returns:
        Dictionary containing all configuration

    """
    if config_path is None:
        config_path = Path(__file__).parent / "examples_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_example_config(example_name: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get configuration for a specific example.

    Args:
        example_name: Name of example (e.g., "example_01" or "01_basic_inference")
        config: Full config dict. If None, loads from file.

    Returns:
        Dictionary with example-specific config merged with global defaults

    """
    if config is None:
        config = load_examples_config()

    # Normalize example name
    if not example_name.startswith("example_"):
        # Convert "01_basic_inference" to "example_01"
        if "_" in example_name:
            example_num = example_name.split("_")[0]
            example_name = f"example_{example_num}"

    # Get example-specific config
    example_config = config.get(example_name, {}).copy()

    # Merge with global config
    global_config = config.get("global", {})
    for key, value in global_config.items():
        if key not in example_config:
            example_config[key] = value

    # Merge with defaults if keys are missing
    defaults = config.get("defaults", {})
    for category, defaults_dict in defaults.items():
        if category not in example_config:
            example_config[category] = {}
        for key, value in defaults_dict.items():
            example_config.setdefault(category, {}).setdefault(key, value)

    return example_config


class ExampleRunner:
    """Enhanced orchestrator for example execution with comprehensive logging and tracking.

    Features:
    - Structured logging with context managers
    - Real-time metrics tracking and validation
    - Performance profiling and timing
    - Automatic data validation and reporting
    - Error handling and recovery
    - THRML integration support
    - Advanced visualization management

    Each example gets:
    - Dedicated timestamped output directory
    - Structured logging to file and console
    - Automatic saving of configurations, data, and plots
    - Performance metrics and profiling data
    - Validation reports and error logs
    """

    def __init__(
        self,
        example_name: str,
        output_base: Path = None,
        log_level: int = None,
        enable_profiling: bool = None,
        enable_validation: bool = None,
        config: Optional[Dict] = None,
        use_config_file: bool = True,
    ):
        """Initialize example runner with enhanced logging and tracking.

        Args:
            example_name: Name of the example (e.g., "01_basic_inference")
            output_base: Base directory for all outputs (overrides config)
            log_level: Logging level (overrides config)
            enable_profiling: Whether to enable performance profiling (overrides config)
            enable_validation: Whether to enable data validation (overrides config)
            config: Pre-loaded config dict (optional)
            use_config_file: Whether to load config from examples_config.yaml

        """
        self.example_name = example_name

        # Load configuration if requested
        self.config = {}
        if use_config_file:
            try:
                self.config = get_example_config(example_name, config)
            except FileNotFoundError:
                print(f"Warning: Config file not found for {example_name}, using defaults")
        elif config:
            self.config = config

        # Set parameters from config or arguments (arguments override config)
        if output_base is None:
            output_base_str = self.config.get("output_base", "../output")
            output_base = Path(output_base_str)
        else:
            output_base = Path(output_base)

        if log_level is None:
            log_level_str = self.config.get("log_level", "INFO")
            log_level = getattr(logging, log_level_str, logging.INFO)

        if enable_profiling is None:
            enable_profiling = self.config.get("enable_profiling", False)

        if enable_validation is None:
            enable_validation = self.config.get("enable_validation", True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enable_profiling = enable_profiling
        self.enable_validation = enable_validation

        # Create output directory structure
        self.output_base = output_base
        self.output_dir = output_base / example_name / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.logs_dir = self.output_dir / "logs"
        self.data_dir = self.output_dir / "data"
        self.plots_dir = self.output_dir / "plots"
        self.config_dir = self.output_dir / "config"
        self.reports_dir = self.output_dir / "reports"
        self.profile_dir = self.output_dir / "profiling"

        for dir_path in [
            self.logs_dir,
            self.data_dir,
            self.plots_dir,
            self.config_dir,
            self.reports_dir,
            self.profile_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging(log_level)

        # Track execution metrics
        self.metrics: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.validation_results: List[Dict[str, Any]] = []
        self.section_timings: Dict[str, float] = {}

        # Execution state
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_section: Optional[str] = None
        self.section_start_time: Optional[float] = None
        self.errors: List[Dict[str, Any]] = []

        self.logger.info(f"{'='*60}")
        self.logger.info(f"  {example_name}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Profiling: {'enabled' if enable_profiling else 'disabled'}")
        self.logger.info(f"Validation: {'enabled' if enable_validation else 'disabled'}")

    def _setup_logging(self, log_level: int) -> logging.Logger:
        """Configure logging to both file and console."""
        logger = logging.getLogger(self.example_name)
        logger.setLevel(log_level)
        logger.handlers.clear()  # Remove any existing handlers

        # File handler
        log_file = self.logs_dir / "execution.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    @contextmanager
    def section(self, section_name: str):
        """Context manager for timed and logged sections.

        Example:
            with runner.section("Model Creation"):
                model = create_model()

        """
        self.current_section = section_name
        self.section_start_time = time.time()

        self.logger.info(f"\n{'─'*60}")
        self.logger.info(f"▸ {section_name}")
        self.logger.info(f"{'─'*60}")

        try:
            yield
        except Exception as e:
            self.logger.error(f"✗ Error in section '{section_name}': {str(e)}")
            self.errors.append(
                {
                    "section": section_name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            raise
        finally:
            duration = time.time() - self.section_start_time
            self.section_timings[section_name] = duration
            self.logger.info(f"✓ {section_name} completed in {duration:.2f}s")
            self.current_section = None

    def start(self):
        """Mark start of example execution."""
        self.start_time = datetime.now()
        self.logger.info(f"\n▸ Execution started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.enable_profiling:
            self.logger.info("⏱️  Performance profiling enabled")

    def end(self):
        """Mark end of example execution and save comprehensive summary."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        self.logger.info(f"\n{'='*60}")
        self.logger.info("  EXECUTION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Completed at: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total duration: {duration:.2f}s")

        # Log section timings
        if self.section_timings:
            self.logger.info("\nSection Timings:")
            for section, timing in self.section_timings.items():
                pct = (timing / duration) * 100
                self.logger.info(f"  • {section}: {timing:.2f}s ({pct:.1f}%)")

        # Log metrics
        if self.metrics:
            self.logger.info(f"\nRecorded Metrics: {len(self.metrics)}")
            for key, value in list(self.metrics.items())[:5]:  # Show first 5
                self.logger.info(f"  • {key}: {value}")
            if len(self.metrics) > 5:
                self.logger.info(f"  ... and {len(self.metrics) - 5} more")

        # Log validation results
        if self.validation_results:
            passed = sum(1 for v in self.validation_results if v.get("passed", False))
            failed = len(self.validation_results) - passed
            self.logger.info(f"\nValidation Results: {passed} passed, {failed} failed")

        # Log errors
        if self.errors:
            self.logger.warning(f"\nErrors encountered: {len(self.errors)}")
            for err in self.errors:
                self.logger.warning(f"  • {err['section']}: {err['error']}")

        # Create comprehensive output manifest
        self.create_output_manifest()

        # Save comprehensive summary
        self.save_summary(duration)

        # Save validation report if enabled
        if self.enable_validation and self.validation_results:
            self.save_validation_report()

        self.logger.info(f"\n{'='*60}")

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """Save configuration to JSON file with detailed logging.

        Args:
            config: Configuration dictionary
            filename: Output filename

        """
        config_path = self.config_dir / filename

        # Convert numpy/jax arrays to lists
        serializable_config = self._make_serializable(config)

        with open(config_path, "w") as f:
            json.dump(serializable_config, f, indent=2)

        # Get file size
        file_size = config_path.stat().st_size
        size_str = self._format_file_size(file_size)

        self.logger.info(f"💾 Config saved: {config_path.name} ({size_str})")

        # Log key parameters (first 5)
        if config:
            self.logger.info("   Key parameters:")
            for i, (key, value) in enumerate(list(config.items())[:5]):
                if not isinstance(value, (dict, list, np.ndarray, jnp.ndarray)):
                    self.logger.info(f"   • {key}: {value}")
            if len(config) > 5:
                self.logger.info(f"   ... and {len(config) - 5} more parameters")

    def save_data(
        self,
        data: Any,
        filename: str,
        format: str = "npz",
    ):
        """Save data to file with detailed logging.

        Args:
            data: Data to save (numpy/jax array or dict of arrays)
            filename: Output filename (without extension)
            format: Format ('npz', 'npy', 'json')

        """
        if format == "npz":
            data_path = self.data_dir / f"{filename}.npz"
            if isinstance(data, dict):
                # Convert jax arrays to numpy
                np_data = {k: np.array(v) for k, v in data.items()}
                np.savez(data_path, **np_data)
                n_arrays = len(np_data)
            else:
                np.savez(data_path, data=np.array(data))
                n_arrays = 1
        elif format == "npy":
            data_path = self.data_dir / f"{filename}.npy"
            np.save(data_path, np.array(data))
            n_arrays = 1
        elif format == "json":
            data_path = self.data_dir / f"{filename}.json"
            serializable_data = self._make_serializable(data)
            with open(data_path, "w") as f:
                json.dump(serializable_data, f, indent=2)
            n_arrays = len(data) if isinstance(data, dict) else 1
        else:
            raise ValueError(f"Unknown format: {format}")

        # Get file size
        file_size = data_path.stat().st_size
        size_str = self._format_file_size(file_size)

        self.logger.info(f"💾 Data saved: {data_path.name} ({size_str}, {n_arrays} array{'s' if n_arrays > 1 else ''})")
        self.logger.info(f"   Location: {data_path}")

    def save_plot(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = None,
        formats: list = None,
    ):
        """Save matplotlib figure with detailed logging.

        Args:
            fig: Matplotlib figure
            filename: Output filename (without extension)
            dpi: Resolution
            formats: List of formats to save (default: ['png', 'pdf'])

        """
        # Use config values if not specified
        if dpi is None:
            dpi = self.config.get("dpi", 150)
        if formats is None:
            formats = self.config.get("plot_formats", ["png"])

        saved_files = []
        for fmt in formats:
            plot_path = self.plots_dir / f"{filename}.{fmt}"
            fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")

            # Get file size
            file_size = plot_path.stat().st_size
            size_str = self._format_file_size(file_size)
            saved_files.append((plot_path.name, size_str, fmt))

        # Log all saved formats
        if len(saved_files) == 1:
            name, size, fmt = saved_files[0]
            self.logger.info(f"📊 Plot saved: {name} ({size})")
        else:
            self.logger.info(f"📊 Plot saved in {len(saved_files)} formats:")
            for name, size, fmt in saved_files:
                self.logger.info(f"   • {name} ({size})")

    def record_metric(self, name: str, value: Any, log: bool = True):
        """Record a metric value with optional logging.

        Args:
            name: Metric name
            value: Metric value
            log: Whether to log the metric immediately

        """
        self.metrics[name] = value
        if log:
            self.logger.info(f"📊 Metric: {name} = {value}")

    def validate_data(self, data: Any, name: str, checks: Optional[Dict[str, callable]] = None) -> bool:
        """Validate data with custom checks.

        Args:
            data: Data to validate
            name: Name for logging
            checks: Dictionary of {check_name: check_function}

        Returns:
            True if all checks pass

        """
        if not self.enable_validation:
            return True

        validation_result = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "passed": True,
        }

        # Default checks
        default_checks = {
            "not_none": lambda d: d is not None,
            "not_empty": lambda d: (not hasattr(d, "__len__")) or len(d) > 0,
        }

        # Add array-specific checks if applicable
        if isinstance(data, (np.ndarray, jnp.ndarray)):
            default_checks["no_nan"] = lambda d: not np.any(np.isnan(d))
            default_checks["finite"] = lambda d: np.all(np.isfinite(d))

        all_checks = {**default_checks, **(checks or {})}

        for check_name, check_fn in all_checks.items():
            try:
                result = check_fn(data)
                validation_result["checks"][check_name] = {"passed": bool(result), "error": None}
                if not result:
                    validation_result["passed"] = False
                    self.logger.warning(f"⚠️  Validation '{check_name}' failed for {name}")
            except Exception as e:
                validation_result["checks"][check_name] = {"passed": False, "error": str(e)}
                validation_result["passed"] = False
                self.logger.warning(f"⚠️  Validation '{check_name}' error for {name}: {e}")

        self.validation_results.append(validation_result)

        if validation_result["passed"]:
            self.logger.info(f"✓ Validation passed for {name}")
        else:
            self.logger.warning(f"✗ Validation failed for {name}")

        return validation_result["passed"]

    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling specific operations.

        Example:
            with runner.profile("Inference"):
                posterior = infer_states(...)

        """
        if not self.enable_profiling:
            yield
            return

        start_time = time.time()
        self.logger.debug(f"⏱️  Starting profiling: {operation_name}")

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.performance_metrics[operation_name] = duration
            self.logger.debug(f"⏱️  {operation_name}: {duration:.4f}s")

    def save_validation_report(self):
        """Save validation report to file."""
        report_path = self.reports_dir / "validation_report.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "example": self.example_name,
            "total_validations": len(self.validation_results),
            "passed": sum(1 for v in self.validation_results if v["passed"]),
            "failed": sum(1 for v in self.validation_results if not v["passed"]),
            "validations": self.validation_results,
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"📋 Validation report saved to {report_path}")

    def create_output_manifest(self):
        """Create comprehensive manifest of all generated outputs."""
        self.logger.info(f"\n{'─'*60}")
        self.logger.info("OUTPUT MANIFEST")
        self.logger.info(f"{'─'*60}")

        manifest = {
            "example_name": self.example_name,
            "timestamp": self.timestamp,
            "output_directory": str(self.output_dir),
            "files": {},
        }

        # Scan each output directory
        for dir_name, dir_path in [
            ("config", self.config_dir),
            ("data", self.data_dir),
            ("plots", self.plots_dir),
            ("logs", self.logs_dir),
            ("reports", self.reports_dir),
            ("profiling", self.profile_dir),
        ]:
            files_info = []
            total_size = 0

            if dir_path.exists():
                for file_path in dir_path.iterdir():
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        total_size += size
                        files_info.append(
                            {
                                "name": file_path.name,
                                "size": size,
                                "size_formatted": self._format_file_size(size),
                            }
                        )

            if files_info:
                manifest["files"][dir_name] = {
                    "count": len(files_info),
                    "total_size": total_size,
                    "total_size_formatted": self._format_file_size(total_size),
                    "files": files_info,
                }

                self.logger.info(
                    f"\n{dir_name.upper()}/ ({len(files_info)} files, {self._format_file_size(total_size)})"
                )
                for file_info in sorted(files_info, key=lambda x: x["name"]):
                    self.logger.info(f"  • {file_info['name']} ({file_info['size_formatted']})")

        # Calculate total
        total_files = sum(cat["count"] for cat in manifest["files"].values())
        total_size = sum(cat["total_size"] for cat in manifest["files"].values())

        manifest["totals"] = {
            "files": total_files,
            "size": total_size,
            "size_formatted": self._format_file_size(total_size),
        }

        self.logger.info(f"\n{'─'*60}")
        self.logger.info(f"TOTAL: {total_files} files, {self._format_file_size(total_size)}")
        self.logger.info(f"{'─'*60}")

        # Save manifest to JSON
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"\n📋 Output manifest saved to {manifest_path}")

    def save_summary(self, duration: float):
        """Save comprehensive execution summary with all metrics and timings."""
        # Get manifest data for totals
        manifest_totals = {"files": 0, "size": 0}
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
                manifest_totals = manifest_data.get("totals", manifest_totals)

        summary = {
            "example_name": self.example_name,
            "timestamp": self.timestamp,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "duration_seconds": duration,
            "metrics": self._make_serializable(self.metrics),
            "performance_metrics": self._make_serializable(self.performance_metrics),
            "section_timings": self._make_serializable(self.section_timings),
            "validation_summary": {
                "total": len(self.validation_results),
                "passed": sum(1 for v in self.validation_results if v.get("passed", False)),
                "failed": sum(1 for v in self.validation_results if not v.get("passed", True)),
            },
            "output_summary": {
                "total_files": manifest_totals.get("files", 0),
                "total_size_bytes": manifest_totals.get("size", 0),
                "total_size_formatted": manifest_totals.get("size_formatted", "0B"),
            },
            "errors": self.errors,
            "profiling_enabled": self.enable_profiling,
            "validation_enabled": self.enable_validation,
            "output_directory": str(self.output_dir),
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"📋 Summary saved to {summary_path}")

    def get_plot_path(self, filename: str) -> Path:
        """Get full path for a plot file.

        Args:
            filename: Plot filename (with extension)

        Returns:
            Full path to plot file

        """
        return self.plots_dir / filename

    def get_report_path(self, filename: str) -> Path:
        """Get full path for a report file.

        Args:
            filename: Report filename (with extension)

        Returns:
            Full path to report file

        """
        return self.reports_dir / filename

    def save_report(self, content: str, filename: str):
        """Save text report to file.

        Args:
            content: Report content (text)
            filename: Report filename

        """
        report_path = self.reports_dir / filename
        with open(report_path, "w") as f:
            f.write(content)

        file_size = report_path.stat().st_size
        size_str = self._format_file_size(file_size)

        self.logger.info(f"📋 Report saved: {filename} ({size_str})")

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (can use dot notation for nested, e.g., "model.n_states")
            default: Default value if key not found

        Returns:
            Configuration value or default

        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating, jnp.floating)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


def create_figure(nrows: int = 1, ncols: int = 1, figsize: tuple = None) -> tuple:
    """Create matplotlib figure with consistent styling.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height)

    Returns:
        Tuple of (figure, axes)

    """
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.tight_layout(pad=3.0)

    return fig, axes


def plot_array_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    cmap: str = "viridis",
):
    """Plot 2D array as heatmap.

    Args:
        ax: Matplotlib axes
        data: 2D array to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap name

    """
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)


def plot_line_series(
    ax: plt.Axes,
    data: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    label: str = None,
):
    """Plot 1D array as line plot.

    Args:
        ax: Matplotlib axes
        data: 1D array to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        label: Line label for legend

    """
    ax.plot(data, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend()
    ax.grid(True, alpha=0.3)


def plot_distribution(
    ax: plt.Axes,
    probs: np.ndarray,
    title: str = "",
    xlabel: str = "State",
    ylabel: str = "Probability",
    labels: list = None,
):
    """Plot probability distribution as bar chart.

    Args:
        ax: Matplotlib axes
        probs: Probability distribution
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        labels: State labels

    """
    x = np.arange(len(probs))
    ax.bar(x, probs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis="y")


def plot_confusion_matrix(
    ax: plt.Axes,
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    n_states: int,
    title: str = "Confusion Matrix",
):
    """Plot confusion matrix for state predictions.

    Args:
        ax: Matplotlib axes
        true_states: Array of true state indices
        predicted_states: Array of predicted state indices
        n_states: Number of possible states
        title: Plot title

    """
    confusion = np.zeros((n_states, n_states))
    for true, pred in zip(true_states, predicted_states):
        confusion[true, pred] += 1

    # Normalize by row
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums, where=row_sums != 0, out=np.zeros_like(confusion))

    im = ax.imshow(confusion_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted State")
    ax.set_ylabel("True State")
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))

    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            text = ax.text(
                j,
                i,
                f"{confusion[i, j]:.0f}",
                ha="center",
                va="center",
                color="white" if confusion_norm[i, j] > 0.5 else "black",
                fontsize=8,
            )

    plt.colorbar(im, ax=ax, label="Accuracy")


def calculate_entropy(probs: np.ndarray) -> float:
    """Calculate Shannon entropy of probability distribution.

    Args:
        probs: Probability distribution

    Returns:
        Entropy in bits

    """
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    return -np.sum(probs * np.log2(probs + 1e-16))


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KL divergence KL[P || Q].

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        KL divergence in nats

    """
    p = np.array(p) + 1e-16
    q = np.array(q) + 1e-16
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * (np.log(p) - np.log(q)))


def calculate_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Calculate prediction accuracy.

    Args:
        predicted: Predicted indices
        actual: Actual indices

    Returns:
        Accuracy (0 to 1)

    """
    return np.mean(predicted == actual)


def plot_metrics_summary(
    ax: plt.Axes,
    metrics: Dict[str, float],
    title: str = "Metrics Summary",
):
    """Plot summary of numerical metrics as horizontal bar chart.

    Args:
        ax: Matplotlib axes
        metrics: Dictionary of metric names and values
        title: Plot title

    """
    names = list(metrics.keys())
    values = list(metrics.values())

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center")
