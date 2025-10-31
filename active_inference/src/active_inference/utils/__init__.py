"""Utility functions for active inference."""

from .metrics import calculate_kl_divergence, calculate_prediction_accuracy
from .resource_tracking import (
    PerformanceProfile,
    ResourceSnapshot,
    ResourceTracker,
    estimate_resources,
    print_resource_estimates,
)
from .statistical_analysis import (
    CorrelationResults,
    RegressionResults,
    anova_one_way,
    compute_effect_size,
    compute_summary_statistics,
    generate_statistical_report,
    linear_regression,
    pearson_correlation,
    t_test_independent,
)
from .validation import DataValidator, ValidationResult, validate_experiment_outputs
from .visualization import plot_belief_trajectory, plot_free_energy

__all__ = [
    # Metrics
    "calculate_kl_divergence",
    "calculate_prediction_accuracy",
    # Visualization
    "plot_belief_trajectory",
    "plot_free_energy",
    # Statistical Analysis
    "linear_regression",
    "pearson_correlation",
    "compute_effect_size",
    "t_test_independent",
    "anova_one_way",
    "compute_summary_statistics",
    "generate_statistical_report",
    "RegressionResults",
    "CorrelationResults",
    # Validation
    "DataValidator",
    "ValidationResult",
    "validate_experiment_outputs",
    # Resource Tracking
    "ResourceTracker",
    "ResourceSnapshot",
    "PerformanceProfile",
    "estimate_resources",
    "print_resource_estimates",
]
