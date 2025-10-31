"""Comprehensive visualization module for active inference and THRML.

This module provides extensive plotting and visualization capabilities for:
- Active inference agents and their behavior
- THRML sampling trajectories and statistics
- Probabilistic graphical models and networks
- Statistical distributions and convergence metrics
- Environment states and agent interactions
"""

from .active_inference_plots import (  # Active inference visualizations
    plot_action_distribution,
    plot_agent_performance,
    plot_belief_evolution,
    plot_belief_trajectory,
    plot_expected_free_energy,
    plot_free_energy,
    plot_free_energy_components,
    plot_policy_selection,
    plot_precision_trajectory,
    plot_state_occupancy,
)
from .animation import (  # Animation utilities
    create_belief_animation,
    create_sampling_animation,
    create_trajectory_animation,
)
from .comparison_plots import (  # Multi-agent/multi-run comparisons
    plot_ablation_study,
    plot_learning_curves,
    plot_multi_agent_comparison,
    plot_parameter_sweep,
)
from .core import (  # Core plotting utilities; Configuration
    VisualizationConfig,
    create_figure,
    get_default_output_dir,
    save_figure,
    set_output_dir,
    set_style,
)
from .environment_plots import (  # Environment visualization
    plot_agent_trajectory,
    plot_grid_world,
    plot_heatmap_occupancy,
    plot_observation_history,
    plot_reward_over_time,
    plot_tmaze,
)
from .network_plots import (  # Network and graph visualization
    plot_factor_graph,
    plot_generative_model_structure,
    plot_graphical_model,
    plot_interaction_graph,
    plot_markov_blanket,
)
from .statistical_plots import (  # Statistical distributions; Statistical analysis plots
    plot_convergence_diagnostics,
    plot_correlation_matrix,
    plot_distribution,
    plot_distribution_comparison,
    plot_effective_sample_size,
    plot_histogram,
    plot_kde,
    plot_pairwise_relationships,
    plot_qq_plot,
    plot_residuals,
    plot_rhat,
    plot_scatter_with_regression,
)
from .thrml_plots import (  # THRML integration
    plot_acceptance_rates,
    plot_autocorrelation,
    plot_block_structure,
    plot_categorical_states,
    plot_energy_landscape,
    plot_ising_spins,
    plot_mixing_diagnostics,
    plot_sample_statistics,
    plot_sampling_trajectory,
)

__all__ = [
    # Core
    "create_figure",
    "save_figure",
    "set_style",
    "VisualizationConfig",
    "get_default_output_dir",
    "set_output_dir",
    # Active Inference
    "plot_belief_trajectory",
    "plot_belief_evolution",
    "plot_free_energy",
    "plot_free_energy_components",
    "plot_expected_free_energy",
    "plot_action_distribution",
    "plot_policy_selection",
    "plot_precision_trajectory",
    "plot_agent_performance",
    "plot_state_occupancy",
    # THRML
    "plot_sampling_trajectory",
    "plot_sample_statistics",
    "plot_energy_landscape",
    "plot_block_structure",
    "plot_mixing_diagnostics",
    "plot_autocorrelation",
    "plot_acceptance_rates",
    "plot_ising_spins",
    "plot_categorical_states",
    # Networks
    "plot_graphical_model",
    "plot_factor_graph",
    "plot_interaction_graph",
    "plot_generative_model_structure",
    "plot_markov_blanket",
    # Statistical
    "plot_distribution",
    "plot_distribution_comparison",
    "plot_histogram",
    "plot_kde",
    "plot_qq_plot",
    "plot_convergence_diagnostics",
    "plot_rhat",
    "plot_effective_sample_size",
    "plot_scatter_with_regression",
    "plot_correlation_matrix",
    "plot_residuals",
    "plot_pairwise_relationships",
    # Environment
    "plot_grid_world",
    "plot_tmaze",
    "plot_agent_trajectory",
    "plot_observation_history",
    "plot_reward_over_time",
    "plot_heatmap_occupancy",
    # Comparison
    "plot_multi_agent_comparison",
    "plot_learning_curves",
    "plot_parameter_sweep",
    "plot_ablation_study",
    # Animation
    "create_belief_animation",
    "create_trajectory_animation",
    "create_sampling_animation",
]
