# Visualization Module

Comprehensive visualization utilities for active inference and THRML.

## Overview

This visualization module provides extensive plotting capabilities for:

- **Active Inference**: Belief trajectories, free energy, action distributions, agent performance
- **THRML Sampling**: Sampling trajectories, energy landscapes, mixing diagnostics, convergence metrics
- **Networks**: Graphical models, factor graphs, interaction graphs, Markov blankets
- **Statistical Analysis**: Distributions, convergence diagnostics, R-hat, effective sample size
- **Environments**: Grid worlds, T-mazes, agent trajectories, occupancy heatmaps
- **Comparisons**: Multi-agent comparisons, learning curves, hyperparameter sweeps, ablation studies
- **Animations**: Dynamic visualizations of beliefs, trajectories, and sampling

## Quick Start

```python
from active_inference import visualization as viz

# Set output directory
viz.set_output_dir("my_experiment/output")

# Plot belief trajectory
fig, ax = viz.plot_belief_trajectory(
    beliefs=belief_history,
    true_states=true_states,
    save_path="belief_trajectory.png"
)

# Plot free energy
viz.plot_free_energy(
    free_energies=fe_values,
    save_path="free_energy.png"
)

# Plot grid world with agent
viz.plot_grid_world(
    grid_shape=(10, 10),
    agent_pos=(5, 5),
    goal_pos=(9, 9),
    obstacles=[(3, 3), (4, 4)],
    save_path="grid_world.png"
)
```

## Module Structure

```
visualization/
├── __init__.py                 # Main exports
├── core.py                     # Core utilities and configuration
├── active_inference_plots.py   # Active inference visualizations
├── thrml_plots.py              # THRML-specific plots
├── network_plots.py            # Graph and network visualizations
├── statistical_plots.py        # Statistical analysis plots
├── environment_plots.py        # Environment visualizations
├── comparison_plots.py         # Multi-agent and comparison plots
└── animation.py                # Animation utilities
```

## Configuration

The visualization module uses a global configuration that can be customized:

```python
from active_inference.visualization import get_config, set_output_dir

# Set output directory
set_output_dir("output")

# Access configuration
config = get_config()
config.dpi = 300
config.figure_format = "pdf"
config.default_figsize = (12, 8)
```

### Output Directory Structure

By default, plots are saved to `output/` with the following subdirectories:

- `plots/` - General plots
- `graphs/` - Network and graph visualizations
- `trajectories/` - Trajectory and path visualizations
- `distributions/` - Statistical distributions
- `networks/` - Network structure plots
- `animations/` - Animated visualizations
- `reports/` - Generated reports

## Active Inference Plots

### Belief Visualization

```python
# Plot belief trajectory
viz.plot_belief_trajectory(beliefs, true_states, save_path="beliefs.png")

# Plot belief evolution for individual states
viz.plot_belief_evolution(beliefs, state_labels=["State A", "State B"])

# Plot state occupancy
viz.plot_state_occupancy(state_visits, state_labels=labels)
```

### Free Energy

```python
# Plot variational free energy
viz.plot_free_energy(vfe_values, show_moving_average=True)

# Plot free energy components
viz.plot_free_energy_components(vfe, accuracy, complexity)

# Plot expected free energy by policy
viz.plot_expected_free_energy(efe_per_policy, selected_policy=2)
```

### Agent Behavior

```python
# Plot action distribution
viz.plot_action_distribution(action_probs, action_names=["Up", "Down", "Left", "Right"])

# Plot policy selection
viz.plot_policy_selection(policy_history, n_policies=4)

# Plot precision trajectory
viz.plot_precision_trajectory(precisions, labels=["Sensory", "State", "Action"])

# Plot comprehensive performance
viz.plot_agent_performance(rewards, free_energies, entropies)
```

## THRML Plots

### Sampling

```python
# Plot sampling trajectory
viz.plot_sampling_trajectory(samples, dims=(0, 1))

# Plot sample statistics
viz.plot_sample_statistics(samples)

# Plot energy landscape
viz.plot_energy_landscape(energy_fn, x_range=(-3, 3), y_range=(-3, 3))
```

### Convergence

```python
# Plot mixing diagnostics
viz.plot_mixing_diagnostics(samples_by_chain)

# Plot autocorrelation
viz.plot_autocorrelation(samples[:, 0], max_lag=50)

# Plot acceptance rates
viz.plot_acceptance_rates(acceptance_history)
```

### Model-Specific

```python
# Plot Ising spins
viz.plot_ising_spins(spin_configurations, timesteps=[0, 10, 50, 100])

# Plot categorical states
viz.plot_categorical_states(categorical_samples, n_categories=5)

# Plot block structure
viz.plot_block_structure(blocks)
```

## Network Plots

```python
# Plot graphical model
viz.plot_graphical_model(nodes, edges, node_labels=labels)

# Plot factor graph
viz.plot_factor_graph(variable_nodes, factor_nodes, connections)

# Plot generative model structure
viz.plot_generative_model_structure(n_obs=4, n_states=4, n_actions=2)

# Plot Markov blanket
viz.plot_markov_blanket(node_idx=0, nodes=all_nodes, edges=all_edges)
```

## Statistical Plots

```python
# Plot distribution
viz.plot_distribution(samples, bins=30)

# Compare distributions
viz.plot_distribution_comparison([samples1, samples2], labels=["Method A", "Method B"])

# Q-Q plot
viz.plot_qq_plot(samples, theoretical_dist='norm')

# Convergence diagnostics
viz.plot_convergence_diagnostics(multi_chain_samples)

# R-hat
viz.plot_rhat(rhat_values, parameter_names=param_names, threshold=1.1)

# Effective sample size
viz.plot_effective_sample_size(ess_values, n_samples=1000)
```

## Environment Plots

```python
# Plot grid world
viz.plot_grid_world(
    grid_shape=(10, 10),
    agent_pos=(5, 5),
    goal_pos=(9, 9),
    obstacles=[(3, 3), (4, 4)]
)

# Plot T-maze
viz.plot_tmaze(agent_pos=1, cue="left", reward_left=True)

# Plot agent trajectory
viz.plot_agent_trajectory(positions, grid_shape=(10, 10))

# Plot observation history
viz.plot_observation_history(observations, obs_labels=labels)

# Plot rewards
viz.plot_reward_over_time(rewards, cumulative=True)

# Plot occupancy heatmap
viz.plot_heatmap_occupancy(occupancy_matrix, obstacles=obstacles)
```

## Comparison Plots

```python
# Multi-agent comparison
viz.plot_multi_agent_comparison(
    metrics_by_agent={"Agent A": rewards_a, "Agent B": rewards_b},
    metric_name="Cumulative Reward"
)

# Learning curves
viz.plot_learning_curves(
    curves={"Method 1": returns1, "Method 2": returns2},
    smooth_window=10
)

# Parameter sweep
viz.plot_parameter_sweep(param_values, metric_values, param_name="Learning Rate")

# Ablation study
viz.plot_ablation_study(conditions, metrics, baseline_idx=0)

# Hyperparameter heatmap
viz.plot_hyperparameter_heatmap(lr_values, gamma_values, performance_matrix)

# Multi-metric comparison
viz.plot_multi_metric_comparison(agents, {"Reward": rewards, "Steps": steps})

# Statistical comparison
viz.plot_statistical_comparison({"Group A": samples_a, "Group B": samples_b})
```

## Animations

```python
# Belief animation
anim = viz.create_belief_animation(
    beliefs=belief_history,
    true_states=true_states,
    interval=100,
    save_path="belief_animation.gif"
)

# Trajectory animation
anim = viz.create_trajectory_animation(
    positions=agent_positions,
    grid_shape=(10, 10),
    obstacles=obstacles,
    save_path="trajectory.gif"
)

# Sampling animation
anim = viz.create_sampling_animation(
    samples=sample_history,
    dims=(0, 1),
    save_path="sampling.mp4"
)
```

## Customization

All plotting functions support extensive customization through keyword arguments:

```python
viz.plot_belief_trajectory(
    beliefs=beliefs,
    figsize=(14, 8),
    cmap='plasma',
    show_entropy=True,
    save_path="custom_beliefs.png",
    dpi=300
)
```

## Dependencies

Core dependencies:
- `matplotlib` - Core plotting (required)
- `jax` - Array operations (required)

Optional dependencies:
- `seaborn` - Enhanced statistical visualizations
- `networkx` - Network graph visualizations
- `scipy` - Advanced statistical functions
- `pandas` - Data manipulation for complex plots
- `pillow` - GIF export for animations
- `ffmpeg` - Video export for animations

Install all visualization dependencies:
```bash
uv pip install active_inference[viz]
```

## Best Practices

1. **Output Management**: Always set an output directory for your experiment
   ```python
   viz.set_output_dir(f"experiments/{experiment_name}/output")
   ```

2. **Batch Plotting**: Use `save_path` parameter to automatically save plots
   ```python
   for i, belief in enumerate(belief_checkpoints):
       viz.plot_belief_trajectory(belief, save_path=f"belief_step_{i}.png")
   ```

3. **Configuration**: Set visualization config once at the start
   ```python
   config = viz.get_config()
   config.dpi = 300
   config.figure_format = "pdf"
   ```

4. **Close Figures**: When generating many plots, close them to save memory
   ```python
   fig, ax = viz.plot_free_energy(fe_values, save_path="fe.png")
   plt.close(fig)
   ```

## Examples

See `examples/` directory for complete examples:
- `visualization_demo.py` - Comprehensive visualization demonstration
- `thrml_viz_demo.py` - THRML-specific visualizations
- `animation_demo.py` - Creating animations

## Contributing

To add new visualization functions:

1. Choose appropriate module (e.g., `active_inference_plots.py`)
2. Follow existing function signature patterns
3. Use configuration from `core.py`
4. Add comprehensive docstring with arguments
5. Support `save_path` parameter
6. Export from `__init__.py`

## License

Part of the active_inference package.
