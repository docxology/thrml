# Visualization: AGENTS.md

## Purpose

The `visualization` module does not have a separate AGENTS.md file because:

1. **Function-Based**: The module is primarily function-based, not class-based
2. **Comprehensive README**: [README.md](README.md) already provides extensive documentation
3. **Clear Signatures**: All functions have complete docstrings with usage examples
4. **Consistent Interface**: All plotting functions follow same patterns

## Documentation Structure

Instead of a separate AGENTS.md, the visualization module uses:

- **[README.md](README.md)**: Comprehensive module documentation with:
  - Overview of all plot types
  - Quick start guide
  - Configuration options
  - Usage examples for each function
  - Best practices
  - Dependencies

- **Function Docstrings**: Each plotting function has detailed docstring:
  ```python
  def plot_belief_trajectory(beliefs, true_states=None, figsize=None, save_path=None):
      """Plot belief evolution over time.

      Args:
          beliefs: Array of belief distributions [time_steps, n_states]
          true_states: Optional array of true state indices [time_steps]
          figsize: Figure size tuple (width, height)
          save_path: Optional path to save plot

      Returns:
          Tuple of (figure, axes)

      Example:
          >>> beliefs = jnp.array([[0.25, 0.25, 0.25, 0.25], ...])
          >>> fig, ax = plot_belief_trajectory(beliefs, save_path="beliefs.png")
      """
  ```

## Function Reference

### Active Inference Plots (`active_inference_plots.py`)

**Belief Visualization**:
- `plot_belief_trajectory(beliefs, true_states, save_path)`
- `plot_belief_evolution(beliefs, state_labels)`
- `plot_state_occupancy(state_visits, state_labels)`

**Free Energy**:
- `plot_free_energy(vfe_values, show_moving_average)`
- `plot_free_energy_components(vfe, accuracy, complexity)`
- `plot_expected_free_energy(efe_per_policy, selected_policy)`

**Agent Behavior**:
- `plot_action_distribution(action_probs, action_names)`
- `plot_policy_selection(policy_history, n_policies)`
- `plot_precision_trajectory(precisions, labels)`
- `plot_agent_performance(rewards, free_energies, entropies)`

### THRML Plots (`thrml_plots.py`)

**Sampling**:
- `plot_sampling_trajectory(samples, dims)`
- `plot_sample_statistics(samples)`
- `plot_energy_landscape(energy_fn, x_range, y_range)`

**Convergence**:
- `plot_mixing_diagnostics(samples_by_chain)`
- `plot_autocorrelation(samples, max_lag)`
- `plot_acceptance_rates(acceptance_history)`

**Model-Specific**:
- `plot_ising_spins(spin_configurations, timesteps)`
- `plot_categorical_states(categorical_samples, n_categories)`
- `plot_block_structure(blocks)`

### Network Plots (`network_plots.py`)

- `plot_graphical_model(nodes, edges, node_labels)`
- `plot_factor_graph(variable_nodes, factor_nodes, connections)`
- `plot_generative_model_structure(n_obs, n_states, n_actions)`
- `plot_markov_blanket(node_idx, nodes, edges)`

### Statistical Plots (`statistical_plots.py`)

- `plot_distribution(samples, bins)`
- `plot_distribution_comparison(sample_lists, labels)`
- `plot_qq_plot(samples, theoretical_dist)`
- `plot_convergence_diagnostics(multi_chain_samples)`
- `plot_rhat(rhat_values, parameter_names, threshold)`
- `plot_effective_sample_size(ess_values, n_samples)`

### Environment Plots (`environment_plots.py`)

- `plot_grid_world(grid_shape, agent_pos, goal_pos, obstacles)`
- `plot_tmaze(agent_pos, cue, reward_left)`
- `plot_agent_trajectory(positions, grid_shape)`
- `plot_observation_history(observations, obs_labels)`
- `plot_reward_over_time(rewards, cumulative)`
- `plot_heatmap_occupancy(occupancy_matrix, obstacles)`

### Comparison Plots (`comparison_plots.py`)

- `plot_multi_agent_comparison(metrics_by_agent, metric_name)`
- `plot_learning_curves(curves, smooth_window)`
- `plot_parameter_sweep(param_values, metric_values, param_name)`
- `plot_ablation_study(conditions, metrics, baseline_idx)`
- `plot_hyperparameter_heatmap(lr_values, gamma_values, performance_matrix)`
- `plot_multi_metric_comparison(agents, metrics_dict)`
- `plot_statistical_comparison(groups_dict)`

### Animation (`animation.py`)

- `create_belief_animation(beliefs, true_states, interval, save_path)`
- `create_trajectory_animation(positions, grid_shape, obstacles, save_path)`
- `create_sampling_animation(samples, dims, save_path)`

## Usage Patterns

### Basic Usage

```python
from active_inference import visualization as viz

# Plot belief trajectory
viz.plot_belief_trajectory(beliefs, true_states, save_path="beliefs.png")

# Plot free energy
viz.plot_free_energy(fe_values, show_moving_average=True)

# Plot grid world
viz.plot_grid_world((10, 10), agent_pos=(5,5), goal_pos=(9,9))
```

### Configuration

```python
# Set output directory
viz.set_output_dir("my_experiment/output")

# Access/modify config
config = viz.get_config()
config.dpi = 300
config.figure_format = "pdf"
```

### Batch Plotting

```python
# Plot multiple checkpoints
for i, belief in enumerate(belief_history):
    viz.plot_belief_trajectory(
        belief,
        save_path=f"checkpoint_{i}.png"
    )
    plt.close()  # Close to save memory
```

## Design Philosophy

1. **Consistent Interface**: All functions follow similar patterns
2. **Flexible Customization**: Support extensive keyword arguments
3. **Auto-Save**: Built-in save functionality with `save_path`
4. **Publication Quality**: High DPI, clean layouts
5. **Graceful Degradation**: Work without optional dependencies

## Common Parameters

Most plotting functions accept:

- `figsize`: Figure size tuple (width, height)
- `save_path`: Path to save plot (auto-creates directories)
- `dpi`: Resolution for saved images
- `cmap`: Colormap for heatmaps
- `title`: Custom plot title
- `xlabel`, `ylabel`: Custom axis labels

## Finding Functions

To find a visualization function:

1. **By Category**: See section headings above
2. **By Module**: Check individual module files
3. **By Example**: See `examples/` directory for usage
4. **By README**: Comprehensive guide in [README.md](README.md)

## Import Examples

```python
# Import all visualization
from active_inference import visualization as viz

# Import specific modules
from active_inference.visualization import active_inference_plots
from active_inference.visualization import thrml_plots

# Import specific functions
from active_inference.visualization.active_inference_plots import plot_belief_trajectory
from active_inference.visualization.environment_plots import plot_grid_world
```

## Integration with Active Inference

Visualization functions are designed to work seamlessly with active inference components:

```python
from active_inference import ActiveInferenceAgent, GridWorld
from active_inference import visualization as viz

# Run agent
agent = ActiveInferenceAgent(model)
env = GridWorld(size=10)

beliefs = []
free_energies = []

for step in range(100):
    action, state, fe = agent.step(key, obs, state)
    beliefs.append(state.belief)
    free_energies.append(fe)
    obs, reward, done = env.step(key, action)

# Visualize results
viz.plot_belief_trajectory(beliefs)
viz.plot_free_energy(free_energies)
viz.plot_agent_trajectory(env.trajectory, grid_shape=(10,10))
```

## Dependencies

**Required**:
- `matplotlib` - Core plotting

**Optional**:
- `seaborn` - Enhanced statistics
- `networkx` - Network graphs
- `pillow` - GIF export
- `ffmpeg` - Video export

Install all:
```bash
uv pip install active_inference[viz]
```

## Related Documentation

- **Main Documentation**: [../../docs/README.md](../../docs/README.md)
- **Module Overview**: [../../docs/module_visualization.md](../../docs/module_visualization.md)
- **Examples**: [../../examples/README.md](../../examples/README.md)
- **Utils Module**: [../utils/README.md](../utils/README.md)

## Future Additions

When adding new visualization functions:

1. Add to appropriate module file
2. Export from `__init__.py`
3. Update README.md with example
4. Add docstring with usage example
5. Follow existing patterns
