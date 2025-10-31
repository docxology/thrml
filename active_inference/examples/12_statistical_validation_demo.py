"""Example 12: Statistical Analysis and Validation Demo

This example demonstrates the comprehensive analysis, validation, and
resource tracking features added to the active inference framework.

**Features Demonstrated:**

1. Statistical Analysis
   - Linear regression with diagnostics
   - Correlation analysis
   - Effect size computation
   - Statistical testing
   - Summary statistics

2. Data Validation
   - Model validation
   - Distribution validation
   - Trajectory validation
   - HTML report generation

3. Resource Tracking
   - CPU and memory monitoring
   - Section profiling
   - Resource estimation
   - Comprehensive reporting

4. Enhanced Visualization
   - Scatter plots with regression lines
   - Correlation matrices
   - Residual diagnostics
   - Pairwise relationships

**Real Implementations:**

All methods use real calculations (no mocks):
- Scipy for statistical tests
- JAX for numerical operations
- Psutil for system resource monitoring

**THRML Integration**:
- Uses THRML sampling-based inference (`ThrmlInferenceEngine`) for agent simulation
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- GPU-accelerated block Gibbs sampling for efficient inference
- Statistical analysis validates THRML inference results
- Comprehensive THRML usage in validation pipelines
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # For example_utils

# Import example utilities
from example_utils import ExampleRunner

# Import enhanced visualization
from active_inference import visualization as viz
from active_inference.agents import ActiveInferenceAgent
from active_inference.core import GenerativeModel
from active_inference.inference import ThrmlInferenceEngine

# Import new analysis and validation utilities
from active_inference.utils import (  # Statistical analysis; Validation; Resource tracking
    DataValidator,
    ResourceTracker,
    compute_effect_size,
    compute_summary_statistics,
    estimate_resources,
    generate_statistical_report,
    linear_regression,
    pearson_correlation,
    print_resource_estimates,
    t_test_independent,
)


def create_test_model(n_states=4, n_observations=4, n_actions=4, goal_state=3, goal_preference_strength=2.0):
    """Create a simple generative model for testing."""
    # Observation model (identity)
    A = jnp.eye(n_observations, n_states)

    # Transition model (random walk with self-loops)
    B = jnp.zeros((n_actions, n_states, n_states))
    for a in range(n_actions):
        for s in range(n_states):
            B = B.at[a, s, s].set(0.7)
            next_s = (s + a) % n_states
            B = B.at[a, s, next_s].set(0.3)

    # Prior (uniform)
    D = jnp.ones(n_states) / n_states

    # Preferences (prefer goal_state using config)
    C = jnp.ones(n_states) * -1.0
    C = C.at[goal_state].set(goal_preference_strength)

    return GenerativeModel(A=A, B=B, C=C, D=D, n_states=n_states, n_observations=n_observations, n_actions=n_actions)


def main():
    """Run the statistical analysis and validation demo."""
    # Initialize example runner
    output_base = Path(__file__).parent.parent / "output"
    runner = ExampleRunner(example_name="12_statistical_validation_demo", output_base=output_base)

    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    n_states = runner.get_config("n_states", default=4)
    n_observations = runner.get_config("n_observations", default=4)
    n_actions = runner.get_config("n_actions", default=4)
    n_steps = runner.get_config("n_steps", default=500)
    planning_horizon = runner.get_config("planning_horizon", default=1)
    n_samples_estimate = runner.get_config("n_samples_estimate", default=1000)
    goal_state = runner.get_config("goal_state", default=3)
    goal_preference_strength = runner.get_config("goal_preference_strength", default=2.0)
    # THRML sampling parameters
    n_samples = runner.get_config("n_samples", default=200)
    n_warmup = runner.get_config("n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)

    # Set random seed from config
    key = jax.random.key(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  n_states: {n_states}")
    runner.logger.info(f"  n_observations: {n_observations}")
    runner.logger.info(f"  n_actions: {n_actions}")
    runner.logger.info(f"  n_steps: {n_steps}")
    runner.logger.info(f"  planning_horizon: {planning_horizon}")
    runner.logger.info(f"  n_samples_estimate: {n_samples_estimate}")
    runner.logger.info(f"  goal_state: {goal_state}")
    runner.logger.info(f"  goal_preference_strength: {goal_preference_strength}")

    # Save configuration
    config = {
        "seed": seed,
        "n_states": n_states,
        "n_observations": n_observations,
        "n_actions": n_actions,
        "n_steps": n_steps,
        "planning_horizon": planning_horizon,
        "n_samples_estimate": n_samples_estimate,
        "goal_state": goal_state,
        "goal_preference_strength": goal_preference_strength,
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "steps_per_sample": steps_per_sample,
    }
    runner.save_config(config)

    print("\n" + "=" * 70)
    print("Example 12: Statistical Analysis and Validation Demo")
    print("=" * 70 + "\n")

    # ========================================================================
    # Part 1: Resource Estimation
    # ========================================================================
    with runner.section("Resource Estimation"):
        print("Estimating resource requirements...")
        estimates = estimate_resources(
            n_states=n_states,
            n_observations=n_observations,
            n_actions=n_actions,
            n_steps=n_steps,
            n_samples=n_samples_estimate,
        )
        print_resource_estimates(estimates)
        runner.save_data(estimates, "resource_estimates.json")

    # ========================================================================
    # Part 2: Model Creation and Validation
    # ========================================================================
    with runner.section("Model Validation"):
        print("Creating and validating generative model...")
        model = create_test_model(
            n_states=n_states,
            n_observations=n_observations,
            n_actions=n_actions,
            goal_state=goal_state,
            goal_preference_strength=goal_preference_strength,
        )

        # Create THRML inference engine
        thrml_engine = ThrmlInferenceEngine(
            model=model,
            n_samples=n_samples,
            n_warmup=n_warmup,
            steps_per_sample=steps_per_sample,
        )

        # Initialize validator
        validator = DataValidator()

        # Validate model
        validation_results = validator.validate_generative_model(model)

        # Print validation report
        print("\nValidation Results:")
        validator.print_report()

        # Generate HTML validation report
        html_path = runner.get_report_path("validation_report.html")
        validator.generate_html_report(html_path, title="Model Validation Report")
        print(f"\n✓ HTML validation report saved to {html_path}")

    # ========================================================================
    # Part 3: Resource Tracking During Inference
    # ========================================================================
    tracker = ResourceTracker()
    tracker.start()

    with runner.section("Active Inference Simulation"):
        print("Running active inference simulation with resource tracking...")

        key = jax.random.PRNGKey(0)
        n_steps = 50

        # Initialize agent with THRML engine
        agent = ActiveInferenceAgent(model=model, planning_horizon=1, thrml_engine=thrml_engine)

        # Run simulation
        beliefs = []
        actions = []
        observations = []
        free_energies = []
        entropies = []

        belief = model.D

        snapshot_start = tracker.take_snapshot("inference_start")

        for t in range(n_steps):
            key, subkey = jax.random.split(key)

            # Infer state using THRML sampling
            observation_idx = t % model.n_observations
            model_with_prior = eqx.tree_at(lambda m: m.D, model, belief)
            temp_engine = eqx.tree_at(lambda e: e.model, thrml_engine, model_with_prior)
            belief = temp_engine.infer_with_sampling(
                key=subkey,
                observation=observation_idx,
                n_state_samples=n_samples,
            )
            beliefs.append(belief)

            # Select action
            key, subkey = jax.random.split(key)
            action = agent.act(subkey, belief)
            actions.append(action)

            # Generate observation
            observation = (t + 1) % model.n_observations
            observations.append(observation)

            # Calculate metrics
            from active_inference.core import variational_free_energy

            fe = variational_free_energy(observation, belief, model)
            free_energies.append(float(fe))

            entropy = -float(jnp.sum(belief * jnp.log(belief + 1e-16)))
            entropies.append(entropy)

        tracker.profile_section("inference", snapshot_start)

        # Save trajectory data
        trajectory_data = {
            "beliefs": [belief.tolist() for belief in beliefs],
            "actions": [int(a) for a in actions],
            "observations": observations,
            "free_energies": free_energies,
            "entropies": entropies,
        }
        runner.save_data(trajectory_data, "trajectory.json")
        print(f"✓ Simulated {n_steps} time steps")

    # ========================================================================
    # Part 4: Statistical Analysis
    # ========================================================================
    with runner.section("Statistical Analysis"):
        print("\nPerforming statistical analysis...")

        # Convert to arrays for analysis
        timesteps = jnp.arange(len(free_energies))
        fe_array = jnp.array(free_energies)
        ent_array = jnp.array(entropies)

        # 1. Linear regression: Free Energy over time
        print("\n1. Regression Analysis: Free Energy vs Time")
        print("-" * 50)
        fe_regression = linear_regression(timesteps, fe_array)
        print(fe_regression)
        runner.save_data(
            {
                "slope": float(fe_regression.slope),
                "intercept": float(fe_regression.intercept),
                "r_squared": float(fe_regression.r_squared),
                "p_value": float(fe_regression.p_value),
            },
            "fe_regression.json",
        )

        # 2. Correlation: Free Energy vs Entropy
        print("\n2. Correlation Analysis: Free Energy vs Entropy")
        print("-" * 50)
        fe_ent_corr = pearson_correlation(fe_array, ent_array)
        print(fe_ent_corr)

        # 3. Summary statistics
        print("\n3. Summary Statistics")
        print("-" * 50)
        fe_stats = compute_summary_statistics(fe_array)
        print("Free Energy Statistics:")
        print(f"  Mean: {fe_stats['mean']:.4f} ± {fe_stats['std']:.4f}")
        print(f"  Median: {fe_stats['median']:.4f}")
        print(f"  Range: [{fe_stats['min']:.4f}, {fe_stats['max']:.4f}]")
        print(f"  Skewness: {fe_stats['skewness']:.4f}")
        print(f"  Kurtosis: {fe_stats['kurtosis']:.4f}")

        ent_stats = compute_summary_statistics(ent_array)
        print("\nEntropy Statistics:")
        print(f"  Mean: {ent_stats['mean']:.4f} ± {ent_stats['std']:.4f}")
        print(f"  Median: {ent_stats['median']:.4f}")
        print(f"  Range: [{ent_stats['min']:.4f}, {ent_stats['max']:.4f}]")

        # 4. Compare first half vs second half
        print("\n4. Comparative Analysis: First Half vs Second Half")
        print("-" * 50)
        mid = len(free_energies) // 2
        fe_first_half = fe_array[:mid]
        fe_second_half = fe_array[mid:]

        # T-test
        t_results = t_test_independent(fe_first_half, fe_second_half)
        print(f"t-test: t={t_results['t_statistic']:.4f}, p={t_results['p_value']:.4f}")
        if t_results["significant_05"]:
            print("✓ Significant difference at α=0.05")
        else:
            print("✗ No significant difference at α=0.05")

        # Effect size
        effect = compute_effect_size(fe_first_half, fe_second_half)
        print(f"Effect size (Cohen's d): {effect['cohens_d']:.4f}")
        print(f"Mean difference: {effect['mean_diff']:.4f}")

        # 5. Generate comprehensive statistical report
        print("\n5. Generating Comprehensive Statistical Report")
        print("-" * 50)
        report_data = {
            "first_half": fe_first_half,
            "second_half": fe_second_half,
        }
        stat_report = generate_statistical_report(report_data, compare_groups=True)
        runner.save_report(stat_report, "statistical_analysis.txt")
        print("✓ Statistical report saved")

    # ========================================================================
    # Part 5: Enhanced Visualization
    # ========================================================================
    with runner.section("Enhanced Visualization"):
        print("\nGenerating enhanced visualizations...")

        # 1. Scatter with regression: FE over time
        print("  • Free Energy regression plot...")
        viz.plot_scatter_with_regression(
            timesteps,
            fe_array,
            x_label="Time Step",
            y_label="Free Energy",
            title="Free Energy Over Time (with Regression)",
            save_path=str(runner.get_plot_path("fe_regression.png")),
        )

        # 2. Scatter with regression: FE vs Entropy
        print("  • FE vs Entropy regression plot...")
        viz.plot_scatter_with_regression(
            fe_array,
            ent_array,
            x_label="Free Energy",
            y_label="Entropy",
            title="Free Energy vs Entropy Relationship",
            save_path=str(runner.get_plot_path("fe_entropy_regression.png")),
        )

        # 3. Correlation matrix
        print("  • Correlation matrix...")
        analysis_data = {
            "Free Energy": fe_array,
            "Entropy": ent_array,
            "Time": timesteps,
        }
        viz.plot_correlation_matrix(analysis_data, save_path=str(runner.get_plot_path("correlation_matrix.png")))

        # 4. Pairwise relationships
        print("  • Pairwise relationships...")
        viz.plot_pairwise_relationships(
            analysis_data, save_path=str(runner.get_plot_path("pairwise_relationships.png"))
        )

        # 5. Residual diagnostics for FE regression
        print("  • Residual diagnostics...")
        viz.plot_residuals(timesteps, fe_array, save_path=str(runner.get_plot_path("fe_residuals.png")))

        print("✓ All visualizations generated")

    # ========================================================================
    # Part 6: Trajectory Validation
    # ========================================================================
    with runner.section("Trajectory Validation"):
        print("\nValidating trajectory data...")

        # Clear previous validation results
        validator.clear_results()

        # Validate trajectory
        traj_results = validator.validate_trajectory(beliefs=beliefs, actions=actions, observations=observations)

        print("\nTrajectory Validation Results:")
        validator.print_report()

        # Save validation report
        html_path2 = runner.get_report_path("trajectory_validation.html")
        validator.generate_html_report(html_path2, title="Trajectory Validation Report")

    # ========================================================================
    # Part 7: Resource Report
    # ========================================================================
    with runner.section("Resource Report"):
        print("\nGenerating resource usage report...")

        tracker.stop()

        # Get resource statistics
        peak_memory = tracker.get_peak_memory()
        avg_cpu = tracker.get_avg_cpu()

        print("\nResource Usage Summary:")
        print(f"  Peak Memory: {peak_memory:.1f} MB")
        print(f"  Average CPU: {avg_cpu:.1f}%")

        # Save comprehensive report
        resource_report = tracker.generate_report()
        runner.save_report(resource_report, "resource_usage.txt")
        print("✓ Resource report saved")

    # ========================================================================
    # Finalize
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Model validated with {len(validation_results)} checks")
    print(f"✓ Simulated {n_steps} time steps")
    print("✓ Performed 4 statistical analyses")
    print("✓ Generated 5 enhanced visualizations")
    print(f"✓ Validated trajectory with {len(traj_results)} checks")
    print(f"✓ Tracked resources: {peak_memory:.1f}MB peak, {avg_cpu:.1f}% CPU")
    print(f"\nAll outputs saved to: {runner.output_dir}")
    print("=" * 70 + "\n")

    runner.end()


if __name__ == "__main__":
    main()
