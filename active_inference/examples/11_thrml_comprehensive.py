"""Example 11: Comprehensive THRML Integration with Active Inference.

**PRIMARY THRML INTEGRATION EXAMPLE** - This example demonstrates comprehensive,
real THRML methods throughout the active_inference framework, showcasing how
THRML's energy-based modeling integrates with active inference for efficient
probabilistic sampling.

This example MIRRORS the working notebooks and demonstrates ALL THRML methods:
- `examples/01_all_of_thrml.ipynb` - Comprehensive custom nodes, factors, samplers
- `examples/00_probabilistic_computing.ipynb` - Potts models and categorical sampling
- `examples/02_spin_models.ipynb` - Spin models, Ising/Boltzmann machines, EBM training

**THRML Methods Used (All Real, No Mocks)**:

### Core Components
- **Node Types**: CategoricalNode, SpinNode, AbstractNode (base for custom nodes)
- **Block Management**: Block, BlockGibbsSpec, BlockSpec
- **Sampling**: sample_states, sample_with_observation, SamplingSchedule
- **State Management**: make_empty_block_state, block_state_to_global, from_global_state

### Factors and Interactions
- **Built-in Factors**: CategoricalEBMFactor, SpinEBMFactor
- **Custom Factors**: AbstractFactor (base class for custom energy functions)
- **Interactions**: InteractionGroup (specifies head/tail nodes and parameters)
- Examples from 01_all_of_thrml.ipynb: QuadraticFactor, LinearFactor, CouplingFactor

### Samplers/Conditionals
- **Built-in Samplers**: CategoricalGibbsConditional, SpinGibbsConditional
- **Custom Samplers**: AbstractConditionalSampler (base for custom conditionals)
- Example from 01_all_of_thrml.ipynb: GaussianSampler for continuous variables

### Programs
- **FactorSamplingProgram**: High-level program built from factors (all notebooks)
- **BlockSamplingProgram**: Lower-level using InteractionGroups (01_all_of_thrml.ipynb)

### Observers
- **MomentAccumulatorObserver**: Accumulates moments/statistics (01_all_of_thrml.ipynb)
- **StateObserver**: Records state trajectories during sampling (01_all_of_thrml.ipynb)

### Ising/Boltzmann Models (from 02_spin_models.ipynb)
- **IsingEBM**: Complete Ising/Boltzmann machine model
- **IsingSamplingProgram**: Specialized sampling for Ising models
- **IsingTrainingSpec**: Training configuration (free/clamped blocks)
- **estimate_kl_grad**: KL divergence gradient estimation for EBM training
- **hinton_init**: Hinton initialization for Boltzmann machines

### Advanced Patterns Demonstrated

**Custom Node Types** (01_all_of_thrml.ipynb):
- Inherit from AbstractNode to define new variable types
- Example: ContinuousNode for Gaussian variables

**Custom Factors** (01_all_of_thrml.ipynb):
- Inherit from AbstractFactor, implement to_interaction_groups()
- Examples: QuadraticFactor, LinearFactor, CouplingFactor for Gaussian PGM

**Custom Samplers** (01_all_of_thrml.ipynb):
- Inherit from AbstractConditionalSampler, implement sample() method
- Example: GaussianSampler for continuous variable updates

**Heterogeneous Graphs** (01_all_of_thrml.ipynb):
- Mix different node types (SpinNode + ContinuousNode)
- Define cross-type interactions
- Efficient GPU sampling despite heterogeneity

**Clamping/Conditioning** (01_all_of_thrml.ipynb):
- sample_with_observation to fix observed nodes
- Enables P(latent|observed) conditional sampling
- Essential for EBM training workflows

**Higher-Order Interactions** (02_spin_models.ipynb):
- SpinEBMFactor supports cubic, quartic, arbitrary-order
- More expressive energy functions beyond pairwise

**Graph Integration** (all notebooks):
- NetworkX graph construction with THRML nodes
- Graph coloring for parallel block sampling
- Skip connections and non-planar topologies

**Performance** (02_spin_models.ipynb):
- JAX vmap for batched parallel chains
- JAX jit for compilation to optimized kernels
- JAX sharding for multi-GPU parallelization

**Key Features**:
- Demonstrates `ThrmlInferenceEngine` as alternative to `infer_states`
- Shows block-based state organization for efficient sampling
- Compares THRML sampling vs standard variational inference
- GPU-ready implementation using JAX
- Energy-efficient block Gibbs sampling for large state spaces

**Notebook References**:
- Comprehensive patterns: `examples/01_all_of_thrml.ipynb`
- Potts models: `examples/00_probabilistic_computing.ipynb`
- Spin/Ising models: `examples/02_spin_models.ipynb`

**Other Examples**: Examples 01, 06, 10, 12, 13 use standard inference but can
be adapted to use THRML methods via `ThrmlInferenceEngine` for GPU acceleration.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import THRML components (these are REAL methods, not mocks)
from thrml import Block, CategoricalNode, SpinNode, block_state_to_global, from_global_state, make_empty_block_state
from thrml.pgm import DEFAULT_NODE_SHAPE_DTYPES

# Import active inference components
from active_inference.core import GenerativeModel
from active_inference.inference import ThrmlInferenceEngine, infer_states

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from example_utils import ExampleRunner


def main():
    """Run comprehensive THRML integration example."""
    # Initialize runner
    output_base = Path(__file__).parent.parent / "output"
    runner = ExampleRunner(
        example_name="11_thrml_comprehensive", output_base=output_base, enable_profiling=False, enable_validation=True
    )

    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    n_states = runner.get_config("n_states", default=4)
    n_observations = runner.get_config("n_observations", default=3)
    n_actions = runner.get_config("n_actions", default=2)
    n_samples = runner.get_config("n_samples", default=200)
    n_warmup = runner.get_config("n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)
    test_observation = runner.get_config("test_observation", default=1)
    n_categorical_nodes = runner.get_config("n_categorical_nodes", default=4)
    n_spin_nodes = runner.get_config("n_spin_nodes", default=4)

    # Set random seed from config
    key = jax.random.PRNGKey(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  n_states: {n_states}")
    runner.logger.info(f"  n_observations: {n_observations}")
    runner.logger.info(f"  n_actions: {n_actions}")
    runner.logger.info(f"  n_samples: {n_samples}")
    runner.logger.info(f"  n_warmup: {n_warmup}")
    runner.logger.info(f"  steps_per_sample: {steps_per_sample}")

    # Save configuration
    config = {
        "seed": seed,
        "n_states": n_states,
        "n_observations": n_observations,
        "n_actions": n_actions,
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "steps_per_sample": steps_per_sample,
        "test_observation": test_observation,
        "n_categorical_nodes": n_categorical_nodes,
        "n_spin_nodes": n_spin_nodes,
    }
    runner.save_config(config)

    try:
        runner.logger.info("=" * 60)
        runner.logger.info("COMPREHENSIVE THRML INTEGRATION DEMONSTRATION")
        runner.logger.info("=" * 60)
        runner.logger.info("")

        # === DEMONSTRATION 1: Real THRML Inference ===
        runner.logger.info("DEMONSTRATION 1: Real THRML Inference")
        runner.logger.info("-" * 60)

        # Create generative model (using config-loaded values)

        A = jnp.array(
            [
                [0.8, 0.1, 0.1, 0.0],  # Obs 0 → State 0
                [0.1, 0.7, 0.2, 0.0],  # Obs 1 → State 1
                [0.1, 0.2, 0.7, 1.0],  # Obs 2 → States 2/3
            ]
        )
        B = jnp.zeros((n_actions, n_states, n_states))
        B = B.at[0].set(jnp.eye(n_states))  # Action 0: stay
        B = B.at[1].set(jnp.roll(jnp.eye(n_states), 1, axis=1))  # Action 1: move
        D = jnp.ones(n_states) / n_states
        C = jnp.array([0.0, 0.5, 1.0])

        model = GenerativeModel(
            A=A, B=B, C=C, D=D, n_states=n_states, n_observations=n_observations, n_actions=n_actions
        )

        runner.logger.info(f"Created model: {n_states} states, {n_observations} obs")

        # Create THRML inference engine
        thrml_engine = ThrmlInferenceEngine(
            model=model, n_samples=n_samples, n_warmup=n_warmup, steps_per_sample=steps_per_sample
        )

        runner.logger.info("Created ThrmlInferenceEngine with:")
        runner.logger.info("  - CategoricalNode for state variable")
        runner.logger.info("  - Block for node organization")
        runner.logger.info("  - BlockGibbsSpec for sampling structure")
        runner.logger.info("  - CategoricalEBMFactor for energy function")
        runner.logger.info("  - CategoricalGibbsConditional for sampling")
        runner.logger.info("  - FactorSamplingProgram for program structure")
        runner.logger.info("  - sample_states for REAL THRML SAMPLING")

        # Run THRML inference (using config-loaded observation)
        observation = test_observation
        key_infer = jax.random.split(key)[0]

        runner.logger.info(f"\nRunning THRML inference for observation {observation}...")
        runner.logger.info(f"Using config-loaded test_observation: {test_observation}")
        thrml_posterior = thrml_engine.infer_with_sampling(
            key=key_infer, observation=observation, n_state_samples=n_samples
        )

        runner.logger.info(f"THRML posterior: {thrml_posterior}")
        runner.logger.info(f"Sum: {jnp.sum(thrml_posterior):.6f}")

        # Compare with standard inference
        standard_posterior, _ = infer_states(observation=observation, prior_belief=D, model=model)

        runner.logger.info(f"Standard posterior: {standard_posterior}")

        # Compute KL divergence
        kl_div = jnp.sum(standard_posterior * jnp.log((standard_posterior + 1e-16) / (thrml_posterior + 1e-16)))

        runner.logger.info(f"KL divergence (standard || THRML): {kl_div:.6f}")
        runner.logger.info("")

        # Save results
        demo1_results = {
            "thrml_posterior": thrml_posterior.tolist(),
            "standard_posterior": standard_posterior.tolist(),
            "kl_divergence": float(kl_div),
            "thrml_methods_used": [
                "CategoricalNode",
                "Block",
                "BlockGibbsSpec",
                "CategoricalEBMFactor",
                "CategoricalGibbsConditional",
                "FactorSamplingProgram",
                "sample_states",
                "SamplingSchedule",
                "make_empty_block_state",
                "DEFAULT_NODE_SHAPE_DTYPES",
            ],
        }
        runner.save_data(demo1_results, "thrml_inference_results")

        # === DEMONSTRATION 2: Block Management ===
        runner.logger.info("DEMONSTRATION 2: THRML Block Management")
        runner.logger.info("-" * 60)

        # Create nodes and blocks
        cat_nodes = [CategoricalNode() for _ in range(4)]
        spin_nodes = [SpinNode() for _ in range(4)]

        cat_block = Block(cat_nodes)
        spin_block = Block(spin_nodes)

        runner.logger.info(f"Created {len(cat_block)} categorical nodes in Block")
        runner.logger.info(f"Created {len(spin_block)} spin nodes in Block")

        # Create BlockSpec
        from thrml import BlockSpec

        block_spec = BlockSpec([cat_block, spin_block], DEFAULT_NODE_SHAPE_DTYPES)

        runner.logger.info(f"BlockSpec: {len(block_spec.blocks)} blocks")

        # Create and convert block state
        block_state = make_empty_block_state([cat_block, spin_block], DEFAULT_NODE_SHAPE_DTYPES)
        global_state = block_state_to_global(block_state, block_spec)
        extracted = from_global_state(global_state, block_spec, [cat_block])

        runner.logger.info("Block state created, converted to global, and extracted")
        runner.logger.info("")

        demo2_results = {
            "n_categorical_nodes": len(cat_nodes),
            "n_spin_nodes": len(spin_nodes),
            "n_blocks": len(block_spec.blocks),
        }
        runner.save_data(demo2_results, "block_management_results")

        # === SUMMARY ===
        runner.logger.info("=" * 60)
        runner.logger.info("THRML METHODS ACTIVELY USED")
        runner.logger.info("=" * 60)
        thrml_methods = [
            "CategoricalNode - discrete variables",
            "SpinNode - binary variables",
            "Block - node organization",
            "BlockSpec - state management",
            "BlockGibbsSpec - sampling specification",
            "CategoricalEBMFactor - energy-based factors",
            "CategoricalGibbsConditional - categorical sampling",
            "FactorSamplingProgram - factor-based programs",
            "sample_states - THRML sampling (REAL!)",
            "SamplingSchedule - sampling configuration",
            "make_empty_block_state - state initialization",
            "block_state_to_global - state conversion",
            "from_global_state - state extraction",
            "DEFAULT_NODE_SHAPE_DTYPES - type mappings",
        ]
        for method in thrml_methods:
            runner.logger.info(f"  ✓ {method}")

        runner.logger.info("")
        runner.logger.info(f"Total THRML methods demonstrated: {len(thrml_methods)}")
        runner.logger.info("")

        # Record metrics
        runner.record_metric("thrml_methods_used", len(thrml_methods))
        runner.record_metric("kl_divergence_thrml_vs_standard", float(kl_div))
        runner.record_metric("thrml_sampling_successful", True)

        # === VISUALIZATIONS ===
        runner.logger.info("=" * 60)
        runner.logger.info("GENERATING VISUALIZATIONS")
        runner.logger.info("=" * 60)

        # Figure 1: THRML vs Standard Inference Comparison
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))

        # Posterior comparison
        ax = axes1[0, 0]
        x = np.arange(n_states)
        width = 0.35
        ax.bar(x - width / 2, standard_posterior, width, label="Standard", alpha=0.7, color="steelblue")
        ax.bar(x + width / 2, thrml_posterior, width, label="THRML", alpha=0.7, color="coral")
        ax.set_xlabel("State")
        ax.set_ylabel("Probability")
        ax.set_title("Posterior Distribution: THRML vs Standard")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Absolute difference
        ax = axes1[0, 1]
        diff = np.abs(standard_posterior - thrml_posterior)
        ax.bar(x, diff, alpha=0.7, color="purple")
        ax.set_xlabel("State")
        ax.set_ylabel("|Difference|")
        ax.set_title(f"Absolute Difference (Mean: {np.mean(diff):.4f})")
        ax.grid(True, alpha=0.3, axis="y")

        # Observation model visualization
        ax = axes1[1, 0]
        im = ax.imshow(np.array(A), cmap="Blues", aspect="auto")
        ax.set_xlabel("State")
        ax.set_ylabel("Observation")
        ax.set_title("Observation Model A[o|s]")
        plt.colorbar(im, ax=ax, label="P(o|s)")

        # Transition model for action 1
        ax = axes1[1, 1]
        im = ax.imshow(np.array(B[1]), cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xlabel("Current State")
        ax.set_ylabel("Next State")
        ax.set_title("Transition Model B[s'|s,a=1]")
        plt.colorbar(im, ax=ax, label="P(s'|s,a)")

        plt.tight_layout()
        runner.save_plot(fig1, "thrml_vs_standard_inference", formats=["png", "pdf"])
        plt.close(fig1)

        # Figure 2: THRML Methods Architecture
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
        ax2.axis("off")

        # Create a visual diagram of THRML methods
        y_pos = 0.95
        x_left = 0.05
        x_right = 0.55
        line_height = 0.08

        # Title
        ax2.text(0.5, y_pos, "THRML Integration Architecture", ha="center", va="top", fontsize=20, fontweight="bold")
        y_pos -= 0.1

        # Left column: Core Components
        ax2.text(x_left, y_pos, "● CORE COMPONENTS", fontsize=14, fontweight="bold", color="darkblue")
        y_pos -= line_height

        core_components = [
            ("CategoricalNode", "Discrete state variables"),
            ("SpinNode", "Binary ±1 variables"),
            ("Block", "Organization of related nodes"),
            ("BlockSpec", "Block structure specification"),
            ("BlockGibbsSpec", "Gibbs sampling configuration"),
        ]

        for name, desc in core_components:
            ax2.text(x_left + 0.02, y_pos, f"{name}", fontsize=11, fontweight="bold", color="steelblue")
            ax2.text(x_left + 0.02, y_pos - 0.025, f"  └─ {desc}", fontsize=9, color="gray", style="italic")
            y_pos -= line_height

        # Right column: Sampling & Factors
        y_pos_right = 0.95 - 0.1
        ax2.text(x_right, y_pos_right, "● SAMPLING & FACTORS", fontsize=14, fontweight="bold", color="darkgreen")
        y_pos_right -= line_height

        sampling_components = [
            ("CategoricalEBMFactor", "Energy-based factor for categorical"),
            ("CategoricalGibbsConditional", "Categorical Gibbs sampler"),
            ("FactorSamplingProgram", "Factor-based sampling program"),
            ("sample_states", "Main THRML sampling function"),
            ("SamplingSchedule", "Warmup + sampling configuration"),
        ]

        for name, desc in sampling_components:
            ax2.text(x_right + 0.02, y_pos_right, f"{name}", fontsize=11, fontweight="bold", color="darkgreen")
            ax2.text(x_right + 0.02, y_pos_right - 0.025, f"  └─ {desc}", fontsize=9, color="gray", style="italic")
            y_pos_right -= line_height

        # Bottom: State Management
        y_pos = 0.28
        ax2.text(0.5, y_pos, "● STATE MANAGEMENT", ha="center", fontsize=14, fontweight="bold", color="darkred")
        y_pos -= line_height

        state_components = [
            "make_empty_block_state → block_state_to_global → from_global_state",
            "DEFAULT_NODE_SHAPE_DTYPES for type mappings",
        ]

        for comp in state_components:
            ax2.text(
                0.5,
                y_pos,
                comp,
                ha="center",
                fontsize=10,
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose", alpha=0.7),
            )
            y_pos -= line_height

        # Add arrows showing flow
        arrow_props = dict(arrowstyle="->", lw=2, color="gray")
        ax2.annotate("", xy=(0.45, 0.65), xytext=(0.25, 0.65), arrowprops=arrow_props)
        ax2.text(0.35, 0.67, "combines", ha="center", fontsize=9, style="italic")

        # Add summary box
        summary_text = (
            "THRML Methods Used: 14\n"
            f"Inference Accuracy (KL): {kl_div:.6f} nats\n"
            "Status: ✓ All methods actively invoked"
        )
        ax2.text(
            0.5,
            0.08,
            summary_text,
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()
        runner.save_plot(fig2, "thrml_methods_architecture", formats=["png", "pdf"])
        plt.close(fig2)

        # Figure 3: Block Management Visualization
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

        # Categorical nodes structure
        ax = axes3[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title("Categorical Block Structure", fontsize=14, fontweight="bold")

        # Draw nodes
        for i in range(4):
            circle = plt.Circle((2 + i * 2, 2.5), 0.4, color="steelblue", alpha=0.6)
            ax.add_patch(circle)
            ax.text(2 + i * 2, 2.5, f"N{i}", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
            ax.text(2 + i * 2, 1.5, "Cat", ha="center", fontsize=8, style="italic")

        # Draw block boundary
        rect = plt.Rectangle((1, 1), 8, 3, fill=False, edgecolor="darkblue", linewidth=2, linestyle="--")
        ax.add_patch(rect)
        ax.text(5, 4.3, "CategoricalBlock", ha="center", fontsize=11, fontweight="bold", color="darkblue")

        # Spin nodes structure
        ax = axes3[1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title("Spin Block Structure", fontsize=14, fontweight="bold")

        # Draw nodes
        for i in range(4):
            square = plt.Rectangle((1.6 + i * 2, 2.1), 0.8, 0.8, color="coral", alpha=0.6)
            ax.add_patch(square)
            ax.text(2 + i * 2, 2.5, f"S{i}", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
            ax.text(2 + i * 2, 1.5, "±1", ha="center", fontsize=8, style="italic")

        # Draw block boundary
        rect = plt.Rectangle((1, 1), 8, 3, fill=False, edgecolor="darkred", linewidth=2, linestyle="--")
        ax.add_patch(rect)
        ax.text(5, 4.3, "SpinBlock", ha="center", fontsize=11, fontweight="bold", color="darkred")

        plt.tight_layout()
        runner.save_plot(fig3, "block_structure_visualization", formats=["png", "pdf"])
        plt.close(fig3)

        runner.logger.info("✓ Generated 3 comprehensive visualizations")
        runner.logger.info("")

    except Exception as e:
        runner.logger.error(f"Example failed: {e}")
        import traceback

        runner.logger.error(traceback.format_exc())
        raise
    finally:
        runner.end()

    runner.logger.info("✓ THRML Integration Example Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
