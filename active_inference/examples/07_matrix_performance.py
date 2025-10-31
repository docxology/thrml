"""Example 07: THRML Performance Benchmarking with Potts Model.

Demonstrates THRML performance characteristics using the exact Potts model
from 00_probabilistic_computing.ipynb with performance measurements added.

**THRML Methods Used** (exact from notebook 00):
- `CategoricalNode`, `Block` - Node types and block management
- `BlockGibbsSpec` - Block specification (exactly as in notebook)
- `CategoricalEBMFactor` - Energy-based factor
- `CategoricalGibbsConditional` - Conditional sampler
- `FactorSamplingProgram` - Sampling program (exactly as in notebook)
- `sample_states`, `SamplingSchedule` - Real THRML sampling

This benchmarks THRML's computational efficiency and scaling characteristics.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Import example utilities
from example_utils import ExampleRunner, create_figure

# THRML imports - exact from notebook 00
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode

OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "07_matrix_performance"


def run_potts_model_benchmark(side_length, n_cats, beta, n_batches, schedule, key):
    """Run Potts model exactly as in notebook 00, with timing.

    This is the EXACT code from 00_probabilistic_computing.ipynb with
    timing measurements added.
    """
    # === EXACT CODE FROM NOTEBOOK 00 STARTS HERE ===

    # Create grid graph
    G = nx.grid_graph(dim=(side_length, side_length), periodic=False)

    # Label nodes with CategoricalNode
    coord_to_node = {coord: CategoricalNode() for coord in G.nodes}
    nx.relabel_nodes(G, coord_to_node, copy=False)
    for coord, node in coord_to_node.items():
        G.nodes[node]["coords"] = coord

    # Graph coloring for parallel updates
    bicol = nx.bipartite.color(G)
    color0 = [n for n, c in bicol.items() if c == 0]
    color1 = [n for n, c in bicol.items() if c == 1]

    # Get edges
    u, v = map(list, zip(*G.edges()))

    # Create coupling interaction
    id_mat = jnp.eye(n_cats)
    weights = beta * jnp.broadcast_to(jnp.expand_dims(id_mat, 0), (len(u), *id_mat.shape))
    coupling_interaction = CategoricalEBMFactor([Block(u), Block(v)], weights)

    interactions = [coupling_interaction]

    # Build sampling program
    blocks = [Block(color0), Block(color1)]
    spec = BlockGibbsSpec(blocks, [])

    sampler = CategoricalGibbsConditional(n_cats)

    prog = FactorSamplingProgram(
        spec,
        [sampler for _ in blocks],
        interactions,
        [],
    )

    # === EXACT CODE FROM NOTEBOOK 00 ENDS HERE ===

    # Calculate number of nodes for initialization
    n_nodes = side_length * side_length

    # Now add timing around the sampling (using exact signature from notebook)
    start_time = time.time()

    # Initialize states (list of arrays, one per block - matching notebook signature)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    init_state = [
        jax.random.randint(subkey1, (n_batches, len(blocks[0].nodes)), minval=0, maxval=n_cats, dtype=jnp.uint8),
        jax.random.randint(subkey2, (n_batches, len(blocks[1].nodes)), minval=0, maxval=n_cats, dtype=jnp.uint8),
    ]

    # Create keys for batches
    keys = jax.random.split(key, n_batches)

    # Sample using exact signature from notebook 00
    samples = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], [Block(G.nodes)])))(
        init_state, keys
    )

    # Ensure computation completes
    jax.block_until_ready(samples)

    duration = time.time() - start_time

    # Calculate performance metrics
    n_nodes = side_length * side_length
    total_updates = n_batches * schedule.n_samples * schedule.steps_per_sample * n_nodes
    updates_per_second = total_updates / duration
    time_per_sample = duration / (n_batches * schedule.n_samples)

    return {
        "duration": duration,
        "updates_per_second": updates_per_second,
        "time_per_sample": time_per_sample,
        "total_updates": total_updates,
        "n_nodes": n_nodes,
        "side_length": side_length,
        "samples": samples,
    }


def main():
    """Run THRML performance benchmarks."""
    runner = ExampleRunner(EXAMPLE_NAME, OUTPUT_BASE)
    runner.start()

    # === LOAD CONFIGURATION ===
    seed = runner.get_config("seed", default=42)
    side_lengths = runner.get_config("side_lengths", default=[5, 10, 15, 20])
    batch_sizes = runner.get_config("batch_sizes", default=[1, 10, 50, 100])
    n_cats = runner.get_config("n_cats", default=5)
    beta = runner.get_config("beta", default=1.0)
    n_samples = runner.get_config("n_samples", default=100)
    n_warmup = runner.get_config("n_warmup", default=10)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)

    key = jax.random.key(seed)

    runner.logger.info("THRML Potts Model Performance Benchmarking")
    runner.logger.info("  Using exact code from 00_probabilistic_computing.ipynb")
    runner.logger.info(f"  Grid sizes: {side_lengths}")
    runner.logger.info(f"  Batch sizes: {batch_sizes}")
    runner.logger.info(f"  Categories: {n_cats}, Beta: {beta}")
    runner.logger.info(f"  JAX backend: {jax.default_backend()}")

    # === CONFIGURATION ===
    with runner.section("Configuration"):
        config = {
            "seed": seed,
            "side_lengths": side_lengths,
            "batch_sizes": batch_sizes,
            "n_cats": n_cats,
            "beta": beta,
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "steps_per_sample": steps_per_sample,
            "jax_backend": jax.default_backend(),
        }
        runner.save_config(config)

    # Create schedule (same for all benchmarks)
    schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=steps_per_sample)

    # === BENCHMARK 1: Grid Size Scaling ===
    with runner.section("Grid Size Scaling"):
        runner.logger.info("Benchmarking THRML vs grid size...")

        size_results = []

        for side_length in side_lengths:
            n_nodes = side_length * side_length
            runner.logger.info(f"\n  Testing {side_length}x{side_length} grid ({n_nodes} nodes)...")

            key, subkey = jax.random.split(key)

            # Warmup JIT
            _ = run_potts_model_benchmark(side_length, n_cats, beta, 1, schedule, subkey)

            # Actual benchmark
            key, subkey = jax.random.split(key)
            result = run_potts_model_benchmark(side_length, n_cats, beta, 10, schedule, subkey)
            size_results.append(result)

            runner.logger.info(f"    Updates/sec: {result['updates_per_second']/1e6:.2f}M")
            runner.logger.info(f"    Time/sample: {result['time_per_sample']*1000:.2f}ms")
            runner.logger.info(f"    Total time: {result['duration']:.2f}s")

        scaling_data = {
            "side_lengths": np.array(side_lengths),
            "node_counts": np.array([r["n_nodes"] for r in size_results]),
            "updates_per_sec": np.array([r["updates_per_second"] for r in size_results]),
            "time_per_sample": np.array([r["time_per_sample"] for r in size_results]),
        }
        runner.save_data(scaling_data, "size_scaling")

    # === BENCHMARK 2: Batch Size Scaling ===
    with runner.section("Batch Size Scaling"):
        runner.logger.info("Benchmarking parallel batch scaling...")

        fixed_size = 15  # Moderate grid size
        batch_results = []

        for n_batches in batch_sizes:
            runner.logger.info(f"\n  Testing {n_batches} parallel chains...")

            key, subkey = jax.random.split(key)

            # Warmup JIT
            _ = run_potts_model_benchmark(fixed_size, n_cats, beta, n_batches, schedule, subkey)

            # Actual benchmark
            key, subkey = jax.random.split(key)
            result = run_potts_model_benchmark(fixed_size, n_cats, beta, n_batches, schedule, subkey)
            batch_results.append(result)

            runner.logger.info(f"    Updates/sec: {result['updates_per_second']/1e6:.2f}M")
            runner.logger.info(f"    Time/sample: {result['time_per_sample']*1000:.2f}ms")

        batch_data = {
            "batch_sizes": np.array(batch_sizes),
            "updates_per_sec": np.array([r["updates_per_second"] for r in batch_results]),
            "time_per_sample": np.array([r["time_per_sample"] for r in batch_results]),
        }
        runner.save_data(batch_data, "batch_scaling")

    # === VISUALIZATION ===
    with runner.section("Visualization"):
        fig, axes = create_figure(2, 2, figsize=(14, 10))

        # Throughput vs nodes
        ax = axes[0, 0]
        ax.plot(
            scaling_data["node_counts"],
            scaling_data["updates_per_sec"] / 1e6,
            "o-",
            linewidth=2,
            markersize=8,
            color="blue",
        )
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Throughput (Million updates/s)")
        ax.set_title("THRML Sampling Throughput")
        ax.grid(True, alpha=0.3)

        # Time per sample vs nodes
        ax = axes[0, 1]
        ax.semilogy(
            scaling_data["node_counts"],
            scaling_data["time_per_sample"] * 1000,
            "o-",
            linewidth=2,
            markersize=8,
            color="green",
        )
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Time per Sample (ms, log)")
        ax.set_title("THRML Sample Time Scaling")
        ax.grid(True, alpha=0.3)

        # Batch throughput
        ax = axes[1, 0]
        ax.plot(
            batch_data["batch_sizes"],
            batch_data["updates_per_sec"] / 1e6,
            "o-",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        ax.set_xlabel("Batch Size (Parallel Chains)")
        ax.set_ylabel("Throughput (Million updates/s)")
        ax.set_title("THRML Parallel Scaling")
        ax.grid(True, alpha=0.3)

        # Speedup
        ax = axes[1, 1]
        speedup = batch_data["updates_per_sec"] / batch_data["updates_per_sec"][0]
        efficiency = speedup / batch_data["batch_sizes"] * 100
        ax.plot(batch_data["batch_sizes"], speedup, "o-", linewidth=2, markersize=8, color="orange", label="Speedup")
        ax.plot(batch_data["batch_sizes"], batch_data["batch_sizes"], "--", color="gray", alpha=0.5, label="Ideal")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Speedup")
        ax.set_title("Parallel Efficiency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        runner.save_plot(fig, "thrml_performance", formats=["png", "pdf"])
        plt.close(fig)

    # === METRICS ===
    with runner.section("Summary Metrics"):
        best_throughput = float(np.max(scaling_data["updates_per_sec"]) / 1e6)
        best_parallel = float(np.max(batch_data["updates_per_sec"]) / 1e6)
        speedup = float(batch_data["updates_per_sec"][-1] / batch_data["updates_per_sec"][0])
        efficiency = float(speedup / batch_sizes[-1] * 100)

        runner.record_metric("best_throughput_mupdates_per_sec", best_throughput)
        runner.record_metric("best_parallel_throughput", best_parallel)
        runner.record_metric("parallel_speedup", speedup)
        runner.record_metric("parallel_efficiency_pct", efficiency)

        runner.logger.info(f"\n{'='*70}")
        runner.logger.info("THRML PERFORMANCE SUMMARY (Potts Model)")
        runner.logger.info(f"{'='*70}")
        runner.logger.info(f"Best Throughput: {best_throughput:.2f}M updates/s")
        runner.logger.info(f"Parallel Speedup ({batch_sizes[0]}→{batch_sizes[-1]}): {speedup:.2f}x")
        runner.logger.info(f"Parallel Efficiency: {efficiency:.1f}%")
        runner.logger.info("Method: Exact Potts model from notebook 00")
        runner.logger.info(f"JAX Backend: {jax.default_backend()}")
        runner.logger.info(f"{'='*70}\n")

    runner.end()
    runner.logger.info("✓ THRML performance benchmarking complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
