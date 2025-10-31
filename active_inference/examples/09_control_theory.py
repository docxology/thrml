"""Example 09: Control Theory with Active Inference.

Demonstrates:
- Linear Quadratic Regulator (LQR)
- PID control
- State space models
- Setpoint tracking
- Disturbance rejection
- Active inference as control
- THRML-based state estimation with noisy observations

**THRML Integration**:
- Uses THRML sampling-based inference (`ThrmlInferenceEngine`) for belief state tracking
- Real THRML methods: `CategoricalNode`, `Block`, `BlockGibbsSpec`, `CategoricalEBMFactor`
- Discretizes continuous state space for THRML inference
- GPU-accelerated block Gibbs sampling for energy-efficient state estimation
- Comprehensive demonstration of THRML for control under uncertainty
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from example_utils import ExampleRunner, create_figure

from active_inference.core import GenerativeModel
from active_inference.inference import ThrmlInferenceEngine

# Configuration
OUTPUT_BASE = Path(__file__).parent.parent / "output"
EXAMPLE_NAME = "09_control_theory"


class LinearSystem:
    """Linear dynamical system: x_{t+1} = Ax_t + Bu_t + w_t"""

    def __init__(self, A, B, process_noise_std=0.01):
        self.A = A
        self.B = B
        self.process_noise_std = process_noise_std
        self.n_states = A.shape[0]
        self.n_controls = B.shape[1]

    def step(self, state, control, key):
        """Step the system forward."""
        next_state = jnp.dot(self.A, state) + jnp.dot(self.B, control)
        if self.process_noise_std > 0:
            noise = jax.random.normal(key, shape=state.shape) * self.process_noise_std
            next_state = next_state + noise
        return next_state


def solve_discrete_lqr(A, B, Q, R, max_iter=100, tol=1e-6):
    """Solve discrete-time LQR using iterative Riccati equation."""
    n = A.shape[0]
    P = Q.copy()

    for _ in range(max_iter):
        P_next = (
            Q
            + jnp.dot(jnp.dot(A.T, P), A)
            - jnp.dot(
                jnp.dot(jnp.dot(jnp.dot(A.T, P), B), jnp.linalg.inv(R + jnp.dot(jnp.dot(B.T, P), B))),
                jnp.dot(jnp.dot(B.T, P), A),
            )
        )

        if jnp.linalg.norm(P_next - P) < tol:
            P = P_next
            break
        P = P_next

    # Compute optimal gain
    K = jnp.linalg.inv(R + jnp.dot(jnp.dot(B.T, P), B)) @ jnp.dot(jnp.dot(B.T, P), A)

    return K, P


class PIDController:
    """PID controller with anti-windup."""

    def __init__(self, Kp, Ki, Kd, dt, max_integral=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.max_integral = max_integral
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, error):
        """Compute PID control signal."""
        # Proportional
        p_term = self.Kp * error

        # Integral (with anti-windup)
        self.integral = jnp.clip(self.integral + error * self.dt, -self.max_integral, self.max_integral)
        i_term = self.Ki * self.integral

        # Derivative
        d_term = self.Kd * (error - self.prev_error) / self.dt
        self.prev_error = error

        # Total control
        control = p_term + i_term + d_term

        return control


def main():
    """Run control theory example."""
    runner = ExampleRunner(
        EXAMPLE_NAME,
        OUTPUT_BASE,
        enable_profiling=True,
        enable_validation=True,
    )
    runner.start()

    # === LOAD CONFIGURATION ===
    # All parameters from examples_config.yaml
    seed = runner.get_config("seed", default=42)
    n_steps = runner.get_config("n_steps", default=200)
    dt = runner.get_config("dt", default=0.1)
    setpoint = runner.get_config("setpoint", default=1.0)
    disturbance_time = runner.get_config("disturbance_time", default=100)
    disturbance_magnitude = runner.get_config("disturbance_magnitude", default=0.5)

    # LQR parameters
    lqr_Q_diag = runner.get_config("lqr_Q_diag", default=[10.0, 1.0])
    lqr_R = runner.get_config("lqr_R", default=0.1)

    # PID parameters
    pid_Kp = runner.get_config("pid_Kp", default=2.0)
    pid_Ki = runner.get_config("pid_Ki", default=0.5)
    pid_Kd = runner.get_config("pid_Kd", default=1.0)

    # System matrices
    A_list = runner.get_config("A_matrix", default=[[1.0, 0.1], [0.0, 1.0]])
    B_list = runner.get_config("B_matrix", default=[[0.005], [0.1]])
    A = jnp.array(A_list)
    B = jnp.array(B_list)

    # THRML state estimation parameters
    n_samples = runner.get_config("n_samples", default=200)
    n_warmup = runner.get_config("n_warmup", default=50)
    steps_per_sample = runner.get_config("steps_per_sample", default=5)
    n_state_bins = runner.get_config("n_state_bins", default=50)  # Discretization for THRML
    observation_noise_std = runner.get_config("observation_noise_std", default=0.1)  # Sensor noise

    # Build Q matrix from diagonal
    Q = jnp.diag(jnp.array(lqr_Q_diag))
    R = jnp.array([[lqr_R]])

    # Set random seed from config
    key = jax.random.key(seed)

    runner.logger.info("Configuration loaded:")
    runner.logger.info(f"  seed: {seed}")
    runner.logger.info(f"  n_steps: {n_steps}")
    runner.logger.info(f"  dt: {dt}")
    runner.logger.info(f"  setpoint: {setpoint}")

    # === 1. CONFIGURATION ===
    with runner.section("Configuration"):
        # Use config-loaded values
        Kp = pid_Kp
        Ki = pid_Ki
        Kd = pid_Kd

        runner.logger.info("System: Double integrator (position + velocity)")
        runner.logger.info(f"State dimension: {A.shape[0]}")
        runner.logger.info(f"Control dimension: {B.shape[1]}")
        runner.logger.info(f"Simulation steps: {n_steps}")
        runner.logger.info(f"Setpoint: {setpoint}")
        runner.logger.info(f"PID gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")

        config = {
            "seed": seed,
            "A_matrix": A.tolist(),
            "B_matrix": B.tolist(),
            "Q_matrix": Q.tolist(),
            "R_matrix": R.tolist(),
            "pid_kp": Kp,
            "pid_ki": Ki,
            "pid_kd": Kd,
            "dt": dt,
            "n_steps": n_steps,
            "setpoint": setpoint,
            "disturbance_time": disturbance_time,
            "disturbance_magnitude": disturbance_magnitude,
            "n_state_bins": n_state_bins,
            "observation_noise_std": observation_noise_std,
            "thrml_n_samples": n_samples,
            "thrml_n_warmup": n_warmup,
            "thrml_steps_per_sample": steps_per_sample,
        }
        runner.save_config(config)

    # === 2. LQR CONTROLLER ===
    with runner.section("LQR Controller Design"):
        with runner.profile("LQR Solution"):
            K_lqr, P_lqr = solve_discrete_lqr(A, B, Q, R)

        runner.logger.info(f"LQR gain matrix K:\n{K_lqr}")
        runner.logger.info(f"Cost-to-go matrix P:\n{P_lqr}")
        runner.validate_data(K_lqr, "lqr_gain")
        runner.validate_data(P_lqr, "lqr_cost_matrix")

    # === 3. SIMULATION: LQR CONTROL ===
    with runner.section("LQR Simulation"):
        system_lqr = LinearSystem(A, B, process_noise_std=0.01)

        state_lqr = jnp.array([0.0, 0.0])  # Start at origin
        states_lqr = [state_lqr]
        controls_lqr = []
        errors_lqr = []

        for step in range(n_steps):
            # Add disturbance at midpoint
            target_state = jnp.array([setpoint, 0.0])
            if step == disturbance_time:
                state_lqr = state_lqr + jnp.array([disturbance_magnitude, 0.0])
                runner.logger.info(f"Disturbance applied at step {step}")

            # LQR control: u = -K(x - x_target)
            error = state_lqr - target_state
            control = -jnp.dot(K_lqr, error)

            # Step system
            key, subkey = jax.random.split(key)
            state_lqr = system_lqr.step(state_lqr, control, subkey)

            states_lqr.append(state_lqr)
            controls_lqr.append(control)
            errors_lqr.append(jnp.linalg.norm(error))

        states_lqr = np.array(states_lqr)
        controls_lqr = np.array(controls_lqr)
        errors_lqr = np.array(errors_lqr)

        runner.logger.info(f"Final position: {states_lqr[-1, 0]:.4f}")
        runner.logger.info(f"Final velocity: {states_lqr[-1, 1]:.4f}")
        runner.logger.info(f"Mean squared error: {np.mean(errors_lqr**2):.6f}")

    # === 4. SIMULATION: PID CONTROL ===
    with runner.section("PID Simulation"):
        system_pid = LinearSystem(A, B, process_noise_std=0.01)
        pid = PIDController(Kp, Ki, Kd, dt)

        key = jax.random.key(seed)  # Reset for fair comparison
        state_pid = jnp.array([0.0, 0.0])
        states_pid = [state_pid]
        controls_pid = []
        errors_pid = []

        for step in range(n_steps):
            # Add same disturbance
            if step == disturbance_time:
                state_pid = state_pid + jnp.array([disturbance_magnitude, 0.0])

            # PID control on position error
            position_error = setpoint - state_pid[0]
            control = pid.control(position_error)
            control = jnp.array([control])  # Make it a vector

            # Step system
            key, subkey = jax.random.split(key)
            state_pid = system_pid.step(state_pid, control, subkey)

            states_pid.append(state_pid)
            controls_pid.append(control)
            errors_pid.append(abs(position_error))

        states_pid = np.array(states_pid)
        controls_pid = np.array(controls_pid)
        errors_pid = np.array(errors_pid)

        runner.logger.info(f"Final position: {states_pid[-1, 0]:.4f}")
        runner.logger.info(f"Final velocity: {states_pid[-1, 1]:.4f}")
        runner.logger.info(f"Mean squared error: {np.mean(errors_pid**2):.6f}")

    # === 5. THRML STATE ESTIMATION WITH NOISY OBSERVATIONS ===
    with runner.section("THRML State Estimation"):
        runner.logger.info("Setting up THRML state estimation with discretized state space")
        runner.logger.info(f"State discretization: {n_state_bins} bins per dimension")

        # Discretize state space: position and velocity
        # For simplicity, discretize position only (can extend to 2D later)
        pos_min, pos_max = -2.0, 3.0
        pos_grid = jnp.linspace(pos_min, pos_max, n_state_bins)
        bin_width = (pos_max - pos_min) / (n_state_bins - 1)

        # Create observation model: noisy observations of position
        # Observations are discretized position bins
        # A[i, j] = P(observation_bin i | state_bin j)
        # Use Gaussian observation model with noise
        A_obs = jnp.zeros((n_state_bins, n_state_bins))
        for i in range(n_state_bins):
            for j in range(n_state_bins):
                true_pos = pos_grid[j]
                obs_pos = pos_grid[i]
                # Gaussian likelihood with noise
                likelihood = jnp.exp(-0.5 * ((obs_pos - true_pos) / observation_noise_std) ** 2)
                A_obs = A_obs.at[i, j].set(likelihood)
        # Normalize columns
        A_obs = A_obs / (jnp.sum(A_obs, axis=0, keepdims=True) + 1e-16)

        # Transition model: simple random walk (can be improved with actual dynamics)
        B_trans = jnp.eye(n_state_bins) * 0.9 + jnp.eye(n_state_bins, k=1) * 0.05 + jnp.eye(n_state_bins, k=-1) * 0.05
        B_trans = B_trans / (jnp.sum(B_trans, axis=1, keepdims=True) + 1e-16)
        B_trans = B_trans[jnp.newaxis, :, :]  # Add action dimension

        # Prior: uniform
        D_prior = jnp.ones(n_state_bins) / n_state_bins
        C_prefs = jnp.zeros(n_state_bins)

        # Create generative model for THRML
        thrml_model = GenerativeModel(
            n_states=n_state_bins,
            n_observations=n_state_bins,
            n_actions=1,
            A=A_obs,
            B=B_trans,
            C=C_prefs,
            D=D_prior,
        )

        # Create THRML inference engine
        thrml_engine = ThrmlInferenceEngine(
            model=thrml_model,
            n_samples=n_samples,
            n_warmup=n_warmup,
            steps_per_sample=steps_per_sample,
        )

        runner.logger.info("Running THRML state estimation on LQR trajectory")

        # Run estimation on LQR trajectory with noisy observations
        key_thrml = jax.random.key(seed + 1000)
        beliefs_thrml = []
        observations_thrml = []
        true_positions = []
        estimated_positions = []

        # Reset to uniform prior
        current_belief = D_prior.copy()

        for step in range(n_steps):
            # Get true position from LQR trajectory
            true_pos = float(states_lqr[step, 0])
            true_positions.append(true_pos)

            # Generate noisy observation (discretized)
            key_thrml, subkey_obs = jax.random.split(key_thrml)
            noisy_pos = true_pos + jax.random.normal(subkey_obs) * observation_noise_std
            # Clamp and discretize
            noisy_pos = jnp.clip(noisy_pos, pos_min, pos_max)
            obs_bin = int(jnp.argmin(jnp.abs(pos_grid - noisy_pos)))
            observations_thrml.append(obs_bin)

            # THRML inference
            key_thrml, subkey_infer = jax.random.split(key_thrml)
            model_with_prior = eqx.tree_at(lambda m: m.D, thrml_model, current_belief)
            temp_engine = eqx.tree_at(lambda e: e.model, thrml_engine, model_with_prior)
            posterior = temp_engine.infer_with_sampling(
                key=subkey_infer,
                observation=obs_bin,
                n_state_samples=n_samples,
            )
            beliefs_thrml.append(posterior)

            # Estimate position from belief
            est_pos = float(jnp.sum(posterior * pos_grid))
            estimated_positions.append(est_pos)

            # Update prior for next step (use transition model)
            current_belief = jnp.dot(posterior, B_trans[0, :, :])
            current_belief = current_belief / (jnp.sum(current_belief) + 1e-16)

        beliefs_thrml = np.array(beliefs_thrml)
        estimated_positions = np.array(estimated_positions)
        true_positions = np.array(true_positions)
        observation_positions = np.array([float(pos_grid[obs]) for obs in observations_thrml])

        # Calculate estimation errors
        estimation_errors = np.abs(estimated_positions - true_positions)
        observation_errors = np.abs(observation_positions - true_positions)

        runner.logger.info(f"THRML estimation RMSE: {np.sqrt(np.mean(estimation_errors**2)):.6f}")
        runner.logger.info(f"Raw observation RMSE: {np.sqrt(np.mean(observation_errors**2)):.6f}")
        runner.logger.info(
            f"THRML improvement: {(1 - np.sqrt(np.mean(estimation_errors**2)) / np.sqrt(np.mean(observation_errors**2))) * 100:.2f}%"
        )

        runner.validate_data(beliefs_thrml, "thrml_beliefs")
        runner.validate_data(estimated_positions, "thrml_estimated_positions")

    # === 6. SAVE RESULTS ===
    with runner.section("Data Saving"):
        results = {
            "time": np.arange(n_steps + 1) * dt,
            "states_lqr": states_lqr,
            "controls_lqr": controls_lqr,
            "errors_lqr": errors_lqr,
            "states_pid": states_pid,
            "controls_pid": controls_pid,
            "errors_pid": errors_pid,
            "K_lqr": np.array(K_lqr),
            "P_lqr": np.array(P_lqr),
            "thrml_beliefs": beliefs_thrml,
            "thrml_estimated_positions": estimated_positions,
            "thrml_true_positions": true_positions,
            "thrml_observation_positions": observation_positions,
            "thrml_estimation_errors": estimation_errors,
            "thrml_observation_errors": observation_errors,
            "thrml_pos_grid": np.array(pos_grid),
        }
        runner.save_data(results, "control_results")

    # === 7. VISUALIZATIONS ===
    with runner.section("Visualization"):
        time = np.arange(n_steps + 1) * dt

        fig, axes = create_figure(3, 2, figsize=(14, 12))

        # Position tracking
        ax = axes[0, 0]
        ax.plot(time, states_lqr[:, 0], "b-", label="LQR", linewidth=2)
        ax.plot(time, states_pid[:, 0], "r--", label="PID", linewidth=2)
        ax.axhline(setpoint, color="g", linestyle=":", label="Setpoint")
        ax.axvline(disturbance_time * dt, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")
        ax.set_title("Position Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Velocity
        ax = axes[0, 1]
        ax.plot(time, states_lqr[:, 1], "b-", label="LQR", linewidth=2)
        ax.plot(time, states_pid[:, 1], "r--", label="PID", linewidth=2)
        ax.axhline(0, color="g", linestyle=":", label="Target")
        ax.axvline(disturbance_time * dt, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity")
        ax.set_title("Velocity Profile")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Control signals
        ax = axes[1, 0]
        ax.plot(time[:-1], controls_lqr, "b-", label="LQR", linewidth=2)
        ax.plot(time[:-1], controls_pid, "r--", label="PID", linewidth=2)
        ax.axvline(disturbance_time * dt, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control Signal")
        ax.set_title("Control Effort")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Tracking errors
        ax = axes[1, 1]
        ax.semilogy(time[:-1], errors_lqr, "b-", label="LQR", linewidth=2)
        ax.semilogy(time[:-1], errors_pid, "r--", label="PID", linewidth=2)
        ax.axvline(disturbance_time * dt, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (log scale)")
        ax.set_title("Tracking Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase portrait - LQR
        ax = axes[2, 0]
        ax.plot(states_lqr[:, 0], states_lqr[:, 1], "b-", linewidth=2)
        ax.plot(states_lqr[0, 0], states_lqr[0, 1], "go", markersize=10, label="Start")
        ax.plot(setpoint, 0, "r*", markersize=15, label="Target")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("Phase Portrait - LQR")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase portrait - PID
        ax = axes[2, 1]
        ax.plot(states_pid[:, 0], states_pid[:, 1], "r-", linewidth=2)
        ax.plot(states_pid[0, 0], states_pid[0, 1], "go", markersize=10, label="Start")
        ax.plot(setpoint, 0, "r*", markersize=15, label="Target")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("Phase Portrait - PID")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        runner.save_plot(fig, "control_comparison", formats=["png", "pdf"])
        plt.close(fig)

        # THRML state estimation visualization
        fig_thrml, axes_thrml = create_figure(2, 2, figsize=(14, 10))

        # Position tracking with THRML
        ax = axes_thrml[0, 0]
        time_thrml = time[:-1]  # n_steps points
        ax.plot(time_thrml, true_positions, "g-", label="True Position", linewidth=2, alpha=0.7)
        ax.plot(time_thrml, observation_positions, "r.", label="Noisy Observations", markersize=2, alpha=0.5)
        ax.plot(time_thrml, estimated_positions, "b-", label="THRML Estimate", linewidth=2)
        ax.axhline(setpoint, color="k", linestyle=":", label="Setpoint", alpha=0.5)
        ax.axvline(disturbance_time * dt, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")
        ax.set_title("THRML State Estimation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Estimation error
        ax = axes_thrml[0, 1]
        ax.semilogy(time_thrml, observation_errors, "r-", label="Observation Error", linewidth=2, alpha=0.7)
        ax.semilogy(time_thrml, estimation_errors, "b-", label="THRML Error", linewidth=2)
        ax.axvline(disturbance_time * dt, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (log scale)")
        ax.set_title("Estimation Error Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Belief heatmap (show belief evolution)
        ax = axes_thrml[1, 0]
        belief_matrix = beliefs_thrml.T  # [n_states, n_steps]
        im = ax.imshow(
            belief_matrix, aspect="auto", cmap="viridis", origin="lower", extent=[0, n_steps * dt, pos_min, pos_max]
        )
        ax.plot(time_thrml, true_positions, "r-", linewidth=1, alpha=0.5, label="True Position")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")
        ax.set_title("THRML Belief Evolution")
        plt.colorbar(im, ax=ax, label="Belief")
        ax.legend()

        # Error distribution
        ax = axes_thrml[1, 1]
        ax.hist(observation_errors, bins=30, alpha=0.5, label="Observation", density=True)
        ax.hist(estimation_errors, bins=30, alpha=0.5, label="THRML", density=True)
        ax.set_xlabel("Absolute Error")
        ax.set_ylabel("Density")
        ax.set_title("Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        runner.save_plot(fig_thrml, "thrml_state_estimation", formats=["png", "pdf"])
        plt.close(fig_thrml)

    # === 8. METRICS ===
    with runner.section("Metrics"):
        # LQR metrics
        runner.record_metric("lqr_final_position", float(states_lqr[-1, 0]))
        runner.record_metric("lqr_final_velocity", float(states_lqr[-1, 1]))
        runner.record_metric("lqr_mse", float(np.mean(errors_lqr**2)))
        runner.record_metric("lqr_max_control", float(np.max(np.abs(controls_lqr))))
        runner.record_metric(
            "lqr_settling_time", float(np.argmax(errors_lqr < 0.05) * dt) if np.any(errors_lqr < 0.05) else n_steps * dt
        )

        # PID metrics
        runner.record_metric("pid_final_position", float(states_pid[-1, 0]))
        runner.record_metric("pid_final_velocity", float(states_pid[-1, 1]))
        runner.record_metric("pid_mse", float(np.mean(errors_pid**2)))
        runner.record_metric("pid_max_control", float(np.max(np.abs(controls_pid))))
        runner.record_metric(
            "pid_settling_time", float(np.argmax(errors_pid < 0.05) * dt) if np.any(errors_pid < 0.05) else n_steps * dt
        )

        # Comparison
        runner.record_metric("lqr_vs_pid_mse_ratio", float(np.mean(errors_lqr**2) / np.mean(errors_pid**2)))

        # THRML metrics
        runner.record_metric("thrml_estimation_rmse", float(np.sqrt(np.mean(estimation_errors**2))))
        runner.record_metric("thrml_observation_rmse", float(np.sqrt(np.mean(observation_errors**2))))
        runner.record_metric(
            "thrml_improvement_pct",
            float((1 - np.sqrt(np.mean(estimation_errors**2)) / np.sqrt(np.mean(observation_errors**2))) * 100),
        )
        runner.record_metric("thrml_n_state_bins", n_state_bins)
        runner.record_metric("thrml_n_samples", n_samples)

    runner.end()
    runner.logger.info("✓ Example complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
