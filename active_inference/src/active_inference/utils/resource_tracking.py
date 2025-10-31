"""Resource tracking and profiling utilities.

Provides comprehensive resource monitoring:
- CPU and memory tracking
- GPU monitoring (if available)
- Execution timing
- Disk space tracking
- Performance profiling

All tracking uses real system measurements, not mocks.
"""

import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import psutil


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time.

    Attributes:
        timestamp: When snapshot was taken
        cpu_percent: CPU usage percentage
        memory_used_mb: Memory used in MB
        memory_percent: Memory usage percentage
        disk_usage_mb: Disk usage in MB (optional)
        gpu_info: GPU information if available

    """

    timestamp: float
    cpu_percent: float
    memory_used_mb: float
    memory_percent: float
    disk_usage_mb: Optional[float] = None
    gpu_info: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return (
            f"Resources @ {datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S')}: "
            f"CPU={self.cpu_percent:.1f}%, "
            f"Memory={self.memory_used_mb:.1f}MB ({self.memory_percent:.1f}%)"
        )


@dataclass
class PerformanceProfile:
    """Performance profile for a code section.

    Attributes:
        name: Section name
        duration: Execution time in seconds
        cpu_percent_avg: Average CPU usage
        memory_delta_mb: Memory change in MB
        start_snapshot: Resource snapshot at start
        end_snapshot: Resource snapshot at end

    """

    name: str
    duration: float
    cpu_percent_avg: float
    memory_delta_mb: float
    start_snapshot: ResourceSnapshot
    end_snapshot: ResourceSnapshot

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.duration:.3f}s "
            f"(CPU: {self.cpu_percent_avg:.1f}%, "
            f"Memory Δ: {self.memory_delta_mb:+.1f}MB)"
        )


class ResourceTracker:
    """Track system resources during execution.

    Monitors CPU, memory, disk, and GPU usage throughout execution.

    **Example:**
    ```python
    tracker = ResourceTracker()

    # Start tracking
    tracker.start()

    # Run computation
    result = expensive_computation()

    # Take snapshot
    tracker.snapshot("after_computation")

    # Stop and get report
    tracker.stop()
    report = tracker.generate_report()
    print(report)
    ```
    """

    def __init__(self, interval: float = 1.0):
        """Initialize resource tracker.

        **Arguments:**

        - `interval`: Sampling interval in seconds (for continuous tracking)
        """
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self.profiles: List[PerformanceProfile] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._process = psutil.Process(os.getpid())
        self._tracking = False

    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information if available.

        **Returns:**

        - Dictionary with GPU info or None if unavailable
        """
        try:
            # Check JAX GPU availability
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == "gpu"]

            if gpu_devices:
                return {
                    "n_gpus": len(gpu_devices),
                    "devices": [str(d) for d in gpu_devices],
                    "backend": jax.default_backend(),
                }
        except Exception:
            pass

        # Try nvidia-smi as fallback
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []
                for line in lines:
                    parts = line.split(", ")
                    if len(parts) == 3:
                        gpus.append(
                            {
                                "name": parts[0],
                                "memory_used_mb": float(parts[1]),
                                "memory_total_mb": float(parts[2]),
                            }
                        )
                return {"gpus": gpus, "n_gpus": len(gpus)}
        except Exception:
            pass

        return None

    def take_snapshot(self, name: Optional[str] = None) -> ResourceSnapshot:
        """Take a resource snapshot.

        **Arguments:**

        - `name`: Optional name for this snapshot

        **Returns:**

        - ResourceSnapshot with current resource usage
        """
        # Get memory info
        mem = psutil.virtual_memory()
        process_mem = self._process.memory_info()

        # Get CPU usage (per-process)
        cpu_percent = self._process.cpu_percent()

        # Get disk usage for current directory
        try:
            disk = psutil.disk_usage(os.getcwd())
            disk_usage_mb = (disk.total - disk.free) / (1024**2)
        except Exception:
            disk_usage_mb = None

        # Get GPU info
        gpu_info = self._get_gpu_info()

        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_used_mb=process_mem.rss / (1024**2),
            memory_percent=mem.percent,
            disk_usage_mb=disk_usage_mb,
            gpu_info=gpu_info,
        )

        self.snapshots.append(snapshot)

        if name:
            print(f"📊 Snapshot '{name}': {snapshot}")

        return snapshot

    def start(self):
        """Start resource tracking."""
        self.start_time = time.time()
        self._tracking = True
        self.take_snapshot("start")
        print(f"⏱️  Resource tracking started at {datetime.fromtimestamp(self.start_time).strftime('%H:%M:%S')}")

    def stop(self):
        """Stop resource tracking."""
        if not self._tracking:
            return

        self.end_time = time.time()
        self._tracking = False
        self.take_snapshot("end")

        duration = self.end_time - self.start_time
        print(f"⏱️  Resource tracking stopped. Duration: {duration:.2f}s")

    def profile_section(self, name: str, start_snapshot: ResourceSnapshot) -> PerformanceProfile:
        """Profile a code section.

        **Arguments:**

        - `name`: Section name
        - `start_snapshot`: Snapshot from section start

        **Returns:**

        - PerformanceProfile with section statistics
        """
        end_snapshot = self.take_snapshot(f"{name}_end")

        duration = end_snapshot.timestamp - start_snapshot.timestamp
        cpu_avg = (start_snapshot.cpu_percent + end_snapshot.cpu_percent) / 2
        memory_delta = end_snapshot.memory_used_mb - start_snapshot.memory_used_mb

        profile = PerformanceProfile(
            name=name,
            duration=duration,
            cpu_percent_avg=cpu_avg,
            memory_delta_mb=memory_delta,
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
        )

        self.profiles.append(profile)
        print(f"⏲️  {profile}")

        return profile

    def get_peak_memory(self) -> float:
        """Get peak memory usage.

        **Returns:**

        - Peak memory in MB
        """
        if not self.snapshots:
            return 0.0
        return max(s.memory_used_mb for s in self.snapshots)

    def get_avg_cpu(self) -> float:
        """Get average CPU usage.

        **Returns:**

        - Average CPU percentage
        """
        if not self.snapshots:
            return 0.0
        return sum(s.cpu_percent for s in self.snapshots) / len(self.snapshots)

    def generate_report(self) -> str:
        """Generate comprehensive resource report.

        **Returns:**

        - Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("RESOURCE USAGE REPORT")
        report.append("=" * 70)
        report.append("")

        # System info
        report.append("SYSTEM INFORMATION")
        report.append("-" * 70)
        report.append(f"OS: {platform.system()} {platform.release()}")
        report.append(f"Architecture: {platform.machine()}")
        report.append(f"Processor: {platform.processor()}")
        report.append(
            f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical"
        )

        mem = psutil.virtual_memory()
        report.append(f"Total Memory: {mem.total / (1024 ** 3):.1f} GB")

        # GPU info
        gpu_info = self._get_gpu_info()
        if gpu_info:
            report.append(f"GPUs: {gpu_info.get('n_gpus', 0)} detected")
            if "backend" in gpu_info:
                report.append(f"JAX Backend: {gpu_info['backend']}")
        else:
            report.append("GPUs: None detected")

        report.append("")

        # Execution summary
        if self.start_time and self.end_time:
            report.append("EXECUTION SUMMARY")
            report.append("-" * 70)
            duration = self.end_time - self.start_time
            report.append(f"Total Duration: {duration:.2f}s ({duration / 60:.2f}m)")
            report.append(f"Start Time: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"End Time: {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")

        # Resource statistics
        if self.snapshots:
            report.append("RESOURCE STATISTICS")
            report.append("-" * 70)

            peak_mem = self.get_peak_memory()
            avg_cpu = self.get_avg_cpu()

            mem_values = [s.memory_used_mb for s in self.snapshots]
            cpu_values = [s.cpu_percent for s in self.snapshots]

            report.append("Memory Usage:")
            report.append(f"  Peak: {peak_mem:.1f} MB")
            report.append(f"  Average: {sum(mem_values) / len(mem_values):.1f} MB")
            report.append(f"  Range: [{min(mem_values):.1f}, {max(mem_values):.1f}] MB")
            report.append("")

            report.append("CPU Usage:")
            report.append(f"  Peak: {max(cpu_values):.1f}%")
            report.append(f"  Average: {avg_cpu:.1f}%")
            report.append(f"  Range: [{min(cpu_values):.1f}, {max(cpu_values):.1f}]%")
            report.append("")

        # Section profiles
        if self.profiles:
            report.append("SECTION PROFILES")
            report.append("-" * 70)

            total_profiled = sum(p.duration for p in self.profiles)

            for profile in self.profiles:
                pct = (profile.duration / total_profiled * 100) if total_profiled > 0 else 0
                report.append(f"\n{profile.name}:")
                report.append(f"  Duration: {profile.duration:.3f}s ({pct:.1f}%)")
                report.append(f"  CPU Average: {profile.cpu_percent_avg:.1f}%")
                report.append(f"  Memory Delta: {profile.memory_delta_mb:+.1f} MB")

            report.append("")

        # Disk usage
        if self.snapshots and self.snapshots[0].disk_usage_mb is not None:
            report.append("DISK USAGE")
            report.append("-" * 70)
            disk_start = self.snapshots[0].disk_usage_mb
            disk_end = self.snapshots[-1].disk_usage_mb
            disk_delta = disk_end - disk_start
            report.append(f"Start: {disk_start:.1f} MB")
            report.append(f"End: {disk_end:.1f} MB")
            report.append(f"Change: {disk_delta:+.1f} MB")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def save_report(self, output_path: Path):
        """Save report to file.

        **Arguments:**

        - `output_path`: Path to save report
        """
        report = self.generate_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"📊 Resource report saved to {output_path}")


def estimate_resources(
    n_states: int, n_observations: int, n_actions: int, n_steps: int, n_samples: Optional[int] = None
) -> Dict[str, float]:
    """Estimate resource requirements for an active inference run.

    **Arguments:**

    - `n_states`: Number of hidden states
    - `n_observations`: Number of observations
    - `n_actions`: Number of actions
    - `n_steps`: Number of time steps
    - `n_samples`: Number of samples for THRML (optional)

    **Returns:**

    - Dictionary with estimated time, memory, etc.

    **Example:**
    ```python
    estimates = estimate_resources(
        n_states=10,
        n_observations=10,
        n_actions=4,
        n_steps=100
    )
    print(f"Estimated time: {estimates['time_seconds']:.1f}s")
    print(f"Estimated memory: {estimates['memory_mb']:.1f}MB")
    ```
    """
    # Empirical estimates based on profiling
    # These are rough estimates and should be calibrated

    # Memory estimates (MB)
    # Model storage
    model_mem = (
        (
            n_observations * n_states  # A matrix
            + n_actions * n_states * n_states  # B matrices
            + n_states  # D vector
            + n_observations  # C vector
        )
        * 4
        / (1024**2)
    )  # 4 bytes per float32

    # Trajectory storage
    trajectory_mem = (n_steps * n_states + n_steps) * 4 / (1024**2)  # beliefs  # actions, observations

    # Working memory (conservative estimate)
    working_mem = model_mem * 5  # Factor for intermediate computations

    total_mem = model_mem + trajectory_mem + working_mem

    # Time estimates (seconds)
    # Based on empirical measurements: ~0.001s per inference step for n_states=10
    base_time_per_step = 0.001 * (n_states / 10) ** 1.5  # Scales with state space
    total_time = base_time_per_step * n_steps

    # THRML sampling overhead
    if n_samples:
        sampling_time = 0.0001 * n_samples * (n_states / 10)
        total_time += sampling_time

    # CPU estimate (percentage)
    # Single-threaded JAX operations typically use 100-200% (1-2 cores)
    estimated_cpu = min(100 * 2, psutil.cpu_count(logical=True) * 100)

    return {
        "time_seconds": total_time,
        "time_minutes": total_time / 60,
        "memory_mb": total_mem,
        "memory_gb": total_mem / 1024,
        "model_memory_mb": model_mem,
        "trajectory_memory_mb": trajectory_mem,
        "working_memory_mb": working_mem,
        "estimated_cpu_percent": estimated_cpu,
        "n_operations": n_steps * n_states * n_observations,  # Rough operation count
    }


def print_resource_estimates(estimates: Dict[str, float]):
    """Print resource estimates in a formatted way.

    **Arguments:**

    - `estimates`: Dictionary from estimate_resources
    """
    print("\n" + "=" * 70)
    print("RESOURCE ESTIMATES")
    print("=" * 70)
    print(f"\nEstimated Duration: {estimates['time_seconds']:.2f}s ({estimates['time_minutes']:.2f}m)")
    print(f"Estimated Memory: {estimates['memory_mb']:.1f}MB ({estimates['memory_gb']:.3f}GB)")
    print(f"  - Model: {estimates['model_memory_mb']:.1f}MB")
    print(f"  - Trajectory: {estimates['trajectory_memory_mb']:.1f}MB")
    print(f"  - Working: {estimates['working_memory_mb']:.1f}MB")
    print(f"Estimated CPU: {estimates['estimated_cpu_percent']:.0f}%")
    print(f"Operations: {estimates['n_operations']:.0e}")
    print("=" * 70 + "\n")
