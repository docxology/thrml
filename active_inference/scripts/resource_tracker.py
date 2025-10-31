#!/usr/bin/env python3
"""Resource tracking helper script for run_all_examples.sh

Provides resource usage information that can be called from bash scripts.
"""

import json
import os
import sys
from pathlib import Path

import psutil


def get_system_info():
    """Get comprehensive system information."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())

    info = {
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(mem.total / (1024**3), 2),
        "memory_available_gb": round(mem.available / (1024**3), 2),
        "memory_percent": round(mem.percent, 1),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "disk_percent": round(disk.percent, 1),
    }

    return info


def get_process_info():
    """Get current process resource usage."""
    process = psutil.Process(os.getpid())

    info = {
        "cpu_percent": round(process.cpu_percent(interval=0.1), 1),
        "memory_mb": round(process.memory_info().rss / (1024**2), 1),
        "memory_percent": round(process.memory_percent(), 2),
    }

    return info


def estimate_run_resources(n_examples: int = 11):
    """Estimate resources needed for running examples."""
    # Based on empirical measurements
    avg_time_per_example = 3.0  # seconds
    avg_memory_per_example = 50  # MB
    avg_output_per_example = 0.4  # MB

    estimates = {
        "estimated_time_seconds": n_examples * avg_time_per_example,
        "estimated_time_minutes": round((n_examples * avg_time_per_example) / 60, 1),
        "estimated_memory_mb": n_examples * avg_memory_per_example,
        "estimated_output_mb": round(n_examples * avg_output_per_example, 1),
    }

    return estimates


def format_size(bytes_val):
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}PB"


def get_directory_size(directory):
    """Get total size of directory recursively."""
    total = 0
    try:
        for entry in os.scandir(directory):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_directory_size(entry.path)
    except PermissionError:
        pass
    return total


def print_system_banner():
    """Print formatted system information banner."""
    info = get_system_info()

    print()
    print("=" * 70)
    print("SYSTEM RESOURCES")
    print("=" * 70)
    print(f"CPU:    {info['cpu_cores_physical']} physical, {info['cpu_cores_logical']} logical cores")
    print(
        f"Memory: {info['memory_available_gb']:.1f}GB / {info['memory_total_gb']:.1f}GB available ({100-info['memory_percent']:.0f}% free)"
    )
    print(
        f"Disk:   {info['disk_free_gb']:.1f}GB / {info['disk_total_gb']:.1f}GB free ({100-info['disk_percent']:.0f}% free)"
    )
    print("=" * 70)
    print()


def print_estimates(n_examples=11):
    """Print resource estimates."""
    est = estimate_run_resources(n_examples)

    print()
    print("=" * 70)
    print("RESOURCE ESTIMATES")
    print("=" * 70)
    print(f"Examples:       {n_examples}")
    print(f"Est. Duration:  {est['estimated_time_seconds']:.0f}s ({est['estimated_time_minutes']:.1f}m)")
    print(f"Est. Memory:    ~{est['estimated_memory_mb']:.0f}MB peak")
    print(f"Est. Output:    ~{est['estimated_output_mb']:.1f}MB")
    print("=" * 70)
    print()


def get_output_stats(output_dir):
    """Get statistics about output directory."""
    output_path = Path(output_dir)

    if not output_path.exists():
        return None

    total_size = get_directory_size(output_dir)

    # Count files by type
    stats = {
        "total_size": total_size,
        "total_size_formatted": format_size(total_size),
        "n_directories": 0,
        "n_files": 0,
        "by_extension": {},
    }

    for root, dirs, files in os.walk(output_dir):
        stats["n_directories"] += len(dirs)
        stats["n_files"] += len(files)

        for file in files:
            ext = Path(file).suffix.lower() or ".no_ext"
            stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: resource_tracker.py [system|process|estimates|output-stats]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "system":
        info = get_system_info()
        print(json.dumps(info, indent=2))

    elif command == "process":
        info = get_process_info()
        print(json.dumps(info, indent=2))

    elif command == "estimates":
        n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 11
        est = estimate_run_resources(n_examples)
        print(json.dumps(est, indent=2))

    elif command == "output-stats":
        if len(sys.argv) < 3:
            print("Usage: resource_tracker.py output-stats <directory>")
            sys.exit(1)
        output_dir = sys.argv[2]
        stats = get_output_stats(output_dir)
        if stats:
            print(json.dumps(stats, indent=2))
        else:
            print("{}")

    elif command == "banner":
        print_system_banner()

    elif command == "estimate-banner":
        n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 11
        print_estimates(n_examples)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
