"""Active Inference implementation using THRML.

This package implements Active Inference—a unified framework for perception, action,
and learning based on the Free Energy Principle—using THRML's efficient block Gibbs
sampling for probabilistic graphical models.
"""

from . import agents, core, environments, inference, models, utils, visualization
from .core import *
from .models import *

__version__ = "0.1.0"

__all__ = [
    "core",
    "models",
    "inference",
    "agents",
    "environments",
    "utils",
    "visualization",
]
