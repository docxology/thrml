"""Test environments for active inference agents."""

from .grid_world import GridWorld, GridWorldConfig
from .tmaze import TMaze

__all__ = [
    "GridWorld",
    "GridWorldConfig",
    "TMaze",
]
