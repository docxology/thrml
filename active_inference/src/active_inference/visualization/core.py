"""Core visualization utilities and configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class VisualizationConfig:
    """Configuration for visualization outputs."""

    # Output directories
    output_root: Path = field(default_factory=lambda: Path("output"))
    plots_dir: str = "plots"
    graphs_dir: str = "graphs"
    trajectories_dir: str = "trajectories"
    distributions_dir: str = "distributions"
    networks_dir: str = "networks"
    animations_dir: str = "animations"
    reports_dir: str = "reports"

    # Plot styling
    style: str = "seaborn-v0_8-darkgrid"
    figure_format: str = "png"
    dpi: int = 300
    transparent: bool = False

    # Default sizes
    default_figsize: tuple[int, int] = (10, 6)
    small_figsize: tuple[int, int] = (8, 6)
    large_figsize: tuple[int, int] = (14, 8)
    wide_figsize: tuple[int, int] = (16, 6)

    # Colors
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    accent_color: str = "#2ca02c"
    error_color: str = "#d62728"

    # Fonts
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10

    # Grid and axes
    grid_alpha: float = 0.3
    grid_linestyle: str = "--"

    def get_output_path(self, category: str, filename: str) -> Path:
        """Get full output path for a file in a specific category.

        **Arguments:**

        - `category`: Output category (plots, graphs, etc.)
        - `filename`: Name of the file

        **Returns:**

        - Full path to output file
        """
        cat_dir = getattr(self, f"{category}_dir", category)
        path = self.output_root / cat_dir
        path.mkdir(parents=True, exist_ok=True)

        # Add extension if not present
        if not filename.endswith(f".{self.figure_format}"):
            filename = f"{filename}.{self.figure_format}"

        return path / filename


# Global configuration instance
_config = VisualizationConfig()


def get_config() -> VisualizationConfig:
    """Get the global visualization configuration."""
    return _config


def set_output_dir(path: Union[str, Path]):
    """Set the root output directory for all visualizations."""
    global _config
    _config.output_root = Path(path)


def get_default_output_dir() -> Path:
    """Get the default output directory."""
    return _config.output_root


def create_figure(figsize: Optional[tuple[int, int]] = None, nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a matplotlib figure with standard styling.

    **Arguments:**

    - `figsize`: Figure size (width, height) in inches
    - `nrows`: Number of subplot rows
    - `ncols`: Number of subplot columns
    - `**kwargs`: Additional arguments passed to plt.subplots

    **Returns:**

    - Figure and axes objects
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    if figsize is None:
        figsize = _config.default_figsize

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def save_figure(fig, filename: str, category: str = "plots", close: bool = True, **kwargs):
    """Save a matplotlib figure to the configured output directory.

    **Arguments:**

    - `fig`: Matplotlib figure object
    - `filename`: Name for the saved file
    - `category`: Output category subfolder
    - `close`: Whether to close the figure after saving
    - `**kwargs`: Additional arguments passed to fig.savefig
    """
    output_path = _config.get_output_path(category, filename)

    # Merge kwargs with defaults
    save_kwargs = {
        "dpi": _config.dpi,
        "bbox_inches": "tight",
        "transparent": _config.transparent,
    }
    save_kwargs.update(kwargs)

    fig.savefig(output_path, **save_kwargs)
    print(f"Saved figure to: {output_path}")

    if close:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except ImportError:
            pass


def set_style(style: Optional[str] = None):
    """Set the matplotlib style for plots.

    **Arguments:**

    - `style`: Matplotlib style name (None uses config default)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    if style is None:
        style = _config.style

    try:
        plt.style.use(style)
    except:
        # Fallback to seaborn-v0_8 if specific style not available
        try:
            plt.style.use("seaborn-v0_8")
        except:
            # If seaborn not available, use default
            pass


def apply_standard_formatting(ax, config: Optional[VisualizationConfig] = None):
    """Apply standard formatting to a matplotlib axis.

    **Arguments:**

    - `ax`: Matplotlib axis object
    - `config`: Optional custom configuration
    """
    if config is None:
        config = _config

    ax.grid(True, alpha=config.grid_alpha, linestyle=config.grid_linestyle)
    ax.tick_params(labelsize=config.tick_fontsize)

    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=config.label_fontsize)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=config.label_fontsize)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=config.title_fontsize)


def ensure_matplotlib():
    """Ensure matplotlib is available, raise helpful error if not."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization.\n"
            "Install with: pip install matplotlib\n"
            "Or install all visualization dependencies: pip install active_inference[viz]"
        )


def ensure_seaborn():
    """Ensure seaborn is available, raise helpful error if not."""
    try:
        import seaborn as sns

        return sns
    except ImportError:
        raise ImportError(
            "seaborn is required for advanced visualizations.\n"
            "Install with: pip install seaborn\n"
            "Or install all visualization dependencies: pip install active_inference[viz]"
        )


def ensure_networkx():
    """Ensure networkx is available, raise helpful error if not."""
    try:
        import networkx as nx

        return nx
    except ImportError:
        raise ImportError(
            "networkx is required for graph visualizations.\n"
            "Install with: pip install networkx\n"
            "Or install all visualization dependencies: pip install active_inference[viz]"
        )
