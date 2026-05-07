"""Unified plot style configuration for the paper."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
FONT_SIZE = 10
DPI = 300
FORMAT = "pdf"
FIG_DIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "mathtext.fontset": "stix",
    }
)

# Color palette: first few tab10 colors
COLORS = list(plt.cm.Set2.colors)

# Environment short-name mapping
ENV_SHORT = {
    "halfcheetah-medium-replay-v2": "HalfCheetah",
    "halfcheetah-medium-expert-v2": "HalfCheetah (ME)",
    "hopper-medium-replay-v2": "Hopper",
    "walker2d-medium-replay-v2": "Walker2D",
    "walker2d-medium-v2": "Walker2D (M)",
    "walker2d-medium-expert-v2": "Walker2D (ME)",
}

# Target short-name mapping
TARGET_SHORT = {
    "all": "All",
    "attn": "Attn",
    "attn.layer_0": "Attn L0",
    "attn.layer_1": "Attn L1",
    "attn.layer_2": "Attn L2",
}

# Model colors
MODEL_COLORS = {
    "DT": COLORS[0],
    "MLP": COLORS[1],
    "LSTM": COLORS[2],
}


def save_fig(fig, name, fmt=FORMAT):
    """Save figure to FIG_DIR."""
    path = os.path.join(FIG_DIR, f"{name}.{fmt}")
    fig.savefig(path)
    print(f"Saved: {path}")
    # Also save a PNG copy for easy preview
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
