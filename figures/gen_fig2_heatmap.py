"""Fig 2: Forget gap heatmap — env (rows) x target (columns), aggregated by step."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
import csv
import matplotlib.colors as mcolors


def create_smooth_pink_blue_cmap():
    """
    Create a smoother pink-to-blue gradient colormap.
    """
    colors = [
        (0.0, "#C97CA1"),
        (0.25, "#F6CFDA"),
        (0.5, "#FFFFFF"),
        (0.75, "#C8E0F1"),
        (1.0, "#5E8FBC"),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list("smooth_pink_blue", colors)
    return cmap


custom_cmap = create_smooth_pink_blue_cmap()
# Load data
rows = []
with open("results/analysis/selective_summary.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

envs = [
    "hopper-medium-replay-v2",
    "halfcheetah-medium-replay-v2",
    "walker2d-medium-replay-v2",
]
# Select primary targets (hierarchy levels studied in the paper)
targets = ["all", "attn", "attn.layer_1", "attn.layer_2"]
target_labels = ["All", "Attn", "Attn L1", "Attn L2"]
env_labels = ["Hopper", "HalfCheetah", "Walker2D"]

# Average the gap across all steps for each (env, target) pair
gap_matrix = np.zeros((len(envs), len(targets)))

for i, env in enumerate(envs):
    for j, target in enumerate(targets):
        gaps = [
            float(r["forget_gap_mean"])
            for r in rows
            if r["env"] == env and r["target"] == target
        ]
        if gaps:
            gap_matrix[i, j] = np.mean(gaps)

fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.8))

im = ax.imshow(gap_matrix, cmap=custom_cmap, aspect="auto", vmin=0, vmax=0.20)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Mean Forget Gap", fontsize=FONT_SIZE + 1)

ax.set_xticks(range(len(targets)))
ax.set_xticklabels(target_labels)
ax.set_yticks(range(len(envs)))
ax.set_yticklabels(env_labels)

# Annotate each cell with its numeric value
for i in range(len(envs)):
    for j in range(len(targets)):
        val = gap_matrix[i, j]
        text_color = "white" if val > 0.14 else "black"
        ax.text(
            j,
            i,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=FONT_SIZE + 1,
            color=text_color,
        )

save_fig(fig, "fig2_privacy_gap_heatmap")
plt.close()
