import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(os.path.join(HERE, "error_distribution.csv"))
labels = ["Training", "Validation", "Prediction"]
groups = [d.loc[d["group"] == lab, "error"].tolist() for lab in labels]

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

xpos = np.arange(1, len(groups) + 1)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(groups)]

rain_offset = -0.22
jitter_sd = 0.05

bp = ax.boxplot(
    groups,
    positions=xpos,
    widths=0.18,
    showfliers=False,
    whis=(0, 100),
    patch_artist=True,
    medianprops=dict(linewidth=1.6),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
for i in range(len(groups)):
    c = colors[i]
    bp["boxes"][i].set_facecolor(c)
    bp["boxes"][i].set_alpha(0.35)
    bp["boxes"][i].set_edgecolor(c)
    for artist in (
        bp["medians"][i],
        bp["whiskers"][2 * i], bp["whiskers"][2 * i + 1],
        bp["caps"][2 * i], bp["caps"][2 * i + 1],
    ):
        artist.set_color(c)

rng = np.random.default_rng(0)
for i, g in enumerate(groups):
    y = np.asarray(g, float)
    x = (xpos[i] + rain_offset) + rng.normal(0, jitter_sd, size=len(y))
    ax.scatter(x, y, s=26, alpha=0.85, color=colors[i], zorder=3)

ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=14)
ax.set_ylabel("Error (nm)", fontsize=14)
ax.set_ylim(bottom=0)
ax.grid(False)
plt.tight_layout()
png_path = os.path.join(HERE, "error_distribution_box_scatter.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] saved: {png_path}")
