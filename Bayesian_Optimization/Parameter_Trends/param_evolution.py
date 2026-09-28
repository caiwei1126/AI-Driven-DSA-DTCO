import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(os.path.join(HERE, "param_evolution.csv"))

params = [
    ("phi_s", r"$\phi_s$"),
    ("phi_b", r"$\phi_b$"),
    ("Lambda_s", r"$\Lambda_s$"),
    ("Lambda_b", r"$\Lambda_b$"),
]
bounds = {
    "phi_s": (0, 0.5),
    "phi_b": (0, 0.5),
    "Lambda_s": (0.1, 10.0),
    "Lambda_b": (0.1, 10.0),
}

vmin = d["error"].quantile(0.02)
vmax = d["error"].quantile(0.98)
cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
axes = axes.ravel()

for ax, (col, lab) in zip(axes, params):
    sc = ax.scatter(
        d["eval_idx"], d[col],
        c=d["error"], cmap=cmap, norm=norm,
        s=26, alpha=0.9, edgecolors="none",
    )
    ax.set_ylabel(lab, fontsize=13)

    if col in bounds and bounds[col] is not None:
        lo, hi = bounds[col]
        ax.axhline(lo, linestyle="--", linewidth=1.0, color="orange", alpha=0.8)
        ax.axhline(hi, linestyle="--", linewidth=1.0, color="orange", alpha=0.8)

    ax.grid(alpha=0.2)

axes[2].set_xlabel("Evaluation index", fontsize=13)
axes[3].set_xlabel("Evaluation index", fontsize=13)

fig.tight_layout(rect=[0, 0, 0.88, 0.96])

cax = fig.add_axes([0.90, 0.16, 0.02, 0.70])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cbar.set_label(r"Error", fontsize=12)

png_path = os.path.join(HERE, "FigB_param_evolution_tag0.5.png")
pdf_path = os.path.join(HERE, "FigB_param_evolution_tag0.5.pdf")
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
plt.close(fig)
print(f"[OK] saved: {png_path}")
