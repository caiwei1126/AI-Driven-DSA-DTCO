import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(os.path.join(HERE, "convergence.csv"))
x = d["eval_idx"]
y = d["error"]
best = d["error"].cummin()

vmin, vmax = float(y.min()), float(y.max())
cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(6, 4))
sc = ax.scatter(x, y, c=y, cmap=cmap, norm=norm, s=26, alpha=0.85, edgecolors="none")
ax.plot(x, best, linewidth=1.5, color="grey", alpha=0.9, label="Best-so-far")

ax.set_xlabel("Evaluation index", fontsize=12)
ax.set_ylabel("Error", fontsize=12)
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()

png_path = os.path.join(HERE, "FigA_convergence_linear_tag0.5.png")
pdf_path = os.path.join(HERE, "FigA_convergence_linear_tag0.5.pdf")
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
plt.close(fig)
print(f"[OK] saved: {png_path}")
