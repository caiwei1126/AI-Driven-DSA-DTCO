import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHADE_TAIL_K = 12

d = pd.read_csv(os.path.join(HERE, "acquisition_values.csv"))
n = int(d["eval_idx"].max())

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.axvspan(max(1, n - SHADE_TAIL_K + 1), n, color="0.8", alpha=0.35, zorder=0)

trend = d["acquisition_value"].rolling(window=3, center=True, min_periods=1).mean()

ax.scatter(d["eval_idx"], d["acquisition_value"], s=18, alpha=0.9, zorder=2)
ax.plot(d["eval_idx"], trend, color="grey", linewidth=1.8, alpha=0.95,
        zorder=3, label="Trend")
ax.legend(frameon=False, fontsize=12)

ax.set_xlabel("Evaluation index", fontsize=13)
ax.set_ylabel("Acquisition value", fontsize=13)
ax.grid(False)

fig.tight_layout()
png_path = os.path.join(HERE, "Fig_AcqValue_at_sampled_points.png")
pdf_path = os.path.join(HERE, "Fig_AcqValue_at_sampled_points.pdf")
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
plt.close(fig)
print(f"[OK] saved: {png_path}")
