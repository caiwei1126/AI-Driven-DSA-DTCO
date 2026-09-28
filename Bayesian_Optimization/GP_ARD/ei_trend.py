import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(os.path.join(HERE, "ei_trend.csv"))
x = d["evaluation_index"].to_numpy()
y = d["error_nm"].to_numpy(dtype=float)
best = np.minimum.accumulate(y)

y_valid = y[y < 1000]
vmin, vmax = float(y_valid.min()), float(y_valid.max())
cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

fig, (ax_top, ax) = plt.subplots(
    2, 1, figsize=(6, 4), sharex=True,
    gridspec_kw=dict(height_ratios=[1, 5], hspace=0.08),
)
kw = dict(cmap=cmap, norm=norm, s=26, alpha=0.85, edgecolors="none")

fail = y >= 1000
ok = ~fail

sc = ax.scatter(x[ok], y[ok], c=y[ok], **kw)
ax.plot(x, best, linewidth=1.5, color="grey", alpha=0.9, label="Best-so-far")
ax.set_ylim(vmin - 0.03 * (vmax - vmin), vmax + 0.06 * (vmax - vmin))

if fail.any():
    ax_top.scatter(x[fail], np.full(fail.sum(), 1000.0),
                   c=np.full(fail.sum(), 1000.0), **kw)
ax_top.set_ylim(950, 1050)
ax_top.set_yticks([1000])

ax_top.spines["bottom"].set_visible(False)
ax.spines["top"].set_visible(False)
ax_top.tick_params(bottom=False)
dlen = 0.012
ax_top.plot((-dlen, +dlen), (-dlen, +dlen), transform=ax_top.transAxes,
            color="k", clip_on=False, linewidth=1)
ax_top.plot((1 - dlen, 1 + dlen), (-dlen, +dlen), transform=ax_top.transAxes,
            color="k", clip_on=False, linewidth=1)
ax.plot((-dlen, +dlen), (1 - dlen, 1 + dlen), transform=ax.transAxes,
        color="k", clip_on=False, linewidth=1)
ax.plot((1 - dlen, 1 + dlen), (1 - dlen, 1 + dlen), transform=ax.transAxes,
        color="k", clip_on=False, linewidth=1)

ax.set_xlabel("Evaluation index", fontsize=12)
fig.supylabel("Error", fontsize=12)

fig.colorbar(sc, ax=[ax_top, ax], fraction=0.046, pad=0.04)
fig.tight_layout()

png_path = os.path.join(HERE, "GP_Matern52_ARD_EI_trend_v2.png")
pdf_path = os.path.join(HERE, "GP_Matern52_ARD_EI_trend_v2.pdf")
fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)
plt.close(fig)
print(f"[OK] saved: {png_path}")
