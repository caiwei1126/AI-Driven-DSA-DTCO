import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET_CCD = 54
X_MIN, X_MAX = 50.0, 80.0
Y_MIN, Y_MAX = 110.0, 130.0

PARAM_SETS = {
    "phi_s_0.0101": {"phi_s": 0.0101, "phi_b": 0.0461, "lam_s": 2.4515, "lam_b": 1.5720},
    "phi_s_0.0115": {"phi_s": 0.0115, "phi_b": 0.0664, "lam_s": 4.7921, "lam_b": 3.9092},
    "phi_s_0.0106": {"phi_s": 0.0106, "phi_b": 0.0215, "lam_s": 9.3012, "lam_b": 3.5918},
}

def load(tag):
    d = pd.read_csv(os.path.join(HERE, f"{tag}.csv"))
    return d[d["diameter_nm"].between(X_MIN, X_MAX) & d["length_nm"].between(Y_MIN, Y_MAX)]

tags = list(PARAM_SETS)
data = {t: load(t) for t in tags}
vmax = max(d["abs_error_nm"].max() for d in data.values())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
for ax, tag in zip(axes, tags):
    d = data[tag]
    sc = ax.scatter(d["diameter_nm"], d["length_nm"], s=30, c=d["abs_error_nm"],
                    cmap="plasma", vmin=0, vmax=vmax)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    p = PARAM_SETS[tag]
    ax.set_title(f"$\\phi_s$={p['phi_s']}, $\\phi_b$={p['phi_b']},\n"
                 f"$\\Lambda_s$={p['lam_s']}, $\\Lambda_b$={p['lam_b']}", fontsize=11)
    ax.set_xlabel("Diameter (nm)", fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
axes[0].set_ylabel("Length (nm)", fontsize=13)
fig.suptitle(f"CCD = {TARGET_CCD} nm", fontsize=16)
fig.subplots_adjust(left=0.055, right=0.90, top=0.78, bottom=0.13, wspace=0.08)
cax = fig.add_axes([0.915, 0.15, 0.014, 0.60])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label("Error (nm)", fontsize=13)
fig.savefig(os.path.join(HERE, f"parameter_set_comparison_ccd_{TARGET_CCD}nm.png"), dpi=300)
plt.show()
