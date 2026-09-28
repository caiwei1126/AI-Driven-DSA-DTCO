import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET_CCD = 52
X_MIN, X_MAX = 50.0, 80.0
Y_MIN, Y_MAX = 110.0, 130.0

d = pd.read_csv(os.path.join(HERE, f"template_error_ccd_{TARGET_CCD}nm.csv"))
d = d[d["diameter_nm"].between(X_MIN, X_MAX) & d["length_nm"].between(Y_MIN, Y_MAX)]

fig, ax = plt.subplots(figsize=(6, 4))
sc = ax.scatter(d["diameter_nm"], d["length_nm"], s=30, c=d["abs_error_nm"], cmap="plasma")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel("Diameter (nm)", fontsize=14)
ax.set_ylabel("Length (nm)", fontsize=14)
ax.set_title(f"CCD = {TARGET_CCD} nm", fontsize=16)
cbar = fig.colorbar(sc)
cbar.set_label("Error (nm)", fontsize=14)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(HERE, f"template_error_ccd_{TARGET_CCD}nm.png"), dpi=300)
plt.show()
