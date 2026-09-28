import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(os.path.join(HERE, "ccd_response_vs_cdy.csv"))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(d["cdy_nm"], d["ccd_nm"], marker="o", linestyle="-", color="orange")
ax.set_xlabel("CDY (nm)", fontsize=12)
ax.set_ylabel("CCD (nm)", fontsize=12)
ax.set_title("CCD vs CDY (fixed CDX = 68 nm)", fontsize=13)
ax.grid(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "ccd_response_vs_cdy.png"), dpi=300)
plt.show()
