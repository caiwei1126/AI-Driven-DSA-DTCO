import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(os.path.join(HERE, "ccd_response_vs_cdx.csv"))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(d["cdx_nm"], d["ccd_nm"], marker="o", linestyle="-", color="orange")
ax.set_xlabel("CDX (nm)", fontsize=12)
ax.set_ylabel("CCD (nm)", fontsize=12)
ax.set_title("CCD vs CDX (fixed CDY = 127.5 nm)", fontsize=13)
ax.grid(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "ccd_response_vs_cdx.png"), dpi=300)
plt.show()
