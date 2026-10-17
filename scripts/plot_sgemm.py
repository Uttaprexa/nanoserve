#!/usr/bin/env python3
"""Plot SGEMM results: GFLOP/s vs size, and % of cuBLAS at the largest size.
Usage: python3 scripts/plot_sgemm.py results/sgemm_<gpu>.csv
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

csv = Path(sys.argv[1])
df = pd.read_csv(csv)
gpu = df["gpu"].iloc[0]
order = df.drop_duplicates("kernel_id").sort_values("kernel_id")["kernel"].tolist()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for name in order:
    d = df[df["kernel"] == name].sort_values("M")
    style = "k--" if name == "cublas" else "-o"
    ax1.plot(d["M"], d["gflops"], style, label=name)
ax1.set(xscale="log", xlabel="M = N = K", ylabel="GFLOP/s", title=f"SGEMM throughput on {gpu}")
ax1.set_xticks(sorted(df["M"].unique()), labels=[str(s) for s in sorted(df["M"].unique())])
ax1.grid(alpha=0.3)
ax1.legend()

largest = df[df["M"] == df["M"].max()].sort_values("kernel_id")
bars = ax2.bar(largest["kernel"], largest["pct_cublas"])
ax2.bar_label(bars, fmt="%.1f%%")
ax2.set(ylabel="% of cuBLAS", title=f"% of cuBLAS at {df['M'].max()}^3")
ax2.tick_params(axis="x", rotation=30)

fig.tight_layout()
out = csv.with_suffix(".png")
fig.savefig(out, dpi=150)
print(f"Saved {out}")
