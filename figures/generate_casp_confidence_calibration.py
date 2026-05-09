#!/usr/bin/env python3
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-casp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/20260506_rees46_semantic_confidence_full/rees46_semantic_confidence_audit_per_user.csv"
OUT_CSV = ROOT / "paper/figures/casp_confidence_calibration.csv"
OUT_PDF = ROOT / "paper/figures/casp_confidence_calibration.pdf"
OUT_PNG = ROOT / "paper/figures/casp_confidence_calibration.png"

FEATURE = "top_semid_share"
K = 100
SELECTED_THRESHOLD = 0.545455


def summarize(part):
    n = int(len(part))
    gross = int(part[f"gross@{K}"].sum())
    cann = int(part[f"cannibal@{K}"].sum())
    net = int(part[f"net@{K}"].sum())
    return {
        "n_users": n,
        "mean_top_semid_share": float(part[FEATURE].mean()) if n else 0.0,
        f"gross@{K}": gross,
        f"cannibal@{K}": cann,
        f"net@{K}": net,
        f"gross_per_1000@{K}": 1000.0 * gross / n if n else 0.0,
        f"cannibal_per_1000@{K}": 1000.0 * cann / n if n else 0.0,
        f"net_per_1000@{K}": 1000.0 * net / n if n else 0.0,
        f"ratio@{K}": float(cann / gross) if gross else float("nan"),
    }


def main():
    df = pd.read_csv(INPUT)
    val = df[df["phase"] == "val"].copy()
    _, bins = pd.qcut(val[FEATURE], q=4, retbins=True, duplicates="drop")
    bins[0] = min(0.0, bins[0]) - 1e-9
    bins[-1] = max(1.0, bins[-1]) + 1e-9

    rows = []
    for phase in ["val", "test"]:
        phase_df = df[df["phase"] == phase].copy()
        phase_df["confidence_bin"] = pd.cut(phase_df[FEATURE], bins=bins, include_lowest=True)
        for bin_id, (label, part) in enumerate(phase_df.groupby("confidence_bin", observed=True), start=1):
            row = {
                "phase": "validation" if phase == "val" else "test",
                "bin_id": bin_id,
                "bin": str(label),
            }
            row.update(summarize(part))
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    plt.rcParams.update({
        "font.size": 8,
        "font.family": "DejaVu Serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), constrained_layout=True)

    colors = {"validation": "#1f77b4", "test": "#d62728"}
    markers = {"validation": "o", "test": "s"}
    for phase in ["validation", "test"]:
        part = out[out["phase"] == phase]
        axes[0].plot(
            part["mean_top_semid_share"],
            part[f"net_per_1000@{K}"],
            marker=markers[phase],
            linewidth=1.6,
            markersize=4,
            color=colors[phase],
            label=phase,
        )
    axes[0].axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    axes[0].axvline(SELECTED_THRESHOLD, color="#444444", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Mean top_semid_share")
    axes[0].set_ylabel("Net@100 per 1K users")
    axes[0].legend(frameon=False, loc="upper left")

    test = out[out["phase"] == "test"]
    x = test["mean_top_semid_share"]
    axes[1].plot(x, test[f"gross_per_1000@{K}"], marker="o", linewidth=1.6, markersize=4, label="gross")
    axes[1].plot(x, test[f"cannibal_per_1000@{K}"], marker="s", linewidth=1.6, markersize=4, label="cannibal")
    axes[1].axvline(SELECTED_THRESHOLD, color="#444444", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Mean top_semid_share")
    axes[1].set_ylabel("Test count per 1K users")
    axes[1].legend(frameon=False, loc="upper left")

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG)
    print(OUT_CSV)
    print(OUT_PDF)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
