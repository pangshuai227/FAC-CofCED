#!/usr/bin/env python3
"""Generate PPT-ready figures for CofCED / FaC-CofCED presentation.

All plotted values are intentionally hard-coded in this script so the figures
are self-contained and easy to audit for a classroom report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "presentation_figures"


COLORS = {
    "cofced": "#4C78A8",
    "fac": "#F58518",
    "gain": "#54A24B",
    "drop": "#D64F4F",
    "gray": "#6B7280",
    "light_blue": "#D9E8F5",
    "light_orange": "#FCE2C4",
    "light_green": "#DCEFD8",
    "light_red": "#F8D7DA",
    "dark": "#1F2937",
}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
    })


def savefig(fig, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_value_labels(ax, bars, fmt="{:.2f}", dy=0.01):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS["dark"],
        )


def plot_main_metrics():
    # Full LIAR-RAW held-out test metrics.
    metrics = ["Accuracy / Micro-F1", "Macro-F1", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
    cofced = [0.255795, 0.248890, 15.34 / 100, 3.45 / 100, 11.84 / 100]
    fac_v1 = [0.280576, 0.270895, 15.44 / 100, 3.47 / 100, 11.91 / 100]

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.4))
    bars1 = ax.bar(x - width / 2, cofced, width, label="CofCED baseline", color=COLORS["cofced"])
    bars2 = ax.bar(x + width / 2, fac_v1, width, label="FaC-CofCED v1", color=COLORS["fac"])

    add_value_labels(ax, bars1, fmt="{:.3f}", dy=0.006)
    add_value_labels(ax, bars2, fmt="{:.3f}", dy=0.006)

    ax.set_title("Full LIAR-RAW Test Performance")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.34)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "01_main_test_metrics.png")


def plot_improvement_summary():
    # Deltas are FaC-CofCED v1 minus CofCED baseline on full LIAR-RAW test.
    metrics = ["Accuracy /\nMicro-F1", "Macro-F1", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Sent\nMacro-F1"]
    deltas = [0.024781, 0.022005, 0.10 / 100, 0.02 / 100, 0.07 / 100, -0.025505]
    colors = [COLORS["gain"] if v >= 0 else COLORS["drop"] for v in deltas]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    bars = ax.bar(metrics, deltas, color=colors, width=0.58)
    ax.axhline(0, color=COLORS["dark"], linewidth=1.2)
    for bar, value in zip(bars, deltas):
        label = f"{value:+.3f}"
        y = value + (0.002 if value >= 0 else -0.004)
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va=va, fontsize=11)

    ax.set_title("FaC-CofCED v1 Improvement over CofCED Baseline")
    ax.set_ylabel("Absolute score change")
    ax.set_ylim(-0.035, 0.035)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "02_improvement_summary.png")


def plot_class_f1():
    # Class-wise held-out test F1.
    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    cofced_f1 = [0.208589, 0.305026, 0.253456, 0.210300, 0.251627, 0.264339]
    fac_f1 = [0.223881, 0.282655, 0.244792, 0.288052, 0.326923, 0.259067]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x - width / 2, cofced_f1, width, label="CofCED baseline", color=COLORS["cofced"])
    ax.bar(x + width / 2, fac_f1, width, label="FaC-CofCED v1", color=COLORS["fac"])

    for idx, (base, fac) in enumerate(zip(cofced_f1, fac_f1)):
        delta = fac - base
        color = COLORS["gain"] if delta >= 0 else COLORS["drop"]
        ax.text(idx, max(base, fac) + 0.018, f"{delta:+.3f}", ha="center", va="bottom", color=color, fontsize=10)

    ax.set_title("Class-wise Test F1 Comparison")
    ax.set_ylabel("F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 0.38)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "03_classwise_f1.png")


def plot_validation_curve():
    # Full v1 validation trajectory and original CofCED best validation Macro-F1.
    epochs = np.arange(1, 9)
    fac_current_val_macrof1 = [0.164941, 0.193474, 0.229380, 0.235826, 0.259630, 0.264436, 0.270046, 0.269993]
    fac_best_so_far = [0.164941, 0.193474, 0.229380, 0.235826, 0.259630, 0.264436, 0.270046, 0.270046]
    cofced_best = 0.263725

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.plot(epochs, fac_current_val_macrof1, marker="o", linewidth=2.4, color=COLORS["fac"], label="FaC-CofCED v1 current")
    ax.plot(epochs, fac_best_so_far, marker="s", linewidth=2.0, color=COLORS["gain"], label="FaC-CofCED v1 best so far")
    ax.axhline(cofced_best, linestyle="--", linewidth=2.0, color=COLORS["cofced"], label="CofCED baseline best")
    ax.scatter([7], [0.270046], s=120, color=COLORS["fac"], edgecolor="white", linewidth=1.5, zorder=5)
    ax.text(7.05, 0.273, "best epoch 7\n0.270046", fontsize=10, color=COLORS["dark"])

    ax.set_title("Validation Macro-F1 Trajectory on Full LIAR-RAW")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_xticks(epochs)
    ax.set_ylim(0.15, 0.285)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "04_validation_macro_f1_curve.png")


def plot_dataset_distribution():
    # LIAR-RAW label distribution used in our experiments.
    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    train = [812, 1958, 1611, 2087, 1950, 1647]
    val = [115, 259, 236, 244, 251, 169]
    test = [86, 249, 210, 263, 238, 205]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.bar(x, train, label="Train", color=COLORS["cofced"])
    ax.bar(x, val, bottom=train, label="Validation", color=COLORS["fac"])
    ax.bar(x, test, bottom=np.array(train) + np.array(val), label="Test", color=COLORS["gain"])

    totals = np.array(train) + np.array(val) + np.array(test)
    for idx, total in enumerate(totals):
        ax.text(idx, total + 55, str(total), ha="center", va="bottom", fontsize=10)

    ax.set_title("LIAR-RAW Label Distribution")
    ax.set_ylabel("Number of claims")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, max(totals) * 1.18)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "05_dataset_distribution.png")


def draw_box(ax, xy, width, height, text, facecolor, edgecolor="#9CA3AF", fontsize=11):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["dark"])
    return box


def draw_arrow(ax, start, end, color="#6B7280"):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.7, color=color)
    ax.add_patch(arrow)


def plot_method_diagram():
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "FaC-CofCED v1: Conflict-aware Evidence Distillation", ha="center", va="center", fontsize=20, weight="bold", color=COLORS["dark"])

    draw_box(ax, (0.05, 0.66), 0.16, 0.12, "Claim", COLORS["light_blue"], fontsize=12)
    draw_box(ax, (0.05, 0.42), 0.16, 0.12, "Raw reports", COLORS["light_blue"], fontsize=12)
    draw_box(ax, (0.28, 0.54), 0.18, 0.14, "CofCED\nreport selection", "#EAF2FB", fontsize=11)
    draw_box(ax, (0.52, 0.54), 0.20, 0.14, "Sentence selection\n+ FAC selector", COLORS["light_orange"], fontsize=11)
    draw_box(ax, (0.78, 0.54), 0.17, 0.14, "Selected\nevidence", "#FFF4E6", fontsize=11)

    draw_box(ax, (0.31, 0.20), 0.22, 0.14, "FAC features\nalignment / support\nrefute / neutral / conflict", COLORS["light_green"], fontsize=10)
    draw_box(ax, (0.60, 0.20), 0.24, 0.14, "Evidence groups\nsupport_repr\nrefute_repr\nconflict_repr", COLORS["light_green"], fontsize=10)
    draw_box(ax, (0.78, 0.78), 0.17, 0.11, "Veracity\nclassifier", COLORS["light_red"], fontsize=11)
    draw_box(ax, (0.58, 0.78), 0.14, 0.11, "Prediction\n6 labels", "#F9E2E2", fontsize=11)

    draw_arrow(ax, (0.21, 0.72), (0.28, 0.62))
    draw_arrow(ax, (0.21, 0.48), (0.28, 0.60))
    draw_arrow(ax, (0.46, 0.61), (0.52, 0.61))
    draw_arrow(ax, (0.72, 0.61), (0.78, 0.61))
    draw_arrow(ax, (0.13, 0.66), (0.34, 0.34), color=COLORS["gain"])
    draw_arrow(ax, (0.13, 0.42), (0.34, 0.34), color=COLORS["gain"])
    draw_arrow(ax, (0.53, 0.27), (0.60, 0.27), color=COLORS["gain"])
    draw_arrow(ax, (0.72, 0.68), (0.81, 0.78), color=COLORS["drop"])
    draw_arrow(ax, (0.72, 0.34), (0.80, 0.78), color=COLORS["gain"])
    draw_arrow(ax, (0.78, 0.835), (0.72, 0.835), color=COLORS["drop"])

    ax.text(0.50, 0.07, "Main idea: keep CofCED's coarse-to-fine pipeline, then add support/refute/conflict-aware signals before final classification.", ha="center", va="center", fontsize=12, color=COLORS["gray"])
    savefig(fig, "06_method_diagram.png")


def main():
    setup_style()
    plot_main_metrics()
    plot_improvement_summary()
    plot_class_f1()
    plot_validation_curve()
    plot_dataset_distribution()
    plot_method_diagram()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
