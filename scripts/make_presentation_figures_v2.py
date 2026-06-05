#!/usr/bin/env python3
"""Generate a second set of PPT figures with explicit hard-coded values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "presentation_figures_v2"


COLORS = {
    "blue": "#356DA3",
    "orange": "#E67E22",
    "green": "#2E8B57",
    "red": "#C0392B",
    "gold": "#D4A017",
    "slate": "#34495E",
    "light_blue": "#D9EAF7",
    "light_orange": "#FBE5D6",
    "light_green": "#DDEFE3",
    "light_red": "#F8DDDA",
    "light_gray": "#EEF1F5",
    "dark": "#1F2D3D",
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
        "axes.edgecolor": "#D0D7DE",
        "axes.grid": True,
        "grid.color": "#E9EDF2",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.28,
    })


def savefig(fig, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_core_results_cn():
    # Full LIAR-RAW final comparison.
    metrics = ["Test Accuracy\n/ Micro-F1", "Test Macro-F1", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Sentence\nMacro-F1"]
    cofced = [0.255795, 0.248890, 0.1534, 0.0345, 0.1184, 0.292483]
    fac = [0.280576, 0.270895, 0.1544, 0.0347, 0.1191, 0.266978]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    bars1 = ax.bar(x - width / 2, cofced, width, label="CofCED", color=COLORS["blue"])
    bars2 = ax.bar(x + width / 2, fac, width, label="FaC-CofCED v1", color=COLORS["orange"])

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.006, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Full LIAR-RAW Main Test Results")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.34)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "01_full_test_results_v2.png")


def plot_gain_focus():
    # Keep only the gains worth highlighting on a class presentation slide.
    labels = ["Accuracy /\nMicro-F1", "Macro-F1", "Sentence-level\nselection"]
    gains = [0.024781, 0.022005, -0.025505]
    colors = [COLORS["green"], COLORS["orange"], COLORS["red"]]

    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    bars = ax.bar(labels, gains, color=colors, width=0.52)
    ax.axhline(0, color=COLORS["dark"], linewidth=1.1)
    for bar, val in zip(bars, gains):
        if val >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0012, f"+{val:.4f}", ha="center", va="bottom", fontsize=12)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val - 0.0022, f"{val:.4f}", ha="center", va="top", fontsize=12)

    ax.set_title("What Improved Most")
    ax.set_ylabel("Absolute gain")
    ax.set_ylim(-0.035, 0.03)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "02_performance_gains_v2.png")


def plot_class_delta_cn():
    # Highlight only the most presentation-worthy class gains.
    labels = ["half-true", "mostly-true", "pants-fire"]
    deltas = [0.077752, 0.075296, 0.015292]
    base_f1 = [0.210300, 0.251627, 0.208589]
    new_f1 = [0.288052, 0.326923, 0.223881]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(y, base_f1, color="#D7E3F3", edgecolor="none", height=0.52, label="CofCED")
    ax.barh(y, new_f1, color=COLORS["orange"], edgecolor="none", height=0.34, label="FaC-CofCED v1")

    for yi, base, new, delta in zip(y, base_f1, new_f1, deltas):
        ax.text(new + 0.006, yi, f"+{delta:.3f}", va="center", ha="left", fontsize=11, color=COLORS["dark"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Largest Class-wise Gains")
    ax.set_xlabel("F1-score")
    ax.set_xlim(0, 0.38)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "03_class_f1_changes_v2.png")


def plot_time_and_env():
    # Explicit experiment metadata for a PPT methods/results page.
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "Experiment Setup and Runtime Cost", ha="center", va="center", fontsize=22, weight="bold", color=COLORS["dark"])

    boxes = [
        ((0.06, 0.58), 0.25, 0.20, "Dataset\nLIAR-RAW\nTrain 10065\nVal 1274\nTest 1251", COLORS["light_blue"]),
        ((0.37, 0.58), 0.25, 0.20, "Hardware\nRTX 4080 ~32GB\n128 CPUs\n503 GiB memory", COLORS["light_orange"]),
        ((0.68, 0.58), 0.25, 0.20, "Training setup\nEpochs = 8\nBatch size = 2\nReports per claim = 30", COLORS["light_green"]),
        ((0.06, 0.24), 0.25, 0.20, "CofCED baseline\nTest Macro-F1 = 0.248890\nObserved runtime\n2h 32m 46s", COLORS["light_gray"]),
        ((0.37, 0.24), 0.25, 0.20, "FaC-CofCED v1\nTest Macro-F1 = 0.270895\nFull training runtime\n7h 18m 27s", COLORS["light_red"]),
        ((0.68, 0.24), 0.25, 0.20, "Core gains\nAccuracy +0.024781\nMacro-F1 +0.022005\nROUGE slightly higher", "#F8F1D7"),
    ]

    for (x, y), w, h, text, fc in boxes:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.4,
            edgecolor="#AAB4BE",
            facecolor=fc,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13, color=COLORS["dark"])

    savefig(fig, "04_experiment_setup_and_cost_v2.png")


def plot_summary_card():
    # A PPT cover/supporting figure summarizing the main message.
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bg = FancyBboxPatch((0.03, 0.06), 0.94, 0.86, boxstyle="round,pad=0.02,rounding_size=0.035", linewidth=0, facecolor="#F7F9FC")
    ax.add_patch(bg)
    ax.text(0.08, 0.84, "FaC-CofCED v1", fontsize=28, weight="bold", color=COLORS["dark"], ha="left")
    ax.text(0.08, 0.76, "Fact-level Conflict-aware CofCED", fontsize=16, color=COLORS["slate"], ha="left")

    ax.text(0.08, 0.62, "Key takeaway", fontsize=18, weight="bold", color=COLORS["dark"], ha="left")
    ax.text(0.08, 0.54, "On full LIAR-RAW, FaC-CofCED v1 improves over the CofCED baseline", fontsize=15, color=COLORS["slate"], ha="left")
    ax.text(0.08, 0.47, "Test Macro-F1 rises from 0.248890 to 0.270895", fontsize=18, color=COLORS["orange"], ha="left", weight="bold")
    ax.text(0.08, 0.40, "Accuracy / Micro-F1 rises from 0.255795 to 0.280576", fontsize=16, color=COLORS["blue"], ha="left")

    cards = [
        (0.63, 0.67, 0.26, 0.14, "Macro-F1\n+0.022005", COLORS["light_red"], COLORS["orange"]),
        (0.63, 0.48, 0.26, 0.14, "Accuracy / Micro-F1\n+0.024781", COLORS["light_blue"], COLORS["blue"]),
        (0.63, 0.29, 0.26, 0.14, "ROUGE-1 / 2 / L\n+0.10 / +0.02 / +0.07", COLORS["light_green"], COLORS["green"]),
    ]
    for x, y, w, h, text, fc, tc in cards:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=0, facecolor=fc)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=15, color=tc, weight="bold")

    ax.text(0.08, 0.20, "Method keywords: support / refute / conflict-aware evidence modeling", fontsize=14, color=COLORS["slate"], ha="left")
    ax.text(0.08, 0.14, "Suitable for a title slide or final results slide.", fontsize=13, color="#667085", ha="left")
    savefig(fig, "05_summary_card_v2.png")


def main():
    setup_style()
    plot_core_results_cn()
    plot_gain_focus()
    plot_class_delta_cn()
    plot_time_and_env()
    plot_summary_card()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
