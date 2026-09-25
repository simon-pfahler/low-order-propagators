"""Create the history comparison figure of the paper."""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def parse_history_file(filepath):
    iterations = []
    train_costs = []
    test_costs = []
    test_iterations = []

    with open(filepath, "r") as f:
        for line in f:
            words = line.strip().split()
            if words[0][0] == "#":
                continue
            if len(words) >= 2:
                iterations.append(int(words[0]))
                train_costs.append(float(words[1]))
            if len(words) == 3:
                test_iterations.append(int(words[0]))
                test_costs.append(float(words[2]))
    return (
        np.array(iterations),
        np.array(train_costs),
        np.array(test_costs),
        np.array(test_iterations),
    )


plt.style.use("./scripts/iclr2027.mplstyle")

sys.path.insert(0, "scripts")

layerss = [1, 4]
action = "WilsonQuenched"
lattice_size_str = "8c16"
mass = "-0.80"

fig, ax = plt.subplots(1, 2, figsize=(9, 2.3))
plt.subplots_adjust(wspace=0.15)
for idx in range(2):
    layers = layerss[idx]

    history_path_HC = f"data/histories/history_{layers}layers_{action}_{lattice_size_str}_HC_m{mass}.txt"
    if not os.path.exists(history_path_HC):
        raise ValueError(
            f"No history file found for model '{layers}layers_{action}_{lattice_size_str}_HC' and mass={mass}"
        )
    history_path_HL = f"data/histories/history_{layers}layers_{action}_{lattice_size_str}_HL_m{mass}.txt"
    if not os.path.exists(history_path_HL):
        raise ValueError(
            f"No history file found for model '{layers}layers_{action}_{lattice_size_str}_HL' and mass={mass}"
        )

    (
        iterations_HC,
        train_costs_HC,
        test_costs_HC,
        test_iterations_HC,
    ) = parse_history_file(history_path_HC)
    (
        iterations_HL,
        train_costs_HL,
        test_costs_HL,
        test_iterations_HL,
    ) = parse_history_file(history_path_HL)

    # Create plot
    ax[idx].plot(
        iterations_HC,
        train_costs_HC,
        color="#ee7733",
        linestyle="-",
        linewidth=1,
        zorder=3,
    )
    ax[idx].scatter(
        test_iterations_HC, test_costs_HC, color="#ee7733", s=10, zorder=3
    )
    ax[idx].plot(
        iterations_HL,
        train_costs_HL,
        color="#009988",
        linestyle="--",
        linewidth=1,
        zorder=2,
    )
    ax[idx].scatter(
        test_iterations_HL,
        test_costs_HL,
        marker="D",
        color="#009988",
        s=10,
        zorder=2,
    )

    hc_handle = Line2D(
        [-1, 0, 1],
        [0, 0, 0],
        color="#ee7733",
        linestyle="-",
        linewidth=1,
        marker="o",
        markersize=np.sqrt(10),
    )
    hc_handle.set_markevery([1])

    hl_handle = Line2D(
        [-1.5, 0, 1.5],
        [0, 0, 0],
        color="#009988",
        linestyle="--",
        linewidth=1,
        marker="D",
        markersize=np.sqrt(10),
    )
    hl_handle.set_markevery([1])

    ax[idx].legend(
        [hc_handle, hl_handle], ["HC model", "HL model"], handlelength=2
    )

    ax[idx].set_xlabel("Iteration")
    if idx == 0:
        ax[idx].set_ylabel("Cost")
    ax[idx].set_title(
        f"Training history for {layers} layers, " rf"$m={float(mass):.1f}$"
    )
    ax[idx].grid(True, which="both", alpha=0.5)
    ax[idx].set_yscale("log")

os.makedirs("plots/histories", exist_ok=True)

plt.savefig(
    f"plots/histories/history_comparison.pdf",
    bbox_inches="tight",
)
