"""Plot training history for comparison between HC and HL models.

Usage:
    plot_history_comparsion.py --layers=<n> --action=<name> --lattice_size=<str> --mass=<m>

Options:
    --layers=<n>            Number of layers
    --action=<name>         Action name ("WilsonQuenched", "WilsonDynamic" or "Haar")
    --lattice_size=<str>    Lattice size string (e.g. "8c16")
    --mass=<mass>           Mass parameter value
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
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

# Parse docopt arguments
args = docopt(__doc__)
layers = args["--layers"]
action = args["--action"]
lattice_size_str = args["--lattice_size"]
mass = args["--mass"]

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
plt.figure(figsize=(4.5, 2.3))
plt.plot(
    iterations_HC,
    train_costs_HC,
    color="#ee7733",
    linestyle="-",
    linewidth=1,
    zorder=3,
)
plt.scatter(test_iterations_HC, test_costs_HC, color="#ee7733", s=10, zorder=3)
plt.plot(
    iterations_HL,
    train_costs_HL,
    color="#009988",
    linestyle="--",
    linewidth=1,
    zorder=2,
)
plt.scatter(
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
hc_handle.set_markevery([1])  # Show marker only at the middle point

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

plt.legend([hc_handle, hl_handle], ["HC model", "HL model"], handlelength=2)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title(
    f"Training History for {layers} layers, {action}\n"
    r"$8^3\times16$"
    rf", $m={float(mass):.1f}$, "
    f"HC vs HL Model"
)
plt.grid(True, which="both", alpha=0.5)
plt.yscale("log")

os.makedirs("plots/histories", exist_ok=True)

plt.savefig(
    f"plots/histories/history_comparison_{layers}layers_{action}_{lattice_size_str}_m{mass}.pdf",
    bbox_inches="tight",
)
