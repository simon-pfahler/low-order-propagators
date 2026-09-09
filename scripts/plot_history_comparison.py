"""Plot training history for comparison between Clifford and 4x4 models.

Usage:
    plot_history_comparsion.py --layers=<n> --volume=<vol> --mass=<m>

Options:
    --layers=<n>    Number of layers
    --volume=<vol>  Volume string
    --mass=<m>      Mass parameter value
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt


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


matplotlib.use("Agg")

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
layers = args["--layers"]
vol = args["--volume"]
mass = args["--mass"]

history_path_Clifford = (
    f"data/histories/history_{layers}layers_{vol}_Clifford_m{mass}.txt"
)
if not os.path.exists(history_path_Clifford):
    raise ValueError(
        f"No history file found for model '{layers}layers_{vol}_Clifford' and mass={mass}"
    )
history_path_4x4 = (
    f"data/histories/history_{layers}layers_{vol}_4x4_m{mass}.txt"
)
if not os.path.exists(history_path_4x4):
    raise ValueError(
        f"No history file found for model '{layers}layers_{vol}_4x4' and mass={mass}"
    )


(
    iterations_Clifford,
    train_costs_Clifford,
    test_costs_Clifford,
    test_iterations_Clifford,
) = parse_history_file(history_path_Clifford)
(
    iterations_4x4,
    train_costs_4x4,
    test_costs_4x4,
    test_iterations_4x4,
) = parse_history_file(history_path_4x4)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(
    iterations_Clifford,
    train_costs_Clifford,
    "b-",
    label="Clifford model",
    linewidth=1,
)
plt.scatter(test_iterations_Clifford, test_costs_Clifford, color="b", s=10)
plt.plot(iterations_4x4, train_costs_4x4, "r-", label="4x4 model", linewidth=1)
plt.scatter(test_iterations_4x4, test_costs_4x4, color="r", s=10)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title(
    f"Training History for {layers} layers, {vol}, mass={mass}\n"
    f"Clifford vs 4x4 Model"
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")

os.makedirs("plots/png/histories", exist_ok=True)
os.makedirs("plots/pdf/histories", exist_ok=True)

plt.savefig(
    f"plots/png/histories/history_comparison_{layers}layers_{vol}_m{mass}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/histories/history_comparison_{layers}layers_{vol}_m{mass}.pdf",
    bbox_inches="tight",
)
