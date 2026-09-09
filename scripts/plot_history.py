"""Plot training history from saved history file.

Usage:
    plot_history.py --model=<name> --mass=<mass>

Options:
    --model=<name>          Name of the model json file (in `models/{name}.json`)
    --mass=<mass>           Mass parameter value
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
model_name = args["--model"]
mass = args["--mass"]

history_path = f"data/histories/history_{model_name}_m{mass}.txt"
if not os.path.exists(history_path):
    raise ValueError(
        f"No history file found for model '{model_name}' and mass={mass}"
    )

iterations, train_costs, test_costs, test_iterations = parse_history_file(
    history_path
)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_costs, "b-", label="Training cost", linewidth=1)
plt.scatter(test_iterations, test_costs, color="red", label="Test cost", s=10)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title(f"Training History for {model_name}, mass={mass}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")

os.makedirs("plots/png/histories", exist_ok=True)
os.makedirs("plots/pdf/histories", exist_ok=True)

plt.savefig(
    f"plots/png/histories/history_{model_name}_m{mass}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/histories/history_{model_name}_m{mass}.pdf",
    bbox_inches="tight",
)
