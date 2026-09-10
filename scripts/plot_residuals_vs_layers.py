"""Plot residuals for different approximations of D^-1.

Usage:
    plot_residuals_vs_layers.py --mass=<f> --volume=<vol>

Options:
    --mass=<f>              Mass parameter
    --volume=<vol>          Volume string
"""

import os
import re

import matplotlib.pyplot as plt
import torch
from docopt import docopt


def extract_layers_or_steps(filename):
    try:
        nsteps = int(
            re.search(r"residuals_(\d+)(?:steps|layers)", filename).group(1)
        )
    except:
        raise ValueError(
            f"Number of layers/steps not extractable from filename '{filename}'!"
        )
    return nsteps


# Parse docopt arguments
args = docopt(__doc__)
mass = float(args["--mass"])
vol = args["--volume"]

means_hopping = []
stds_hopping = []
nstepss_hopping = []
means_Clifford = []
stds_Clifford = []
nstepss_Clifford = []
means_restricted = []
stds_restricted = []
nstepss_restricted = []

# Find all nsteps
nstepss = sorted(
    set(extract_layers_or_steps(f) for f in os.listdir("data/residuals"))
)

# Aggregate all available residuals data
for nsteps in nstepss:
    # Hopping expansion
    hopping_path = (
        f"data/residuals/residuals_{nsteps}steps_{vol}_hopping_m{mass:.2f}.pt"
    )
    if os.path.exists(hopping_path):
        data = torch.load(hopping_path, weights_only=True)

        nstepss_hopping.append(nsteps)
        means_hopping.append(torch.mean(data))
        stds_hopping.append(torch.std(data))

    # Clifford model
    clifford_path = (
        f"data/residuals/residuals_{nsteps}layers_{vol}_Clifford_m{mass:.2f}.pt"
    )
    if os.path.exists(clifford_path):
        data = torch.load(clifford_path, weights_only=True)
        nstepss_Clifford.append(nsteps)
        means_Clifford.append(torch.mean(data))
        stds_Clifford.append(torch.std(data))

    # Restricted model
    restricted_path = f"data/residuals/residuals_{nsteps}layers_{vol}_restricted_m{mass:.2f}.pt"
    if os.path.exists(restricted_path):
        data = torch.load(restricted_path, weights_only=True)
        nstepss_restricted.append(nsteps)
        means_restricted.append(torch.mean(data))
        stds_restricted.append(torch.std(data))

# Create plot
plt.figure(figsize=(10, 6))

plt.errorbar(
    nstepss_hopping,
    means_hopping,
    yerr=stds_hopping,
    linestyle="none",
    ecolor="blue",
    capsize=5,
    marker="o",
    markerfacecolor="none",
    label="Hopping expansion",
)
plt.errorbar(
    nstepss_Clifford,
    means_Clifford,
    yerr=stds_Clifford,
    linestyle="none",
    ecolor="orange",
    capsize=5,
    marker="s",
    markerfacecolor="none",
    label="Clifford model",
)
plt.errorbar(
    nstepss_restricted,
    means_restricted,
    yerr=stds_restricted,
    linestyle="none",
    ecolor="green",
    capsize=5,
    marker="D",
    markerfacecolor="none",
    label="Restricted model",
)

plt.xlabel("Number of steps/layers")
plt.ylabel("Approximation quality")
plt.title(f"Approximation quality vs steps/layers (mass={mass:.2f})")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.savefig(
    f"plots/png/residuals/residuals_vs_layers_m{mass:.2f}_{vol}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/residuals/residuals_vs_layers_m{mass:.2f}_{vol}.pdf",
    bbox_inches="tight",
)
