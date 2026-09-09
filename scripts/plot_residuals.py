"""Plot residuals for different approximations of D^-1.

Usage:
    plot_residuals.py --nsteps=<n> --volume=<vol>

Options:
    --nsteps=<n>            Number of steps in the hopping expansion
    --volume=<vol>          Volume string
"""

import os
import re

import matplotlib.pyplot as plt
import torch
from docopt import docopt


def extract_mass(filename):
    try:
        mass = float(re.search(r"_m([+-]?\d+\.?\d*)\.pt", filename).group(1))
    except:
        raise ValueError(f"Mass not extractable from filename '{filename}'!")
    return mass


# Parse docopt arguments
args = docopt(__doc__)
nsteps = int(args["--nsteps"])
vol = args["--volume"]

means_hopping = []
stds_hopping = []
masses_hopping = []
means_Clifford = []
stds_Clifford = []
masses_Clifford = []
means_restricted = []
stds_restricted = []
masses_restricted = []

# Find all masses
masses = sorted(set(extract_mass(f) for f in os.listdir("data/residuals")))

# Aggregate all available residuals data
for mass in masses:
    # Hopping expansion
    hopping_path = f"data/residuals/residuals_{nsteps}steps_{vol}_hopping_m{mass:.2f}.pt"
    if os.path.exists(hopping_path):
        data = torch.load(hopping_path, weights_only=True)

        masses_hopping.append(mass)
        means_hopping.append(torch.mean(data))
        stds_hopping.append(torch.std(data))

    # Clifford model
    clifford_path = f"data/residuals/residuals_{nsteps}layers_{vol}_Clifford_m{mass:.2f}.pt"
    if os.path.exists(clifford_path):
        data = torch.load(clifford_path, weights_only=True)
        masses_Clifford.append(mass)
        means_Clifford.append(torch.mean(data))
        stds_Clifford.append(torch.std(data))

    # Restricted model
    restricted_path = f"data/residuals/residuals_{nsteps}layers_{vol}_restricted_m{mass:.2f}.pt"
    if os.path.exists(restricted_path):
        data = torch.load(restricted_path, weights_only=True)
        masses_restricted.append(mass)
        means_restricted.append(torch.mean(data))
        stds_restricted.append(torch.std(data))

# Create plot
plt.figure(figsize=(10, 6))

plt.errorbar(
    masses_hopping,
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
    masses_Clifford,
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
    masses_restricted,
    means_restricted,
    yerr=stds_restricted,
    linestyle="none",
    ecolor="green",
    capsize=5,
    marker="D",
    markerfacecolor="none",
    label="Restricted model",
)

plt.xlabel("Mass")
plt.ylabel("Approximation Quality")
plt.title(f"Approximation Quality vs Mass (nlayers={nsteps})")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.ylim(0, 1.1 * max(means_Clifford + means_restricted))

plt.savefig(
    f"plots/png/residuals/residuals_{nsteps}steps_{vol}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/residuals/residuals_{nsteps}steps_{vol}.pdf",
    bbox_inches="tight",
)
