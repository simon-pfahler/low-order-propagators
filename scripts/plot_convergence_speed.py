"""Plot the convergence speed of the residuals as a function of mass.

Usage:
    plot_convergence_speed.py --volume=<vol>

Options:
    --volume=<vol>          Volume string
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from docopt import docopt


def extract_mass(filename):
    try:
        mass = float(re.search(r"_m([+-]?\d+\.?\d*)\.pt", filename).group(1))
    except:
        raise ValueError(f"Mass not extractable from filename '{filename}'!")
    return mass


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


def fit_convergence_factor(nsteps, means):
    """Fit r(n) = A * b^n via log-linear regression; return b."""
    nsteps = np.asarray(nsteps, dtype=float)
    log_means = np.log(np.asarray(means, dtype=float))
    slope, _ = np.polyfit(nsteps, log_means, 1)
    return float(np.exp(slope))


# Parse docopt arguments
args = docopt(__doc__)
vol = args["--volume"]

masses_hopping = []
factors_hopping = []
masses_Clifford = []
factors_Clifford = []
masses_restricted = []
factors_restricted = []

# Find all masses
masses = sorted(set(extract_mass(f) for f in os.listdir("data/residuals")))

# For each mass, gather residuals across all available steps/layers and fit
for mass in masses:
    # Hopping expansion
    hopping_points = []
    for nsteps in sorted(
        set(extract_layers_or_steps(f) for f in os.listdir("data/residuals"))
    ):
        path = f"data/residuals/residuals_{nsteps}steps_{vol}_hopping_m{mass:.2f}.pt"
        if os.path.exists(path):
            data = torch.load(path, weights_only=True)
            hopping_points.append((nsteps, float(torch.mean(data))))

    if len(hopping_points) >= 2:
        ns, ms = zip(*hopping_points)
        masses_hopping.append(mass)
        factors_hopping.append(fit_convergence_factor(ns, ms))

    # Clifford model
    clifford_points = []
    for nsteps in sorted(
        set(extract_layers_or_steps(f) for f in os.listdir("data/residuals"))
    ):
        path = f"data/residuals/residuals_{nsteps}layers_{vol}_Clifford_m{mass:.2f}.pt"
        if os.path.exists(path):
            data = torch.load(path, weights_only=True)
            clifford_points.append((nsteps, float(torch.mean(data))))

    if len(clifford_points) >= 2:
        ns, ms = zip(*clifford_points)
        masses_Clifford.append(mass)
        factors_Clifford.append(fit_convergence_factor(ns, ms))

    # Restricted model
    restricted_points = []
    for nsteps in sorted(
        set(extract_layers_or_steps(f) for f in os.listdir("data/residuals"))
    ):
        path = f"data/residuals/residuals_{nsteps}layers_{vol}_restricted_m{mass:.2f}.pt"
        if os.path.exists(path):
            data = torch.load(path, weights_only=True)
            restricted_points.append((nsteps, float(torch.mean(data))))

    if len(restricted_points) >= 2:
        ns, ms = zip(*restricted_points)
        masses_restricted.append(mass)
        factors_restricted.append(fit_convergence_factor(ns, ms))

# Create plot
plt.figure(figsize=(10, 6))

plt.plot(
    masses_hopping,
    factors_hopping,
    linestyle="none",
    color="blue",
    marker="o",
    markerfacecolor="none",
    label="Hopping expansion",
)
plt.plot(
    masses_Clifford,
    factors_Clifford,
    linestyle="none",
    color="orange",
    marker="s",
    markerfacecolor="none",
    label="Clifford model",
)
plt.plot(
    masses_restricted,
    factors_restricted,
    linestyle="none",
    color="green",
    marker="D",
    markerfacecolor="none",
    label="Restricted model",
)

plt.xlabel("Mass")
plt.ylabel(r"Convergence factor $b$ ($Q \propto b^n$)")
plt.title(f"Convergence speed vs mass (volume={vol})")
plt.ylim(0, 1)
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

os.makedirs("plots/png/convergence", exist_ok=True)
os.makedirs("plots/pdf/convergence", exist_ok=True)

plt.savefig(
    f"plots/png/convergence/convergence_speed_{vol}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/convergence/convergence_speed_{vol}.pdf",
    bbox_inches="tight",
)
