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


def fit_convergence_factor(nsteps, means, sigmas):
    """Fit r(n) = A * b^n via weighted log-linear regression."""
    nsteps = np.asarray(nsteps, dtype=float)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    log_means = np.log(means)
    log_sigmas = sigmas / means
    cov_mode = True if len(nsteps) > 2 else "unscaled"
    coeffs, cov = np.polyfit(
        nsteps, log_means, 1, w=1.0 / log_sigmas, cov=cov_mode
    )
    slope, slope_var = coeffs[0], cov[0, 0]
    b = float(np.exp(slope))
    b_err = float(b * np.sqrt(slope_var))
    return b, b_err


# Parse docopt arguments
args = docopt(__doc__)
vol = args["--volume"]

masses_hopping = []
factors_hopping = []
factors_hopping_err = []
masses_Clifford = []
factors_Clifford = []
factors_Clifford_err = []
masses_restricted = []
factors_restricted = []
factors_restricted_err = []
masses_GMRES = []
factors_GMRES = []
factors_GMRES_err = []

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
            mean = float(torch.mean(data))
            sem = float(torch.std(data) / np.sqrt(data.numel()))
            hopping_points.append((nsteps, mean, sem))

    if len(hopping_points) >= 2:
        ns, ms, ss = zip(*hopping_points)
        b, b_err = fit_convergence_factor(ns, ms, ss)
        masses_hopping.append(mass)
        factors_hopping.append(b)
        factors_hopping_err.append(b_err)

    # Clifford model
    clifford_points = []
    for nsteps in sorted(
        set(extract_layers_or_steps(f) for f in os.listdir("data/residuals"))
    ):
        path = f"data/residuals/residuals_{nsteps}layers_{vol}_Clifford_m{mass:.2f}.pt"
        if os.path.exists(path):
            data = torch.load(path, weights_only=True)
            mean = float(torch.mean(data))
            sem = float(torch.std(data) / np.sqrt(data.numel()))
            clifford_points.append((nsteps, mean, sem))

    if len(clifford_points) >= 2:
        ns, ms, ss = zip(*clifford_points)
        b, b_err = fit_convergence_factor(ns, ms, ss)
        masses_Clifford.append(mass)
        factors_Clifford.append(b)
        factors_Clifford_err.append(b_err)

    # Restricted model
    restricted_points = []
    for nsteps in sorted(
        set(extract_layers_or_steps(f) for f in os.listdir("data/residuals"))
    ):
        path = f"data/residuals/residuals_{nsteps}layers_{vol}_restricted_m{mass:.2f}.pt"
        if os.path.exists(path):
            data = torch.load(path, weights_only=True)
            mean = float(torch.mean(data))
            sem = float(torch.std(data) / np.sqrt(data.numel()))
            restricted_points.append((nsteps, mean, sem))

    if len(restricted_points) >= 2:
        ns, ms, ss = zip(*restricted_points)
        b, b_err = fit_convergence_factor(ns, ms, ss)
        masses_restricted.append(mass)
        factors_restricted.append(b)
        factors_restricted_err.append(b_err)

    # GMRES
    gmres_points = []
    for nsteps in sorted(
        set(extract_layers_or_steps(f) for f in os.listdir("data/residuals"))
    ):
        path = f"data/residuals/residuals_{nsteps}steps_{vol}_GMRES_m{mass:.2f}.pt"
        if os.path.exists(path):
            data = torch.load(path, weights_only=True)
            mean = float(torch.mean(data))
            sem = float(torch.std(data) / np.sqrt(data.numel()))
            gmres_points.append((nsteps, mean, sem))

    if len(gmres_points) >= 2:
        ns, ms, ss = zip(*gmres_points)
        b, b_err = fit_convergence_factor(ns, ms, ss)
        masses_GMRES.append(mass)
        factors_GMRES.append(b)
        factors_GMRES_err.append(b_err)

# Create plot
plt.figure(figsize=(10, 6))

plt.errorbar(
    masses_hopping,
    factors_hopping,
    yerr=factors_hopping_err,
    linestyle="none",
    color="blue",
    marker="o",
    markerfacecolor="none",
    capsize=3,
    label="Hopping expansion",
)
plt.errorbar(
    masses_Clifford,
    factors_Clifford,
    yerr=factors_Clifford_err,
    linestyle="none",
    color="orange",
    marker="s",
    markerfacecolor="none",
    capsize=3,
    label="Clifford model",
)
plt.errorbar(
    masses_restricted,
    factors_restricted,
    yerr=factors_restricted_err,
    linestyle="none",
    color="green",
    marker="D",
    markerfacecolor="none",
    capsize=3,
    label="Restricted model",
)
plt.errorbar(
    masses_GMRES,
    factors_GMRES,
    yerr=factors_GMRES_err,
    linestyle="none",
    color="red",
    marker="^",
    markerfacecolor="none",
    capsize=3,
    label="GMRES",
)

plt.xlabel("Mass")
plt.ylabel(r"Convergence rate $b$ ($Q \propto b^n$)")
plt.title(f"Convergence rate vs mass (volume={vol})")
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
