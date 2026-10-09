"""Create the approximation quality mass dependence plot of the paper.

Usage:
    figure_propagator_error.py --layers=<n>

Options:
    --layers=<n>            Number of layers
"""

import os
import re

import matplotlib.pyplot as plt
import torch
from docopt import docopt

plt.style.use("scripts/iclr2027.mplstyle")

# Parse docopt arguments
args = docopt(__doc__)
layers = int(args["--layers"])
action = "WilsonQuenched"
lattice_size_str = "8c16"
model_action = "WilsonQuenched"
model_lattice_size_str = "8c16"

masses = [-5, -4, -3, -2, -1, 0, 1, 2]

data_hopping_mean = torch.nan * torch.zeros(len(masses))
data_hopping_std = torch.nan * torch.zeros(len(masses))
data_hc_mean = torch.nan * torch.zeros(len(masses))
data_hc_std = torch.nan * torch.zeros(len(masses))
data_restricted_mean = torch.nan * torch.zeros(len(masses))
data_restricted_std = torch.nan * torch.zeros(len(masses))

for mass_idx, mass in enumerate(masses):
    hopping_path = f"data/propagators/timesliced_errors_{layers}layers_hopping_WilsonQuenched_8c16_m{mass:.2f}.pt"
    hc_path = f"data/propagators/timesliced_errors_{layers}layers_WilsonQuenched_8c16_HC_WilsonQuenched_8c16_m{mass:.2f}.pt"
    restricted_path = f"data/propagators/timesliced_errors_{layers}layers_WilsonQuenched_8c16_restricted_WilsonQuenched_8c16_m{mass:.2f}.pt"

    try:
        data_hopping_mass = torch.load(hopping_path, weights_only=True)
        data_hc_mass = torch.load(hc_path, weights_only=True)
        data_restricted_mass = torch.load(restricted_path, weights_only=True)
    except:
        continue

    data_hopping_mass = (
        torch.einsum(
            "...t,...t->...", data_hopping_mass.conj(), data_hopping_mass
        ).real
        ** 0.5
    )
    data_hc_mass = (
        torch.einsum("...t,...t->...", data_hc_mass.conj(), data_hc_mass).real
        ** 0.5
    )
    data_restricted_mass = (
        torch.einsum(
            "...t,...t->...", data_restricted_mass.conj(), data_restricted_mass
        ).real
        ** 0.5
    )

    for idx in range(4):
        for spin_idx in range(4):
            for color_idx in range(3):
                exact_propagator = torch.load(
                    f"data/propagators/exact_propagator_{action}_{lattice_size_str}_U{idx}_s{spin_idx}_c{color_idx}_m{mass:.2f}.pt",
                    weights_only=True,
                )

                norm = (
                    torch.einsum(
                        "xyztsc,xyztsc->",
                        exact_propagator.conj(),
                        exact_propagator,
                    ).real
                    ** 0.5
                )
                data_hopping_mass[idx, spin_idx, color_idx] /= norm
                data_hc_mass[idx, spin_idx, color_idx] /= norm

    data_hopping_mean[mass_idx] = torch.mean(data_hopping_mass)
    data_hopping_std[mass_idx] = torch.std(data_hopping_mass)
    data_hc_mean[mass_idx] = torch.mean(data_hc_mass)
    data_hc_std[mass_idx] = torch.std(data_hc_mass)
    data_restricted_mean[mass_idx] = torch.mean(data_hc_mass)
    data_restricted_std[mass_idx] = torch.std(data_hc_mass)

plt.errorbar(
    masses,
    data_hopping_mean,
    yerr=data_hopping_std,
    linestyle="none",
    color="#33bbee",
    capsize=5,
    marker="o",
    markerfacecolor="none",
    label="Hopping expansion",
)
plt.errorbar(
    masses,
    data_restricted_mean,
    yerr=data_restricted_std,
    linestyle="none",
    color="#0077bb",
    capsize=5,
    marker="D",
    markerfacecolor="none",
    label="Restricted model",
)
plt.errorbar(
    masses,
    data_hc_mean,
    yerr=data_hc_std,
    linestyle="none",
    color="#ee7733",
    capsize=5,
    marker="s",
    markerfacecolor="none",
    label="HC model",
)

plt.legend()
plt.xlabel("Mass")
plt.ylabel("Relative propagator error")
plt.title(f"Relative propagator error vs mass for {layers} layers")
plt.yscale("log")

os.makedirs("plots/propagator/", exist_ok=True)

plt.savefig(
    f"plots/propagator/propagator_error_{layers}layers.pdf",
    bbox_inches="tight",
)
