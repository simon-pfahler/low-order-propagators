"""Create the action dependence of approximation quality figure of the paper."""

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


plt.style.use("scripts/iclr2027.mplstyle")

# Parse docopt arguments
layers = 4
lattice_size_str = "8c16"
model_lattice_size_str = "8c16"

fig, ax = plt.subplots(3, 3, figsize=(9, 7))
plt.subplots_adjust(wspace=0.08, hspace=0.08)

actions = ["WilsonQuenched", "WilsonDynamic", "Haar"]

for train in range(3):
    model_action = actions[train]
    for test in range(3):
        action = actions[test]

        means_hopping = []
        stds_hopping = []
        masses_hopping = []
        means_HC = []
        stds_HC = []
        masses_HC = []
        means_restricted = []
        stds_restricted = []
        masses_restricted = []
        means_GMRES = []
        stds_GMRES = []
        masses_GMRES = []

        # Find all masses
        masses = sorted(set(extract_mass(f) for f in os.listdir("data/Qs")))

        # Aggregate all available Qs data
        for mass in masses:
            # Hopping expansion
            hopping_path = f"data/Qs/Qs_{layers}layers_hopping_{action}_{lattice_size_str}_m{mass:.2f}.pt"
            if os.path.exists(hopping_path):
                data = torch.load(hopping_path, weights_only=True)

                masses_hopping.append(mass)
                means_hopping.append(torch.mean(data))
                stds_hopping.append(torch.std(data))

            # GMRES
            gmres_path = f"data/Qs/Qs_{layers}layers_GMRES_{action}_{lattice_size_str}_m{mass:.2f}.pt"
            if os.path.exists(gmres_path):
                data = torch.load(gmres_path, weights_only=True)
                masses_GMRES.append(mass)
                means_GMRES.append(torch.mean(data))
                stds_GMRES.append(torch.std(data))

            # HC model
            hc_path = f"data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size_str}_HC_{action}_{lattice_size_str}_m{mass:.2f}.pt"
            if os.path.exists(hc_path):
                data = torch.load(hc_path, weights_only=True)
                masses_HC.append(mass)
                means_HC.append(torch.mean(data))
                stds_HC.append(torch.std(data))

            # Restricted model
            restricted_path = f"data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size_str}_restricted_{action}_{lattice_size_str}_m{mass:.2f}.pt"
            if os.path.exists(restricted_path):
                data = torch.load(restricted_path, weights_only=True)
                masses_restricted.append(mass)
                means_restricted.append(torch.mean(data))
                stds_restricted.append(torch.std(data))

        # Create plot

        ax[test][train].errorbar(
            masses_hopping,
            means_hopping,
            yerr=stds_hopping,
            linestyle="none",
            color="#33bbee",
            capsize=4,
            marker="o",
            markersize=4,
            markerfacecolor="none",
            label="Hopping expansion",
        )
        ax[test][train].errorbar(
            masses_GMRES,
            means_GMRES,
            yerr=stds_GMRES,
            linestyle="none",
            color="#ee3377",
            capsize=4,
            marker="^",
            markersize=4,
            markerfacecolor="none",
            label="GMRES",
        )
        ax[test][train].errorbar(
            masses_restricted,
            means_restricted,
            yerr=stds_restricted,
            linestyle="none",
            color="#0077bb",
            capsize=4,
            marker="D",
            markersize=4,
            markerfacecolor="none",
            label="Restricted model",
        )
        ax[test][train].errorbar(
            masses_HC,
            means_HC,
            yerr=stds_HC,
            linestyle="none",
            color="#ee7733",
            capsize=4,
            marker="s",
            markersize=4,
            markerfacecolor="none",
            label="HC model",
        )

        if action == "Haar":
            ax[test][train].set_xlabel("Mass")
        else:
            ax[test][train].tick_params(labelbottom=False)
        if model_action == "WilsonQuenched":
            if action == "WilsonQuenched":
                ax[test][train].set_ylabel(
                    f"Approximation error\non quenched gauge fields"
                )
            elif action == "WilsonDynamic":
                ax[test][train].set_ylabel(
                    f"Approximation error\non dynamical gauge fields"
                )
            else:
                ax[test][train].set_ylabel(
                    f"Approximation error\non Haar-distributed gauge fields"
                )
        else:
            ax[test][train].tick_params(labelleft=False)
        if action == "WilsonQuenched":
            if model_action == "WilsonQuenched":
                ax[test][train].set_title(f"Trained on quenched gauge fields")
            elif model_action == "WilsonDynamic":
                ax[test][train].set_title(f"Trained on dynamical gauge fields")
            else:
                ax[test][train].set_title(
                    f"Trained on Haar-distributed gauge fields"
                )
        ax[test][train].set_yscale("log")
        ax[test][train].legend()
        ax[test][train].grid(True, which="both", linestyle="--", alpha=0.5)

        ax[test][train].set_ylim(1.2e-5, 1.9)

os.makedirs("plots/Qs/", exist_ok=True)

plt.savefig(
    f"plots/Qs/Qs_action_dependence.pdf",
    bbox_inches="tight",
)
