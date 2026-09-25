"""Create the approximation quality mass dependence volume dependence plot of the paper."""

import os
import re

import matplotlib.pyplot as plt
import torch


def extract_mass(filename):
    try:
        mass = float(re.search(r"_m([+-]?\d+\.?\d*)\.pt", filename).group(1))
    except:
        raise ValueError(f"Mass not extractable from filename '{filename}'!")
    return mass


plt.style.use("scripts/iclr2027.mplstyle")

# Parse docopt arguments
layers = 4
action = "WilsonQuenched"
lattice_size_strs = ["8c16", "16c32"]
model_action = "WilsonQuenched"
model_lattice_size_str = "8c16"


fig, ax = plt.subplots(1, 2, figsize=(9, 2.3))
plt.subplots_adjust(wspace=0.12, hspace=0.08)

for idx in range(2):
    lattice_size_str = lattice_size_strs[idx]

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
    ax[idx].errorbar(
        masses_hopping,
        means_hopping,
        yerr=stds_hopping,
        linestyle="none",
        color="#33bbee",
        capsize=5,
        marker="o",
        markerfacecolor="none",
        label="Hopping expansion",
    )
    ax[idx].errorbar(
        masses_GMRES,
        means_GMRES,
        yerr=stds_GMRES,
        linestyle="none",
        color="#ee3377",
        capsize=5,
        marker="^",
        markerfacecolor="none",
        label="GMRES",
    )
    ax[idx].errorbar(
        masses_restricted,
        means_restricted,
        yerr=stds_restricted,
        linestyle="none",
        color="#0077bb",
        capsize=5,
        marker="D",
        markerfacecolor="none",
        label="Restricted model",
    )
    ax[idx].errorbar(
        masses_HC,
        means_HC,
        yerr=stds_HC,
        linestyle="none",
        color="#ee7733",
        capsize=5,
        marker="s",
        markerfacecolor="none",
        label="HC model",
    )

    ax[idx].set_xlabel("Mass")
    if idx == 0:
        ax[idx].set_ylabel("Approximation error")
    if idx == 0:
        ax[idx].set_title(
            r"Approximation error vs mass on $8^3\times16$ volume"
        )
    else:
        ax[idx].set_title(
            r"Approximation error vs mass on $16^3\times32$ volume"
        )
    ax[idx].set_yscale("log")
    ax[idx].legend()
    ax[idx].grid(True, which="both", linestyle="--", alpha=0.5)

    ax[idx].set_ylim(
        min(means_HC + means_restricted + means_hopping + means_GMRES) / 1.3,
        1.3 * max(means_HC + means_restricted + means_GMRES),
    )

os.makedirs("plots/Qs/", exist_ok=True)

plt.savefig(
    f"plots/Qs/Qs_volume_dependence.pdf",
    bbox_inches="tight",
)
