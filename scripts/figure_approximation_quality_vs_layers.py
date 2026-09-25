"""Create the approximation quality vs layers figure of the paper."""

import os
import re

import matplotlib.pyplot as plt
import torch


def extract_layers(filename):
    try:
        layers = int(re.search(r"Qs_(\d+)layers", filename).group(1))
    except:
        raise ValueError(
            f"Number of layers not extractable from filename '{filename}'!"
        )
    return layers


plt.style.use("./scripts/iclr2027.mplstyle")

masses = [1.4, -3]
action = "WilsonQuenched"
lattice_size_str = "16c32"
model_action = "WilsonQuenched"
model_lattice_size_str = "8c16"

fig, ax = plt.subplots(1, 2, figsize=(9, 2.3))
plt.subplots_adjust(wspace=0.15, hspace=0.08)

for idx in range(2):
    mass = masses[idx]

    means_hopping = []
    stds_hopping = []
    layerss_hopping = []
    means_HC = []
    stds_HC = []
    layerss_HC = []
    means_restricted = []
    stds_restricted = []
    layerss_restricted = []
    means_GMRES = []
    stds_GMRES = []
    layerss_GMRES = []

    layerss = [1, 2, 3, 4, 6]
    if idx == 1:
        layerss += [8, 12, 16]

    # Aggregate all available Qs data
    for layers in layerss:
        # Hopping expansion
        hopping_path = f"data/Qs/Qs_{layers}layers_hopping_{action}_{lattice_size_str}_m{mass:.2f}.pt"
        if os.path.exists(hopping_path):
            data = torch.load(hopping_path, weights_only=True)

            layerss_hopping.append(layers)
            means_hopping.append(torch.mean(data))
            stds_hopping.append(torch.std(data))

        # GMRES
        gmres_path = f"data/Qs/Qs_{layers}layers_GMRES_{action}_{lattice_size_str}_m{mass:.2f}.pt"
        if os.path.exists(gmres_path):
            data = torch.load(gmres_path, weights_only=True)
            layerss_GMRES.append(layers)
            means_GMRES.append(torch.mean(data))
            stds_GMRES.append(torch.std(data))

        # HC model
        hc_path = f"data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size_str}_HC_{action}_{lattice_size_str}_m{mass:.2f}.pt"
        if os.path.exists(hc_path):
            data = torch.load(hc_path, weights_only=True)
            layerss_HC.append(layers)
            means_HC.append(torch.mean(data))
            stds_HC.append(torch.std(data))

        # Restricted model
        restricted_path = f"data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size_str}_restricted_{action}_{lattice_size_str}_m{mass:.2f}.pt"
        if os.path.exists(restricted_path):
            data = torch.load(restricted_path, weights_only=True)
            layerss_restricted.append(layers)
            means_restricted.append(torch.mean(data))
            stds_restricted.append(torch.std(data))

    # Create plot
    ax[idx].errorbar(
        layerss_hopping,
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
        layerss_GMRES,
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
        layerss_restricted,
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
        layerss_HC,
        means_HC,
        yerr=stds_HC,
        linestyle="none",
        color="#ee7733",
        capsize=5,
        marker="s",
        markerfacecolor="none",
        label="HC model",
    )

    ax[idx].set_xlabel("Number of steps/layers")
    if idx == 0:
        ax[idx].set_ylabel("Approximation error")
    if idx == 0:
        ax[idx].set_xticks([1, 2, 3, 4, 6])
    else:
        ax[idx].set_xticks([2, 4, 6, 8, 12, 16])
        ax[idx].set_ylim(6.2e-3, 1.28e2)
    ax[idx].set_title(rf"Approximation error vs layers ($m={mass}$)")
    ax[idx].set_yscale("log")
    ax[idx].legend()
    ax[idx].grid(True, which="both", linestyle="--", alpha=0.5)

os.makedirs("plots/Qs/", exist_ok=True)

plt.savefig(
    f"plots/Qs/Qs_vs_layers.pdf",
    bbox_inches="tight",
)
