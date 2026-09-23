"""Plot approximation quality vs layers for different approximations of D^-1.

Usage:
    plot_approximation_quality_vs_layers.py --mass=<mass> --action=<name> --lattice_size=<str> --model_action=<name> --model_lattice_size=<str>

Options:
    --mass=<mas>            Mass parameter
    --action=<name>             Action name ("WilsonQuenched", "WilsonDynamic" or "Haar")
    --lattice_size=<str>        Lattice size string (e.g. "8c16")
    --model_action=<name>       Action the model was trained on
    --model_lattice_size=<str>  Lattice size the model was trained on
"""

import os
import re

import matplotlib.pyplot as plt
import torch
from docopt import docopt


def extract_layers(filename):
    try:
        layers = int(re.search(r"Qs_(\d+)layers", filename).group(1))
    except:
        raise ValueError(
            f"Number of layers not extractable from filename '{filename}'!"
        )
    return layers


plt.style.use("./scripts/iclr2027.mplstyle")

# Parse docopt arguments
args = docopt(__doc__)
mass = float(args["--mass"])
action = args["--action"]
lattice_size_str = args["--lattice_size"]
model_action = args["--model_action"]
model_lattice_size_str = args["--model_lattice_size"]

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

# Find all layers
layerss = sorted(set(extract_layers(f) for f in os.listdir("data/Qs")))

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
plt.figure(figsize=(4.5, 2.3))

plt.errorbar(
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
plt.errorbar(
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
plt.errorbar(
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
plt.errorbar(
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

plt.xlabel("Number of steps/layers")
plt.ylabel("Approximation Error")
plt.xticks([2, 4, 6, 8, 12, 16])
plt.title(rf"Approximation Error vs Layers ($m={mass}$)")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

os.makedirs("plots/Qs/", exist_ok=True)

plt.savefig(
    f"plots/Qs/Qs_vs_layers_{action}_{lattice_size_str}_{model_action}_{model_lattice_size_str}_m{mass:.2f}.pdf",
    bbox_inches="tight",
)
