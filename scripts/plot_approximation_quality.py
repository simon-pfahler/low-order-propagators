"""Plot mass-dependence of approximation quality for different approximations of D^-1.

Usage:
    plot_approximation_quality.py --layers=<n> --action=<name> --lattice_size=<str> --model_action=<name> --model_lattice_size=<str>

Options:
    --layers=<n>                Number of layers of the model, or number of steps of the hopping expansion/GMRES
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


def extract_mass(filename):
    try:
        mass = float(re.search(r"_m([+-]?\d+\.?\d*)\.pt", filename).group(1))
    except:
        raise ValueError(f"Mass not extractable from filename '{filename}'!")
    return mass


plt.style.use("scripts/iclr2027.mplstyle")

# Parse docopt arguments
args = docopt(__doc__)
layers = int(args["--layers"])
action = args["--action"]
lattice_size_str = args["--lattice_size"]
model_action = args["--model_action"]
model_lattice_size_str = args["--model_lattice_size"]

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
plt.figure(figsize=(4.5, 2.3))

plt.errorbar(
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
plt.errorbar(
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
plt.errorbar(
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
plt.errorbar(
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

plt.xlabel("Mass")
plt.ylabel("Approximation Error")
plt.title(
    f"Approximation Error vs Mass for {layers} layers\n"
    f"{action}, {lattice_size_str} volume\n"
    f"Trained on {model_action}, {model_lattice_size_str} volume"
)
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.ylim(
    min(means_HC + means_restricted + means_hopping + means_GMRES) / 1.3,
    1.3 * max(means_HC + means_restricted + means_GMRES),
)

os.makedirs("plots/Qs/", exist_ok=True)

plt.savefig(
    f"plots/Qs/Qs_{layers}layers_{action}_{lattice_size_str}_{model_action}_{model_lattice_size_str}.pdf",
    bbox_inches="tight",
)

improvements_HC = torch.tensor(
    [
        min((eGMRES - eHC) / eGMRES, (eH - eHC) / eH)
        for eGMRES, eH, eHC in zip(means_GMRES, means_hopping, means_HC)
    ]
)
improvements_restricted = torch.tensor(
    [
        min((eGMRES - eR) / eGMRES, (eH - eR) / eH)
        for eGMRES, eH, eR in zip(means_GMRES, means_hopping, means_restricted)
    ]
)

print(
    f"Relative improvement between HC and best baseline: {100*torch.mean(improvements_HC):.2f}% ({100*torch.mean(improvements_HC[:23]):.2f}% below m_h, {100*torch.mean(improvements_HC[23:]):.2f}% above m_h)"
)
print(
    f"Relative improvement between restricted and best baseline: {100*torch.mean(improvements_restricted):.2f}% ({100*torch.mean(improvements_restricted[:23]):.2f}% below m_h, {100*torch.mean(improvements_restricted[23:]):.2f}% above m_h)"
)
