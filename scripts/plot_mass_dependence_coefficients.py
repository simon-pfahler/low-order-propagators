"""Plot the mass dependence of coefficients for length-n paths.

Usage:
    plot_mass_dependence_coefficients.py --layers=<n> --model_type=<type> --action=<name> --lattice_size=<str> --path_length=<n>

Options:
    --layers=<n>                Number of layers of the model, or number of steps of the hopping expansion/GMRES
    --model_type=<name>         Model type ("HC", "HL", "restricted"), "hopping" or "GMRES"
    --action=<name>       Action the model was trained on
    --lattice_size=<str>  Lattice size the model was trained on
    --path_length=<n>           Length of the paths to create the plot for
"""

import os
import re
import sys
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import torch
from docopt import docopt
from utility import (
    canonicalize_path,
    consolidate_path,
    generators,
    get_path_length,
    model_paths,
)


def extract_mass(filename):
    try:
        mass = float(re.search(r"_m([+-]?\d+\.?\d*)\.pt", filename).group(1))
    except:
        raise ValueError(f"Mass not extractable from filename '{filename}'!")
    return mass


def get_reference_curve(order, prefactor):
    """Generate reference curve for hopping expansion."""
    mass_range = torch.linspace(-3.98, 2.5, 1000)
    line_values = prefactor / ((2**order) * (mass_range + 4) ** (order + 1))
    return mass_range, line_values


matplotlib.use("Agg")

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
layers = int(args["--layers"])
model_type = args["--model_type"]
if model_type not in ["restricted", "HL", "HC"]:
    raise ValueError(f"Model type '{model_type}' not supported!")
action = args["--action"]
lattice_size_str = args["--lattice_size"]
path_length = int(args["--path_length"])

# Find all masses
masses = sorted(
    set(
        extract_mass(f)
        for f in os.listdir("data/coefficients")
        if not f.startswith("seeded")
    )
)

# Get hopping coefficients for mass 1
hopping_coefficients = {
    k: v
    for k, v in torch.load(
        f"data/coefficients/coefficients_4layers_hopping_m1.00.pt",
        weights_only=True,
    ).items()
    if get_path_length(k) == path_length
}

# get categories of paths
categories = list(
    set(tuple(canonicalize_path(k)[0]) for k in hopping_coefficients.keys())
)
category_index = dict()
for i, c in enumerate(categories):
    for e in c:
        category_index[tuple(e)] = i

category_names = categories + ["zero"]

scatter_points = [[[] for _ in masses] for _ in range(len(categories) + 1)]
for mass_idx, mass in enumerate(masses):
    try:
        coefficients = {
            k: v
            for k, v in torch.load(
                f"data/coefficients/coefficients_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass:.2f}.pt",
                weights_only=True,
            ).items()
            if get_path_length(k) == path_length
        }
    except:
        continue

    for k, v in coefficients.items():
        ck = tuple(canonicalize_path(k)[0])
        hopping_v = hopping_coefficients[k]
        for i in range(16):
            if hopping_v[i] != 0:
                scatter_points[categories.index(ck)][mass_idx].append(
                    torch.abs(v[i])
                )
            else:
                scatter_points[-1][mass_idx].append(torch.abs(v[i]))

plt.figure(figsize=(10, 6))
ymin = 0
ymax = 0
for idx in range(len(scatter_points)):
    s = torch.tensor(scatter_points[idx])
    m = torch.tensor(masses).unsqueeze(-1).expand(s.shape)
    if s.numel() == 0:
        continue
    if idx != len(scatter_points) - 1:
        ymax = max(ymax, torch.max(s).item())
    plt.scatter(m, s, label=category_names[idx])

for factor in range(path_length):
    plt.plot(
        *get_reference_curve(path_length, 2**factor),
        label=f"{path_length},{2**factor}",
        c="r",
    )
if path_length == 0:
    plt.plot(
        *get_reference_curve(0, 1),
        label="0,1",
        c="r",
    )

plt.xlabel("Mass")
plt.ylabel("Absolute value of coefficient")
plt.ylim(ymin, 1.1 * ymax)
plt.title(
    f"Dependence of coefficients on mass parameter\n"
    f"for {layers} layers, {action} {lattice_size_str}, {model_type}\n"
    f"Coefficients for paths of length {path_length}"
)
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

os.makedirs("plots/mass_dependence", exist_ok=True)

plt.savefig(
    f"plots/mass_dependence/mass_dependence_{layers}layers_{model_type}_{action}_{lattice_size_str}_pathlength{path_length}.pdf",
    bbox_inches="tight",
)
