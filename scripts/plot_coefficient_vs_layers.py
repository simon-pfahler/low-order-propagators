"""Plot coefficient of models vs number of layers.

Usage:
    plot_coefficient_vs_layers.py --model_type=<name> --action=<name> --lattice_size=<str> --mass=<mass> --path=<str> --gamma_index=<n>

Options:
    --model_type=<name>     Model type ("HC", "HL", "restricted")
    --action=<name>         Action name ("WilsonQuenched", "WilsonDynamic" or "Haar")
    --lattice_size=<str>    Lattice size string (e.g. "8c16")
    --mass=<mass>           Mass parameter value
    --path=<str>            Path to plot coefficients of
    --gamma_index=<n>       Gamma structure index to plot coefficients of
"""

import ast
import json
import os
import sys

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from docopt import docopt
from utility import canonicalize_path, generator_mapping, get_path_length

matplotlib.use("Agg")

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_type = args["--model_type"]
action = args["--action"]
lattice_size_str = args["--lattice_size"]
mass = args["--mass"]
path = ast.literal_eval(args["--path"])
if get_path_length(path) > 4:
    raise ValueError(
        f"Only coefficients for paths up to length 4 are available!"
    )
if path != canonicalize_path(path)[0]:
    raise ValueError(
        f"Please pass a canonical path! This path's canonical path is {canonicalize_path(path)[0]}"
    )
gamma_index = int(args["--gamma_index"])

nrseeds = 5

layerss = [
    layers
    for layers in range(20)
    if os.path.exists(
        f"data/coefficients/seeded_coefficients_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}_seed0.pt"
    )
]

path_length = get_path_length(path)

coefficients = torch.zeros(nrseeds, len(layerss), dtype=torch.cdouble)

# Get coefficients
for i, layers in enumerate(layerss):
    for seed in range(nrseeds):
        # Load coefficients
        all_coefficients = torch.load(
            f"data/coefficients/seeded_coefficients_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}_seed{seed}.pt",
            weights_only=True,
        )

        nr_paths = 0
        for p, v in all_coefficients.items():
            canonical_path, new_indices, new_signs = canonicalize_path(p)
            new_generator_indices, new_generator_signs = generator_mapping(
                new_indices, new_signs
            )
            if canonical_path == path:
                coefficients[seed, i] += (
                    new_generator_signs[gamma_index]
                    * v[new_generator_indices[gamma_index]]
                )
                nr_paths += 1
        coefficients[seed, i] /= nr_paths

hopping_coefficient = 0
all_coefficients = torch.load(
    f"data/coefficients/coefficients_4layers_hopping_m{mass}.pt",
    weights_only=True,
)

nr_paths = 0
for k, v in all_coefficients.items():
    canonical_path, new_indices, new_signs = canonicalize_path(k)
    new_generator_indices, new_generator_signs = generator_mapping(
        new_indices, new_signs
    )
    if canonical_path == path:
        hopping_coefficient += (
            new_generator_signs[gamma_index]
            * v[new_generator_indices[gamma_index]]
        )
        nr_paths += 1
hopping_coefficient /= nr_paths

plt.figure(figsize=(5, 5))

plt.errorbar(
    layerss,
    coefficients.real.mean(dim=0),
    yerr=coefficients.real.std(dim=0),
    linestyle="none",
    color="blue",
    capsize=5,
    marker="o",
    markerfacecolor="none",
    label=f"{model_type} model",
)

xlim = plt.xlim()

plt.axhline(hopping_coefficient.real, label=f"Hopping expansion", c="red")

plt.xlim(xlim)

plt.grid()
plt.legend()
plt.xlabel("Layers")
plt.ylabel("Real part of coefficient")
plt.title(
    f"Coefficient evolution for {model_type} models\n"
    f"path {path}, Gamma index {gamma_index}, at mass {mass}"
)

os.makedirs("plots/coefficients/", exist_ok=True)

plt.savefig(
    f"plots/coefficients/coefficients_vs_layers_{model_type}_{action}_{lattice_size_str}_p{args["--path"]}_g{gamma_index}_m{mass}.pdf",
    bbox_inches="tight",
)
