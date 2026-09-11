"""Plot coefficient convergence of models with increasing number of layers.

Usage:
    plot_coefficient_convergence.py --model_type=<name> --mass=<mass> [--show_hopping]

Options:
    --model_type=<name>     Model type ("Clifford" or "restricted")
    --mass=<mass>           Mass parameter value
    --show_hopping          Show the coefficient of the hopping expansion
"""

import json
import os
import sys

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from docopt import docopt
from utility import canonicalize_path, generator_mapping, get_path_length

# matplotlib.use("Agg")

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_type = args["--model_type"]
mass = args["--mass"]
show_hopping = True if args["--show_hopping"] else False

layers = [
    i
    for i in range(9)
    if os.path.exists(
        f"data/weights/weights_{i}layers_8c16_{model_type}_m{mass}.pt"
    )
]

path = ((0, 1),)
cpath = canonicalize_path(path)[0]
gamma_index = 0
path_length = get_path_length(cpath)

coefficients = torch.zeros(len(layers), dtype=torch.cdouble)

# Get coefficients
for i, l in enumerate(layers):
    model_name = f"{l}layers_8c16_{model_type}"

    # Load model json
    with open(f"models/{model_name}.json", "r") as f:
        model_json = json.load(f)

    model_type = model_json["model_type"]
    if model_type not in ["restricted", "4x4", "Clifford"]:
        raise ValueError(f"Model type '{model_type}' not supported!")
    lattice_size = model_json["lattice_size"]
    volume = f"{lattice_size[0]}c{lattice_size[3]}"
    nlayers = model_json["nlayers"]
    train_configs = model_json["train_configs"]
    test_configs = model_json["test_configs"]

    # Load coefficients
    all_coefficients = torch.load(
        f"data/coefficients/coefficients_{model_name}_m{mass}.pt",
        weights_only=True,
    )

    nr_paths = 0
    for path, v in all_coefficients.items():
        canonical_path, new_indices, new_signs = canonicalize_path(path)
        new_generator_indices, new_generator_signs = generator_mapping(
            new_indices, new_signs
        )
        if canonical_path == cpath:
            coefficients[i] += (
                new_generator_signs[gamma_index]
                * v[new_generator_indices[gamma_index]]
            )
            nr_paths += 1
    coefficients[i] /= nr_paths

hopping_coefficient = 0
if show_hopping:
    all_coefficients = torch.load(
        f"data/coefficients/coefficients_hopping_m{mass}.pt",
        weights_only=True,
    )

    nr_paths = 0
    for k, v in all_coefficients.items():
        if canonicalize_path(k)[0] == cpath:
            hopping_coefficient += v[gamma_index]
            nr_paths += 1
    hopping_coefficient /= nr_paths

plt.figure(figsize=(5, 5))

plt.plot(
    layers,
    coefficients.real,
    "o",
    linestyle="None",
    label=f"{model_type} model",
)
plt.axhline(hopping_coefficient.real, label=f"Hopping expansion", c="red")

plt.grid()
plt.legend()
plt.xlabel("Layers")
plt.ylabel("Real part of coefficient")
plt.title(
    f"Coefficient evolution for {model_type} models\n"
    f"path {cpath}, Gamma index {gamma_index}, at mass {mass}"
)
plt.tight_layout()
plt.show()
