"""Plot coefficient categories of a model, and in comparison to the hopping expansion.

Usage:
    plot_categories.py --model=<name> --mass=<mass>

Options:
    --model=<name>          Name of the model json file (in `models/{name}.json`)
    --mass=<mass>           Mass parameter value
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

matplotlib.use("Agg")

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_name = args["--model"]
mass = args["--mass"]
kappa = 1 / (2 * (float(mass) + 4))

# Load model json
model_json_path = f"models/{model_name}.json"
with open(model_json_path, "r") as f:
    model_json = json.load(f)

model_type = model_json["model_type"]
if model_type not in ["restricted", "4x4", "Clifford"]:
    raise ValueError(f"Model type '{model_type}' not supported!")
lattice_size = model_json["lattice_size"]
volume = f"{lattice_size[0]}c{lattice_size[3]}"
nlayers = model_json["nlayers"]
train_configs = model_json["train_configs"]
test_configs = model_json["test_configs"]

# Load weights and get coefficients
coefficients = torch.load(
    f"data/coefficients/coefficients_{model_name}_m{mass}.pt", weights_only=True
)

for k, v in coefficients.items():
    coefficients[k] /= kappa ** get_path_length(k)

# Get hopping coefficients
hopping_coefficients = torch.load(
    f"data/coefficients/coefficients_hopping_m{mass}.pt", weights_only=True
)

for k, v in hopping_coefficients.items():
    hopping_coefficients[k] /= kappa ** get_path_length(k)

# get categories of paths
categories = list()
categories_paths = list()
for k in hopping_coefficients.keys():
    ck = canonicalize_path(k)[0]
    if ck not in categories:
        categories.append(ck)
        categories_paths.append(list())
    categories_paths[categories.index(ck)].append(k)

coefficients_matrix_means = torch.zeros(len(categories), 16)
coefficients_matrix_stds = torch.zeros(len(categories), 16)
hopping_coefficients_matrix_means = torch.zeros(len(categories), 16)
hopping_coefficients_matrix_stds = torch.zeros(len(categories), 16)
for category_idx in range(len(categories)):
    values = list()
    hopping_values = list()
    for path in categories_paths[category_idx]:
        canonical_path, new_indices, new_signs = canonicalize_path(path)
        new_generator_indices, new_generator_signs = generator_mapping(
            new_indices, new_signs
        )
        coef = coefficients[path]
        hopping_coef = hopping_coefficients[path]
        can_coef = torch.stack(
            [
                new_generator_signs[i] * coef[new_generator_indices[i]]
                for i in range(16)
            ]
        )
        hopping_can_coef = torch.stack(
            [
                new_generator_signs[i] * hopping_coef[new_generator_indices[i]]
                for i in range(16)
            ]
        )
        values.append(can_coef.abs())
        hopping_values.append(hopping_can_coef.abs())

    values = torch.stack(values).real
    hopping_values = torch.stack(hopping_values).real
    if values.shape[0] == 1:
        coefficients_matrix_means[category_idx] = values[0]
        hopping_coefficients_matrix_means[category_idx] = hopping_values[0]
        coefficients_matrix_stds[category_idx] = torch.zeros_like(values[0])
        hopping_coefficients_matrix_stds[category_idx] = torch.zeros_like(
            hopping_values[0]
        )
    else:
        coefficients_matrix_means[category_idx] = torch.mean(values, dim=0)
        hopping_coefficients_matrix_means[category_idx] = torch.mean(
            hopping_values, dim=0
        )
        coefficients_matrix_stds[category_idx] = torch.std(values, dim=0)
        hopping_coefficients_matrix_stds[category_idx] = torch.std(
            hopping_values, dim=0
        )

# Create plots
os.makedirs("plots/png/categories", exist_ok=True)
os.makedirs("plots/pdf/categories", exist_ok=True)

coefficients_matrix_means_clipped = torch.max(
    coefficients_matrix_means, 1e-5 * torch.ones_like(coefficients_matrix_means)
)
coefficients_matrix_stds_clipped = torch.max(
    coefficients_matrix_stds, 1e-5 * torch.ones_like(coefficients_matrix_stds)
)
hopping_coefficients_matrix_means_clipped = torch.max(
    hopping_coefficients_matrix_means,
    1e-5 * torch.ones_like(coefficients_matrix_means),
)
hopping_coefficients_matrix_stds_clipped = torch.max(
    hopping_coefficients_matrix_stds,
    1e-5 * torch.ones_like(coefficients_matrix_stds),
)

# Coefficients matrix plot for hopping expansion
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
im_means = ax.imshow(
    hopping_coefficients_matrix_means_clipped.T,
    # norm=mcolors.LogNorm(vmax=1e0, vmin=1e-5),
    aspect="auto",
)
ax.set_xticks(
    ticks=[i for i in range(coefficients_matrix_means.shape[0])],
    labels=["" for k in categories],
)
ax.set_xlabel("Path")
ax.set_ylabel("Gamma structure index")
fig.colorbar(im_means)
ax.set_title(
    f"Clifford algebra coefficients per path category\n"
    f"hopping expansion, mass {mass}\n"
    f"normalized by kappa^ell"
)
plt.savefig(
    f"plots/png/categories/categories_{nlayers}layers_{volume}_hopping_m{mass}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/categories/categories_{nlayers}layers_{volume}_hopping_m{mass}.pdf",
    bbox_inches="tight",
)

# Coefficients matrix plot
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
im_means = ax.imshow(
    coefficients_matrix_means_clipped.T,
    # norm=mcolors.LogNorm(vmax=1e0, vmin=1e-5),
    aspect="auto",
)
ax.set_xticks(
    ticks=[i for i in range(coefficients_matrix_means.shape[0])],
    labels=[str(k) for k in categories],
    rotation=90,
)
ax.set_xlabel("Path")
ax.set_ylabel("Gamma structure index")
fig.colorbar(im_means)
ax.set_title(
    f"Mean Clifford algebra coefficients per path category\n"
    f"model {model_name}, mass {mass}\n"
    f"normalized by kappa^ell"
)
plt.savefig(
    f"plots/png/categories/categories_{model_name}_m{mass}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/categories/categories_{model_name}_m{mass}.pdf",
    bbox_inches="tight",
)


diff = (coefficients_matrix_means - hopping_coefficients_matrix_means).abs()
diff = torch.max(diff, 1e-5 * torch.ones_like(diff))

# Difference from hopping expansion coefficients
plt.figure(figsize=(10, 3))
plt.imshow(
    diff.T,
    # norm=mcolors.LogNorm(vmin=1e-5, vmax=1e0),
    aspect="auto",
)
plt.xticks(
    ticks=[i for i in range(coefficients_matrix_means.shape[0])],
    labels=[str(k) for k in categories],
    rotation=90,
)
plt.xlabel("Path")
plt.ylabel("Gamma structure index")
plt.colorbar()
plt.title(
    f"Difference from hopping expansion coefficients per path category\n"
    f"model {model_name}, mass {mass}\n"
    f"normalized by kappa^ell"
)
plt.savefig(
    f"plots/png/categories/comparison_hopping_{model_name}_m{mass}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/categories/comparison_hopping_{model_name}_m{mass}.pdf",
    bbox_inches="tight",
)
