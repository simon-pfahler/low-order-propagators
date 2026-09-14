"""Plot coefficients of a model, and in comparison to the hopping expansion.

Usage:
    plot_coefficients.py --model=<name> --mass=<mass>

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
from utility import get_path_length

matplotlib.use("Agg")

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_name = args["--model"]
mass = args["--mass"]

# Load model json
model_json_path = f"models/{model_name}.json"
with open(model_json_path, "r") as f:
    model_json = json.load(f)

model_type = model_json["model_type"]
if model_type not in ["restricted", "4x4", "Clifford"]:
    raise ValueError(f"Model type '{model_type}' not supported!")
lattice_size = model_json["lattice_size"]
nlayers = model_json["nlayers"]
train_configs = model_json["train_configs"]
test_configs = model_json["test_configs"]

# Load weights and get coefficients
weights_path = f"data/weights/weights_{model_name}_m{mass}.pt"
if not os.path.exists(weights_path):
    raise ValueError(
        f"No weights file found for model '{model_name}' and mass={mass}"
    )

coefficients = {
    k: v
    for k, v in torch.load(
        f"data/coefficients/coefficients_{model_name}_m{mass}.pt",
        weights_only=True,
    ).items()
    if get_path_length(k) <= 2
}

# Get hopping coefficients
hopping_coefficients = {
    k: v
    for k, v in torch.load(
        f"data/coefficients/coefficients_hopping_m{mass}.pt", weights_only=True
    ).items()
    if get_path_length(k) <= min(2, nlayers)
}

# Create plots
os.makedirs("plots/png/coefficients", exist_ok=True)
os.makedirs("plots/pdf/coefficients", exist_ok=True)

# Coefficients matrix plot
coefficients_matrix = torch.stack([v for v in coefficients.values()])
hopping_coefficients_matrix = torch.stack(
    [v for v in hopping_coefficients.values()]
)

coefficients_matrix_clipped = torch.max(
    coefficients_matrix.abs(), 1e-5 * torch.ones_like(coefficients_matrix.abs())
)

plt.figure(figsize=(10, 6))
plt.imshow(
    coefficients_matrix_clipped.T,
    norm=mcolors.LogNorm(vmin=1e-5, vmax=1e0),
    aspect="auto",
)
plt.xticks(
    ticks=[i for i in range(coefficients_matrix.shape[0])],
    labels=[str(k) for k in coefficients.keys()],
    rotation=90,
)
plt.xlabel("Path")
plt.ylabel("Gamma structure index")
plt.colorbar()
plt.title(f"Clifford algebra coefficients\nmodel {model_name}, mass {mass}")
plt.savefig(
    f"plots/png/coefficients/coefficients_{model_name}_m{mass}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/coefficients/coefficients_{model_name}_m{mass}.pdf",
    bbox_inches="tight",
)

diff = (coefficients_matrix - hopping_coefficients_matrix).abs()
diff = torch.max(diff, 1e-5 * torch.ones_like(diff))

# Difference from hopping expansion coefficients
plt.figure(figsize=(10, 6))
plt.imshow(
    diff.T,
    norm=mcolors.LogNorm(vmin=1e-5, vmax=1e0),
    aspect="auto",
)
plt.xticks(
    ticks=[i for i in range(coefficients_matrix.shape[0])],
    labels=[str(k) for k in coefficients.keys()],
    rotation=90,
)
plt.xlabel("Path")
plt.ylabel("Gamma structure index")
plt.colorbar()
plt.title(
    f"Difference from hopping expansion coefficients\n"
    f"model {model_name}, mass {mass}"
)
plt.savefig(
    f"plots/png/coefficients/comparison_hopping_{model_name}_m{mass}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/coefficients/comparison_hopping_{model_name}_m{mass}.pdf",
    bbox_inches="tight",
)
