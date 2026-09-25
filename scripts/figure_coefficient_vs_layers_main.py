"""Create the main coefficient layer dependence plot of the paper."""

import ast
import json
import os
import sys

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from utility import canonicalize_path, generator_mapping, get_path_length

plt.style.use("./scripts/iclr2027.mplstyle")

sys.path.insert(0, "scripts")

# Parse docopt arguments
model_type = "HC"
action = "WilsonQuenched"
lattice_size_str = "8c16"
mass = "-3.00"
gamma_index = 0

nrseeds = 5

fig, ax = plt.subplots(1, 2, figsize=(9, 2.3))

for idx in range(2):
    path = []
    if idx == 1:
        path = [(0, 1), (1, 1), (0, -1)]

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

    symbol = "s"
    color = "#ee7733"
    if model_type == "restricted":
        symbol = "D"
        color = "#0077bb"

    ax[idx].errorbar(
        layerss,
        coefficients.real.mean(dim=0),
        yerr=coefficients.real.std(dim=0),
        linestyle="none",
        color=color,
        capsize=5,
        marker=symbol,
        markerfacecolor="none",
        label=f"{model_type} model",
    )

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

    ax[idx].axhline(
        hopping_coefficient.real, label=f"Hopping expansion", c="#33bbee"
    )

    ax[idx].grid(True, which="both", linestyle="--", alpha=0.5)
    ax[idx].legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    ax[idx].set_xlabel("Layers")
    if idx == 0:
        ax[idx].set_xticks([2, 4, 6, 8, 12, 16])
        ax[idx].set_ylim(0, 1.1)
    else:
        ax[idx].set_xticks([4, 6, 8, 12, 16])
    if idx == 0:
        ax[idx].set_ylabel("Real part of coefficient")
    if idx == 0:
        ax[idx].set_title(
            rf"Evolution of $c_{{[],1}}$ for $m={round(float(mass))}$"
        )
    else:
        ax[idx].set_title(
            rf"Evolution of $c_{{[(\pm\mu,1),(\pm\nu,1),(\mp\mu,1)],1}}$ for $m={round(float(mass))}$"
        )

os.makedirs("plots/coefficients/", exist_ok=True)

plt.savefig(
    f"plots/coefficients/coefficients_vs_layers_main.pdf",
    bbox_inches="tight",
)
