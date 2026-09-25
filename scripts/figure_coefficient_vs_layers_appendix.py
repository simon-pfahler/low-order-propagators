"""Create the main coefficient layer dependence plot of the paper.

Usage:
    figure_coefficient_vs_layers_appendix.py --path=<str> --gamma_index=<n>

Options:
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

plt.style.use("./scripts/iclr2027.mplstyle")

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_types = ["HC", "restricted"]
action = "WilsonQuenched"
lattice_size_str = "8c16"
masses = ["-5.00", "-4.00", "-3.00", "-2.00", "-1.00", "0.00", "1.00", "2.00"]
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

fig, ax = plt.subplots(4, 2, figsize=(9, 7))
ax = ax.flatten()

for idx in range(8):
    mass = masses[idx]

    for model_type in model_types:
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
                    canonical_path, new_indices, new_signs = canonicalize_path(
                        p
                    )
                    new_generator_indices, new_generator_signs = (
                        generator_mapping(new_indices, new_signs)
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
        hopping_coefficient.real,
        label=f"Hopping expansion",
        c="#33bbee",
    )

    ax[idx].grid(True, which="both", linestyle="--", alpha=0.5)
    if idx == 0:
        handles, labels = ax[idx].get_legend_handles_labels()
    if idx in [0, 2, 4, 6]:
        ax[idx].set_ylabel("Real part of coefficient")
    if idx in [6, 7]:
        ax[idx].set_xlabel("Layers")
    else:
        ax[idx].tick_params(labelbottom=False)
    ax[idx].set_xticks([2, 4, 6, 8, 12, 16])
    ax[idx].set_title(rf"$m={round(float(mass))}$")
fig.legend(
    handles,
    labels,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    handletextpad=0.5,
    columnspacing=1.5,
)
if path == [] and gamma_index == 0:
    fig.suptitle(rf"Evolution of $c_{{[],1}}$")
elif path == [(0, 1)] and gamma_index == 0:
    fig.suptitle(rf"Evolution of $c_{{[(\pm\mu,1)],1}}$")
elif path == [(0, 1)] and gamma_index == 1:
    fig.suptitle(rf"Evolution of $c_{{[(\pm\mu,1)],\gamma_\mu}}$")
elif path == [(0, 1), (1, 1), (0, -1)] and gamma_index == 0:
    fig.suptitle(rf"Evolution of $c_{{[(\pm\mu,1),(\pm\nu,1),(\mp\mu,1)],1}}$")
elif path == [(0, 3)] and gamma_index == 0:
    fig.suptitle(rf"Evolution of $c_{{[(\pm\mu,3)],1}}$")

os.makedirs("plots/coefficients/", exist_ok=True)

plt.savefig(
    f"plots/coefficients/coefficients_vs_layers_appendix_path{path}_g{gamma_index}.pdf",
    bbox_inches="tight",
)
