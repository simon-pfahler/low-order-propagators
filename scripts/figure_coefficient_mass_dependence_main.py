"""Create the coefficient mass dependence of HC vs restricted models plots of the paper.

Usage:
    figure_coefficient_mass_dependence_main.py
"""

import os
import re
import sys
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import torch
from utility import (
    canonicalize_path,
    consolidate_path,
    generator_mapping,
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


plt.style.use("./scripts/iclr2027.mplstyle")

sys.path.insert(0, "scripts")

model_types = ["HC", "restricted"]
action = "WilsonQuenched"
lattice_size_str = "8c16"
layers = 4
path_length = 0

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

categories = [0]

fig, ax = plt.subplots(1, 2, figsize=(9, 2.3))
plt.subplots_adjust(wspace=0.1, hspace=0.15)

for idx in range(2):
    model_type = model_types[idx]
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
            ck, new_indices, new_signs = canonicalize_path(k)
            new_generator_indices, new_generator_signs = generator_mapping(
                new_indices, new_signs
            )
            v = torch.stack(
                [
                    new_generator_signs[gamma_index]
                    * coefficients[k][new_generator_indices[gamma_index]]
                    for gamma_index in range(16)
                ]
            )
            scatter_points[0][mass_idx].append(v[0].real)
            for i in range(1, 16):
                scatter_points[-1][mass_idx].append(v[i].real)

    ymin = 0
    ymax = 0
    colors = [
        "#cc3311",
        "#009988",
        "#0077bb",
        "#ee7733",
        "#ee3377",
        "#33bbee",
        "#bbbbbb",
    ]
    colors = colors[: len(scatter_points) - 1] + [colors[-1]]
    for jdx in range(len(scatter_points)):
        s = torch.tensor(scatter_points[jdx])
        m = torch.tensor(masses).unsqueeze(-1).expand(s.shape)
        if s.numel() == 0:
            continue
        if jdx != len(scatter_points) - 1:
            ymax = max(ymax, torch.max(s).item())
            ymin = min(ymin, torch.min(s).item())

        ax[idx].scatter(m, s, zorder=10 - jdx, c=colors[jdx])

    for factor in range(path_length):
        ax[idx].plot(
            *get_reference_curve(path_length, 2**factor),
            c="#33bbee",
            zorder=15,
        )
    ax[idx].plot(
        *get_reference_curve(0, 1),
        c="#33bbee",
        zorder=15,
    )

    ax[idx].set_xlabel("Mass")
    if idx == 0:
        ax[idx].set_ylabel("Real part of coefficient")
    else:
        ax[idx].tick_params(labelleft=False)
    ax[idx].set_ylim(-0.155, 0.34)
    if idx == 0:
        ax[idx].set_title(
            rf"Dependence of $c_{{[],k}}$ on $m$, 4-layer HC model"
        )
    else:
        ax[idx].set_title(
            rf"Dependence of $c_{{[],k}}$ on $m$, 4-layer restricted model"
        )
    ax[idx].legend([r"$c_{[],1}$", r"$c_{[],k\ne1}$", r"$\frac1{m+4}$"])
    ax[idx].grid(True, which="both", linestyle="--", alpha=0.5)

os.makedirs("plots/mass_dependence", exist_ok=True)

plt.savefig(
    f"plots/mass_dependence/coefficient_mass_dependence.pdf",
    bbox_inches="tight",
)
