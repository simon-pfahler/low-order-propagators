"""Create the coefficient mass dependence of HC models plots of the paper.

Usage:
    figure_coefficient_mass_dependence_appendix.py --path_length=<n>

Options:
    --path_length=<n>       Length of paths
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

# Parse docopt arguments
args = docopt(__doc__)
model_type = "HC"
action = "WilsonQuenched"
lattice_size_str = "8c16"
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

if path_length == 0:
    categories = [0]

if path_length == 1:
    categories = [0, 1]

if path_length == 2:
    categories = [0, 1, 2, 3, 4, 5]

fig, ax = plt.subplots(2, 2, figsize=(9, 6))
ax = [(), ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
if path_length == 2:
    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    ax = [(), (), ax[0], ax[1], ax[2]]
plt.subplots_adjust(wspace=0.1, hspace=0.15)

for layers in range(1, 5):
    if layers < path_length:
        continue
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
            if path_length == 0:
                scatter_points[0][mass_idx].append(v[0].real)
                for i in range(1, 16):
                    scatter_points[-1][mass_idx].append(v[i].real)
            if path_length == 1:
                scatter_points[0][mass_idx].append(v[0].real)
                scatter_points[1][mass_idx].append(torch.abs(v[1].real))
                for i in range(2, 16):
                    scatter_points[-1][mass_idx].append(v[i].real)
            if path_length == 2:
                if ck == [(0, 2)]:
                    scatter_points[0][mass_idx].append(v[0].real)
                    scatter_points[1][mass_idx].append(v[1].real)
                    for i in range(2, 16):
                        scatter_points[-1][mass_idx].append(v[i].real)
                if ck == [(0, 1), (1, 1)]:
                    scatter_points[2][mass_idx].append(v[0].real)
                    scatter_points[3][mass_idx].append(v[1].real)
                    scatter_points[4][mass_idx].append(v[2].real)
                    for i in range(3, 16):
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
    for idx in range(len(scatter_points)):
        s = torch.tensor(scatter_points[idx])
        m = torch.tensor(masses).unsqueeze(-1).expand(s.shape)
        if s.numel() == 0:
            continue
        if idx != len(scatter_points) - 1:
            ymax = max(ymax, torch.max(s).item())
            ymin = min(ymin, torch.min(s).item())

        ax[layers].scatter(
            m, s, zorder=10 - idx, s=6, c=colors[idx], rasterized=True
        )

    for factor in range(path_length):
        ax[layers].plot(
            *get_reference_curve(path_length, 2**factor),
            c="#33bbee",
            zorder=15,
        )
    if path_length == 0:
        ax[layers].plot(
            *get_reference_curve(0, 1),
            c="#33bbee",
            zorder=15,
        )

    layertext = "layer" if layers == 1 else "layers"
    if path_length == 2:
        if layers == 2:
            ax[layers].set_ylabel("Real part of coefficient")
        else:
            ax[layers].tick_params(labelleft=False)
        ax[layers].set_xlabel("Mass")
    else:
        if layers in [3, 4]:
            ax[layers].set_xlabel("Mass")
        else:
            ax[layers].tick_params(labelbottom=False)
        if layers in [1, 3]:
            ax[layers].set_ylabel("Real part of coefficient")
        else:
            ax[layers].tick_params(labelleft=False)
    if path_length == 0:
        ax[layers].set_ylim(-0.155, 0.34)
        ax[layers].set_title(
            rf"Dependence of $c_{{[],k}}$ on $m$, {layers} {layertext}"
        )
        ax[layers].legend([r"$c_{[],1}$", "others", r"$\frac1{m+4}$"])
    elif path_length == 1:
        ax[layers].set_ylim(-0.08, 0.11)
        ax[layers].set_title(
            rf"Dependence of $c_{{[(\pm\mu,1)],k}}$ on $m$, {layers} {layertext}"
        )
        ax[layers].legend(
            [
                r"$c_{[(\pm\mu,1)],1}$",
                r"$c_{[(\pm\mu,1)],\gamma_\mu}$",
                "others",
                r"$\frac1{2(m+4)^2}$",
            ]
        )
    elif path_length == 2:
        ax[layers].set_ylim(-0.02, 0.02)
        ax[layers].set_title(
            r"Dependence of $c_{[(\pm\mu,2)],k}$ and"
            f"\n"
            r"$c_{[(\pm\mu,1),(\pm\nu,1)],k}$ "
            r"on $m$, "
            f"{layers} "
            f"{layertext}"
        )
        ax[layers].legend(
            [
                r"$c_{[(\pm\mu,2)],1}$",
                r"$c_{[(\pm\mu,2)],\gamma_\mu}$",
                r"$c_{[(\pm\mu,1),(\pm\nu,1)],1}$",
                r"$c_{[(\pm\mu,1),(\pm\nu,1)],\gamma_\mu}$",
                r"$c_{[(\pm\mu,1),(\pm\nu,1)],\gamma_\nu}$",
                "others",
                "hopping",
            ],
            loc="lower right",
        ).set_zorder(20)
    ax[layers].grid(True, which="both", linestyle="--", alpha=0.5)

os.makedirs("plots/mass_dependence", exist_ok=True)

plt.savefig(
    f"plots/mass_dependence/coefficient_mass_dependence_pathlength{path_length}.pdf",
    bbox_inches="tight",
    dpi=600,
)
