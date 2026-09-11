"""Plot the mass dependence of coefficients for length-n paths.

Usage:
    plot_mass_dependence_coefficients.py --model_type=<type> --volume=<vol> --nlayers=<n> --path_length=<n>

Options:
    --model_type=<type>     Model type: 'Clifford' or 'restricted'
    --volume=<vol>          Volume string
    --nlayers=<n>           Number of layers
    --path_length=<n>       Length of the paths to create the plot for
"""

import os
import re
import sys
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import torch
from docopt import docopt
from utility import consolidate_path, generators, get_path_length, model_paths


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
model_type = args["--model_type"]
volume = args["--volume"]
if model_type not in ["Clifford", "restricted"]:
    raise ValueError(f"Model type '{model_type}' not supported!")
nlayers = int(args["--nlayers"])
path_length = int(args["--path_length"])

# Find all masses
masses = sorted(set(extract_mass(f) for f in os.listdir("data/weights")))

# Get hopping coefficients for mass 1
hopping_coefficients = torch.load(
    f"data/coefficients/coefficients_hopping_m1.00.pt", weights_only=True
)

# get categories of paths:
# First, match different directions, then deal with rotations
categories_raw = list()
for k, v in hopping_coefficients.items():
    cat_idx = None
    for idx, category in enumerate(categories_raw):
        if torch.all(
            torch.abs(v) == torch.abs(hopping_coefficients[category[0]])
        ):
            cat_idx = idx
            break
    if cat_idx is None:
        categories_raw.append([k])
    else:
        categories_raw[cat_idx].append(k)

categories_raw_lengths = []
for c in categories_raw:
    if len(c) not in categories_raw_lengths:
        categories_raw_lengths.append(len(c))
categories = [[] for _ in range(len(categories_raw_lengths))]
for c in categories_raw:
    idx = categories_raw_lengths.index(len(c))
    categories[idx].extend(c)
category_index = dict()
for i, c in enumerate(categories):
    for e in c:
        category_index[tuple(e)] = i

category_names = [c[0] for c in categories] + ["zero"]

scatter_points = [[[] for _ in masses] for _ in range(len(categories) + 1)]
for mass_idx, mass in enumerate(masses):
    coefficients = torch.load(
        f"data/coefficients/coefficients_{nlayers}layers_{volume}_{model_type}_m{mass:.2f}.pt",
        weights_only=True,
    )

    for k, v in coefficients.items():
        hopping_v = hopping_coefficients[k]
        for i in range(16):
            if hopping_v[i] != 0:
                scatter_points[category_index[tuple(k)]][mass_idx].append(
                    torch.abs(v[i])
                )
            else:
                scatter_points[-1][mass_idx].append(torch.abs(v[i]))

plt.figure(figsize=(10, 6))
ymin = 0
ymax = 1e3
for idx in range(len(scatter_points)):
    s = torch.tensor(scatter_points[idx])
    m = torch.tensor(masses).unsqueeze(-1).expand(s.shape)
    if s.numel() == 0:
        continue
    if idx != len(scatter_points) - 1:
        ymax = min(ymax, torch.max(s).item())
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
    f"for {nlayers} layers, {volume}, {model_type}\n"
    f"Coefficients for paths of length {path_length}"
)
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

os.makedirs("plots/png/mass_dependence", exist_ok=True)
os.makedirs("plots/pdf/mass_dependence", exist_ok=True)

plt.savefig(
    f"plots/png/mass_dependence/mass_dependence_{nlayers}layers_{volume}_{model_type}_pathlength{path_length}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"plots/pdf/mass_dependence/mass_dependence_{nlayers}layers_{volume}_{model_type}_pathlength{path_length}.pdf",
    bbox_inches="tight",
)
