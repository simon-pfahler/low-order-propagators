"""Print the formula obtained from restricted model weights.

Usage:
    print_formula_restricted.py --layers=<n> --action=<name>  --lattice_size=<str> --mass=<mass>

Options:
    --layers=<n>            Number of layers
    --action=<name>         Action name ("WilsonQuenched", "WilsonDynamic" or "Haar")
    --lattice_size=<str>    Lattice size string (e.g. "8c16")
    --mass=<mass>           Mass parameter value
"""

import json
import os
import sys

import docopt
import torch
from docopt import docopt
from utility import format_pdg

# Parse docopt arguments
args = docopt(__doc__)
layers = int(args["--layers"])
model_type = "restricted"
action = args["--action"]
lattice_size_str = args["--lattice_size"]
mass = args["--mass"]

nrseeds = 5

model_name = f"{layers}layers_{action}_{lattice_size_str}_{model_type}"

# Load weights and get coefficients
weightss = [
    torch.load(
        f"data/weights/seeded_weights_{model_name}_m{mass}_seed{seed}.pt",
        weights_only=True,
    )
    for seed in range(nrseeds)
]

overall_factors = torch.stack([w["overall_factor"] for w in weightss])
layer_weights = torch.stack([w["weights"] for w in weightss])

for seed in range(nrseeds):
    for l in range(layers):
        layer = layer_weights[seed][l]
        all_angles = layer.angle()
        angle = torch.mean(all_angles)
        layer_weights[seed][l] *= torch.exp(-1j * angle)
        overall_factors[seed] *= torch.exp(1j * angle)

    # Sort layers so the sum of absolute values increases with layer index
    sort_order = torch.argsort(-torch.norm(layer_weights[seed], dim=1))
    layer_weights[seed] = layer_weights[seed][sort_order]

overall_factor_mean = torch.mean(overall_factors)
layer_weights_mean = torch.mean(layer_weights, dim=0)
overall_factor_std_real = torch.std(overall_factors.real)
overall_factor_std_imag = torch.std(overall_factors.imag)
layer_weights_std_real = torch.std(layer_weights.real, dim=0)
layer_weights_std_imag = torch.std(layer_weights.imag, dim=0)

mr = format_pdg(
    overall_factor_mean.item().real, overall_factor_std_real.item()
)[0]
mi, z = format_pdg(
    overall_factor_mean.item().imag, overall_factor_std_imag.item()
)
m = f"{mr}"
if not z:
    m += f"+{mi}i"
print(f"Overall factor: {m}")
for l in range(layers):
    mr0 = format_pdg(
        layer_weights_mean[l, 0].item().real,
        layer_weights_std_real[l, 0].item(),
    )[0]
    mr1 = format_pdg(
        layer_weights_mean[l, 1].item().real,
        layer_weights_std_real[l, 1].item(),
    )[0]
    mi0, z0 = format_pdg(
        layer_weights_mean[l, 0].item().imag,
        layer_weights_std_imag[l, 0].item(),
    )
    mi1, z1 = format_pdg(
        layer_weights_mean[l, 1].item().imag,
        layer_weights_std_imag[l, 1].item(),
    )
    m0 = f"{mr0}"
    if not z0:
        m0 += f"+{mi0}i"
    m1 = f"{mr1}"
    if not z1:
        m1 += f"+{mi1}i"
    print(f"Layer {l} formula: ({m0} I + {m1} g)H+ + ({m0} I - {m1} g)H-")
