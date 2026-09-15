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
        if torch.any(
            torch.abs(layer_weights[seed][l].imag / layer_weights[seed][l].real)
            > 1e-1
        ):
            print(
                f"Warning seed {seed}: Layer {l} weights are complex even after rotation!"
            )
            print(
                f"{torch.abs(layer_weights[seed][l].imag/layer_weights[seed][l].real)}"
            )

    if (
        torch.abs(overall_factors[seed].imag / overall_factors[seed].real)
        > 1e-1
    ):
        print(f"Warning: Overall factor is complex even after rotation!")
        print(
            f"{torch.abs(overall_factors[seed].imag/overall_factors[seed].real)}"
        )

    overall_factors[seed] = overall_factors[seed].item().real
    layer_weights[seed] = layer_weights[seed].real

    # Sort layers so the sum of absolute values increases with layer index
    sort_order = torch.argsort(
        -torch.sum(torch.abs(layer_weights[seed]), dim=1)
    )
    layer_weights[seed] = layer_weights[seed][sort_order]

overall_factor_mean = torch.mean(overall_factors.real.to(torch.double))
layer_weights_mean = torch.mean(layer_weights.real.to(torch.double), dim=0)
overall_factor_std = torch.std(overall_factors.real.to(torch.double))
layer_weights_std = torch.std(layer_weights.real.to(torch.double), dim=0)

print(
    f"Overall factor: {format_pdg(overall_factor_mean.item(), overall_factor_std.item())}"
)
for l in range(layers):
    m0 = format_pdg(
        layer_weights_mean[l, 0].item(), layer_weights_std[l, 0].item()
    )
    m1 = format_pdg(
        layer_weights_mean[l, 1].item(), layer_weights_std[l, 1].item()
    )
    print(f"Layer {l} formula: ({m0} I + {m1} g)H+ + ({m0} I - {m1} g)H-")
