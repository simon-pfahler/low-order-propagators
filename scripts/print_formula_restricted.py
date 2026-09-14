"""Print the formula obtained from restricted model weights.

Usage:
    print_formula_restricted.py --model=<name> --mass=<mass>

Options:
    --model=<name>          Name of the model json file (in `models/{name}.json`)
    --mass=<mass>           Mass parameter value
"""

import json
import math
import os
import sys

import docopt
import torch
from docopt import docopt


def format_pdg(mean, std):
    """Format (mean, std) in particle physics notation.

    e.g. 3.27(8)e-3 means (3.27 +/- 0.08)e-3, where the bracketed
    number is the uncertainty in the last displayed digit(s).
    Uses 2 significant figures for the uncertainty when the leading
    digit is 1 or 2, otherwise 1 significant figure (PDG convention).
    """
    if std == 0:
        if mean == 0:
            return "0"
        exp = math.floor(math.log10(abs(mean)))
        norm = mean / 10**exp
        s = f"{norm:.2f}"
        return f"{s}e{exp:+d}" if exp != 0 else s

    std_exp = math.floor(math.log10(std))
    std_norm = round(std / 10**std_exp, 10)

    if std_norm < 3:
        last_digit_exp = std_exp - 1
    else:
        last_digit_exp = std_exp

    scale = 10**last_digit_exp

    if mean == 0:
        exp = last_digit_exp
    else:
        exp = math.floor(math.log10(abs(mean)))
        if exp < last_digit_exp:
            exp = last_digit_exp

    decimals = exp - last_digit_exp

    mean_rounded = round(mean / scale) * scale
    std_rounded = round(std / scale)

    mean_norm = mean_rounded / 10**exp
    result = f"{mean_norm:.{decimals}f}({int(std_rounded)})"
    if exp != 0:
        result += f"e{exp:+d}"
    return result


sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_name = args["--model"]
mass = args["--mass"]

nrseed = 5

# Load model json
model_json_path = f"models/{model_name}.json"
with open(model_json_path, "r") as f:
    model_json = json.load(f)
nlayers = model_json["nlayers"]

# Load weights and get coefficients
weights_paths = [
    f"data/weights/seeded_weights_{model_name}_m{mass}_seed{seed}.pt"
    for seed in range(nrseed)
]

weightss = [
    torch.load(
        f"data/weights/seeded_weights_{model_name}_m{mass}_seed{seed}.pt",
        weights_only=True,
    )
    for seed in range(nrseed)
]

overall_factors = torch.stack([w["overall_factor"] for w in weightss])
layer_weights = torch.stack([w["weights"] for w in weightss])

for seed in range(nrseed):
    for l in range(nlayers):
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
for l in range(nlayers):
    m0 = format_pdg(
        layer_weights_mean[l, 0].item(), layer_weights_std[l, 0].item()
    )
    m1 = format_pdg(
        layer_weights_mean[l, 1].item(), layer_weights_std[l, 1].item()
    )
    print(f"Layer {l} formula: ({m0} I + {m1} g)H+ + ({m0} I - {m1} g)H-")
