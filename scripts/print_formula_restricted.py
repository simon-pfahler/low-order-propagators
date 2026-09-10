"""Print the formula obtained from restricted model weights.

Usage:
    print_formula_restricted.py --model=<name> --mass=<mass>

Options:
    --model=<name>          Name of the model json file (in `models/{name}.json`)
    --mass=<mass>           Mass parameter value
"""

import json
import os
import sys

import docopt
import torch
from docopt import docopt

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_name = args["--model"]
mass = args["--mass"]

# Load model json
model_json_path = f"models/{model_name}.json"
with open(model_json_path, "r") as f:
    model_json = json.load(f)
nlayers = model_json["nlayers"]

# Load weights and get coefficients
weights_path = f"data/weights/weights_{model_name}_m{mass}.pt"
if not os.path.exists(weights_path):
    raise ValueError(
        f"No weights file found for model '{model_name}' and mass={mass}"
    )

weights = torch.load(
    f"data/weights/weights_{model_name}_m{mass}.pt",
    weights_only=True,
)

overall_factor = weights["overall_factor"]
layer_weights = weights["weights"]

layers = layer_weights.shape[0]

angles = torch.zeros(layers)
for l in range(layers):
    layer = layer_weights[l]
    all_angles = layer.angle()
    neg_mask = layer.real < 0
    all_angles[neg_mask] = (-layer[neg_mask]).angle()
    angles[l] = torch.mean(all_angles)
    layer_weights[l] *= torch.exp(-1j * angles[l])
    overall_factor *= torch.exp(1j * angles[l])
    if torch.any(
        torch.abs(layer_weights[l].imag / layer_weights[l].real) > 1e-2
    ):
        print(f"Warning: Layer {l} weights are complex even after rotation!")

if torch.abs(overall_factor.imag / overall_factor.real) > 1e-2:
    print(f"Warning: Overall factor is complex even after rotation!")

overall_factor = overall_factor.item().real
layer_weights = layer_weights.real

print(f"Overall factor: {overall_factor:.2e}")
for l in range(layers):
    print(
        f"Layer {l} formula: ({layer_weights[l,0]:.2e} I + {layer_weights[l,1]:.2e} g)H+ + ({layer_weights[l,0]:.2e} I - {layer_weights[l,1]:.2e} g)H-)"
    )
