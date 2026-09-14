"""Get coefficients of a model or the hopping expansion

Usage:
    plot_categories.py [--model=<name>] --mass=<mass>

Options:
    --model=<name>          Name of the model json file (in `models/{name}.json`)
    --mass=<mass>           Mass parameter value
"""

import os
import sys

import torch
from docopt import docopt
from utility import (
    get_coefficients_from_weights,
    get_hopping_weights,
    get_weights_from_4x4,
    get_weights_from_Clifford,
    get_weights_from_restricted,
)

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_name = args["--model"]
mass = args["--mass"]

if model_name:
    weights = torch.load(
        f"data/weights/weights_{model_name}_m{mass}.pt", weights_only=True
    )
    match model_name.split("_")[-1]:
        case "4x4":
            weights = get_weights_from_4x4(weights)
        case "restricted":
            weights = get_weights_from_restricted(weights)
        case "Clifford":
            weights = get_weights_from_Clifford(weights)
else:
    weights = get_hopping_weights(float(mass), 4)

coefficients = get_coefficients_from_weights(
    weights,
    path_length_min=0,
    path_length_max=4,
)

os.makedirs("data/coefficients", exist_ok=True)

if model_name:
    out_name = f"data/coefficients/coefficients_{model_name}_m{mass}.pt"
else:
    out_name = f"data/coefficients/coefficients_hopping_m{mass}.pt"

torch.save(coefficients, out_name)
