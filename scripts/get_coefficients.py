"""Get coefficients of a model or the hopping expansion

Usage:
    get_coefficients.py --layers=<n> --type=<name> [--action=<name>] [--lattice_size=<str>] --mass=<mass> [--seed=<n>]

Options:
    --layers=<n>                Number of layers of the model, or number of steps of the hopping expansion/GMRES
    --type=<name>               Model type ("HC", "HL", "restricted") or "hopping"
    --action=<name>             Action the model was trained on
    --lattice_size=<str>        Lattice size the model was trained on
    --mass=<mass>               Mass parameter value
    --seed=<n>                  Seed used for training
"""

import os
import sys

import torch
from docopt import docopt
from utility import (
    get_coefficients_from_weights,
    get_hopping_weights,
    get_weights_from_HC,
    get_weights_from_HL,
    get_weights_from_restricted,
)

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
layers = int(args["--layers"])
model_type = args["--type"]
action = args["--action"]
lattice_size_str = args["--lattice_size"]
if model_type not in ["restricted", "HL", "HC", "hopping"]:
    raise ValueError(f"Type '{model_type}' not supported!")
mass = args["--mass"]
seed = int(args["--seed"]) if args["--seed"] is not None else None

match model_type:
    case "restricted":
        if seed is None:
            weights = torch.load(
                f"data/weights/weights_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}.pt",
                weights_only=True,
            )
        else:
            weights = torch.load(
                f"data/weights/seeded_weights_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}_seed{seed}.pt",
                weights_only=True,
            )
        weights = get_weights_from_restricted(weights)
    case "HL":
        if seed is None:
            weights = torch.load(
                f"data/weights/weights_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}.pt",
                weights_only=True,
            )
        else:
            weights = torch.load(
                f"data/weights/seeded_weights_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}_seed{seed}.pt",
                weights_only=True,
            )
        weights = get_weights_from_HL(weights)
    case "HC":
        if seed is None:
            weights = torch.load(
                f"data/weights/weights_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}.pt",
                weights_only=True,
            )
        else:
            weights = torch.load(
                f"data/weights/seeded_weights_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}_seed{seed}.pt",
                weights_only=True,
            )
        weights = get_weights_from_HC(weights)
    case _:
        weights = get_hopping_weights(float(mass), layers)

coefficients = get_coefficients_from_weights(
    weights,
    path_length_min=0,
    path_length_max=4,
)

os.makedirs("data/coefficients", exist_ok=True)

if model_type in ["restricted", "HL", "HC"]:
    if seed is None:
        out_name = f"data/coefficients/coefficients_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}.pt"
    else:
        out_name = f"data/coefficients/seeded_coefficients_{layers}layers_{action}_{lattice_size_str}_{model_type}_m{mass}_seed{seed}.pt"
else:
    out_name = (
        f"data/coefficients/coefficients_{layers}layers_hopping_m{mass}.pt"
    )

torch.save(coefficients, out_name)
