"""Calculate residuals on 16c32 lattice for a trained model.

Usage:
    get_residuals16c32.py --model=<name> --mass=<m> [--random]

Options:
    --model=<name>          Name of the model json file (in `models/{name}.json`)
    --mass=<mass>           Mass parameter value
    --random                Calculate residuals using random gauge fields
"""

import json
import os
import sys

import qcd_ml
import torch
from docopt import docopt
from model import *
from utility import get_config_path, get_gauge_field, model_paths

sys.path.insert(0, "scripts")

torch.set_grad_enabled(False)

# Parse docopt arguments
args = docopt(__doc__)
model_name = args["--model"]
mass = float(args["--mass"])
random = True if args["--random"] else False

# Load model json
model_json_path = f"models/{model_name}.json"
with open(model_json_path, "r") as f:
    model_json = json.load(f)

model_type = model_json["model_type"]
if model_type not in ["restricted", "4x4", "Clifford"]:
    raise ValueError(f"Model type '{model_type}' not supported!")
lattice_size = [16, 16, 16, 32]
nlayers = model_json["nlayers"]
test_configs = [282, 472, 329, 591]

print(f"Calculating residuals for model {model_name} with mass={mass:.2f}")

test_Us = [
    torch.load(get_config_path(lattice_size, c), weights_only=True)
    for c in test_configs
]
if random:
    test_Us = [get_gauge_field(lattice_size, 3, i) for i in range(4)]
test_ws = [qcd_ml.qcd.dirac.dirac_wilson(U, mass) for U in test_Us]

# Define model
match model_type:
    case "restricted":
        model = Model_restricted(nlayers, model_paths)
    case "4x4":
        model = Model_4x4(nlayers, model_paths)
    case _:
        model = Model_Clifford(nlayers, model_paths)

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load weights
weights_path = f"data/weights/weights_{model_name}_m{mass:.2f}.pt"
model.load_state_dict(
    torch.load(weights_path, weights_only=True, map_location=device)
)


def residual(w, winv_approx, x):
    return qcd_ml.util.linear_algebra.norm(w(winv_approx(x)) - x)


residuals = torch.zeros(len(test_ws), 100)
for sample_idx in range(100):
    v = torch.randn(*lattice_size, 4, 3, dtype=torch.cdouble)
    v /= qcd_ml.util.linear_algebra.norm(v)
    for test_idx, w in enumerate(test_ws):
        residuals[test_idx, sample_idx] = (
            residual(w, lambda x: model(x, w.U), v.clone()) ** 2
        )

# Ensure output directory exists
os.makedirs("data/residuals", exist_ok=True)

model_name = model_name.replace("8c16", "16c32")

outpath = f"data/residuals/residuals_{model_name}_m{mass:.2f}.pt"
if random:
    outpath = f"data/residuals/random_residuals_{model_name}_m{mass:.2f}.pt"

# Save output
torch.save(residuals, outpath)
