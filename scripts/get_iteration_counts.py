"""Perform test solves using a model as perconditioner.

Usage:
    get_iteration_count.py --model=<name> --mass=<mass>

Options:
    --model=<name>          Name of the model json file (in `models/{name}.json`)
    --mass=<mass>           Mass parameter value
"""

import json
import os
import sys

import qcd_ml
import torch
from docopt import docopt
from model import *
from utility import get_config_path, model_paths

sys.path.insert(0, "scripts")

torch.set_grad_enabled(False)

# Parse docopt arguments
args = docopt(__doc__)
model_name = args["--model"]
mass = float(args["--mass"])

# Load model json
model_json_path = f"models/{model_name}.json"
with open(model_json_path, "r") as f:
    model_json = json.load(f)

model_type = model_json["model_type"]
if model_type not in ["restricted", "4x4", "Clifford"]:
    raise ValueError(f"Model type '{model_type}' not supported!")
lattice_size = model_json["lattice_size"]
nlayers = model_json["nlayers"]
train_configs = model_json["train_configs"]
test_configs = model_json["test_configs"]

print(f"Test GMRES solves for model {model_name} with mass={mass:.2f}")

test_Us = [
    torch.load(get_config_path(lattice_size, c), weights_only=True)
    for c in test_configs
]
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

iteration_counts = torch.zeros(len(test_ws), 5)
for sample_idx in range(5):
    v = torch.randn(*lattice_size, 4, 3, dtype=torch.cdouble)
    v /= qcd_ml.util.linear_algebra.norm(v)
    for test_idx, w in enumerate(test_ws):
        iteration_counts[test_idx, sample_idx] = len(
            qcd_ml.util.solver.GMRES(
                w,
                v.clone(),
                torch.zeros_like(v),
                maxiter=1000,
                eps=1e-8,
                preconditioner=lambda x: model(x, w.U),
            )[1]["history"]
        )

# Ensure output directory exists
os.makedirs("data/iteration_counts", exist_ok=True)

# Save output
torch.save(
    iteration_counts,
    f"data/iteration_counts/iteration_counts_{model_name}_m{mass:.2f}.pt",
)
