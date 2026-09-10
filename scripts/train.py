"""Train a model specified in a json file.

Usage:
    train.py --model=<name> --mass=<mass> [--training-steps=<n>] [--lr=<f>] [--seed=<n>]

Options:
    --model=<name>          Name of the model json file (in `models/{name}.json`)
    --mass=<mass>           Mass parameter value
    --training-steps=<n>    Number of training steps [default: 10000]
    --lr=<f>                Learning rate of optimizer [default: 1e-3]
    --seed=<n>              Seed of the training
"""

import json
import os
import sys

import numpy as np
import qcd_ml
import torch
from docopt import docopt
from model import *
from utility import get_config_path, model_paths

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_name = args["--model"]
mass = float(args["--mass"])
training_steps = int(args["--training-steps"])
lr = float(args["--lr"])
seed = int(args["--seed"]) if args["--seed"] else None

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

print(f"Starting training of model {model_name} with mass={mass:.2f}")

# Get config paths and load configs
train_Us = [
    torch.load(get_config_path(lattice_size, c), weights_only=True)
    for c in train_configs
]
test_Us = [
    torch.load(get_config_path(lattice_size, c), weights_only=True)
    for c in test_configs
]
train_ws = [qcd_ml.qcd.dirac.dirac_wilson(U, mass) for U in train_Us]
test_ws = [qcd_ml.qcd.dirac.dirac_wilson(U, mass) for U in test_Us]

# Define model
match model_type:
    case "restricted":
        model = Model_restricted(nlayers, model_paths)
    case "4x4":
        model = Model_4x4(nlayers, model_paths)
    case _:
        model = Model_Clifford(nlayers, model_paths)

# Initialize weights
for p in model.parameters():
    p.data *= 1e-3

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create output filenames
weights_path = f"data/weights/weights_{model_name}_m{mass:.2f}.pt"
history_path = f"data/histories/history_{model_name}_m{mass:.2f}.txt"
if seed is not None:
    weights_path = (
        f"data/weights/weights_{model_name}_m{mass:.2f}_seed{seed}.pt"
    )
    history_path = (
        f"data/histories/history_{model_name}_m{mass:.2f}_seed{seed}.txt"
    )
    torch.manual_seed(seed)

# Ensure output directories exist
os.makedirs("data/weights", exist_ok=True)
os.makedirs("data/histories", exist_ok=True)

# Header of history file
f_history = open(history_path, "w")
f_history.write(f"# Training history for {model_name}\n")
f_history.write(f"# mass={mass}, lr={lr}, training_steps={training_steps}\n")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

test_cost = torch.inf

old_state = torch.random.get_rng_state()
torch.manual_seed(1337)
v_test = torch.randn(*lattice_size, 4, 3, dtype=torch.cdouble)
v_test /= qcd_ml.util.linear_algebra.norm(v_test)
torch.random.set_rng_state(old_state)

for t in range(training_steps + 1):
    v = torch.randn(*lattice_size, 4, 3, dtype=torch.cdouble)
    v /= qcd_ml.util.linear_algebra.norm(v)

    diffs = [model.forward(w(v), U) - v for w, U in zip(train_ws, train_Us)]
    cost = sum(
        qcd_ml.util.linear_algebra.innerproduct(diff, diff).real
        for diff in diffs
    ) / len(train_configs)

    optimizer.zero_grad()
    cost.backward()

    # Save training cost, and check for convergence
    if t % 100 == 0:
        # Every 100 iterations, also compute and write test cost
        with torch.no_grad():
            diffs_test = [
                model.forward(w(v_test), U) - v_test
                for w, U in zip(test_ws, test_Us)
            ]
            new_test_cost = sum(
                qcd_ml.util.linear_algebra.innerproduct(diff, diff).real
                for diff in diffs_test
            ) / len(test_configs)

            f_history.write(f"{t} {cost} {new_test_cost}\n")

            # check for convergence
            if abs(test_cost - new_test_cost) / test_cost < 1e-3:
                break

            test_cost = new_test_cost
    else:
        # Otherwise, just write the training cost
        f_history.write(f"{t} {cost}\n")

    optimizer.step()

# Save final weights
torch.save(model.state_dict(), weights_path)

print(f"Training complete. Weights saved to {weights_path}")
