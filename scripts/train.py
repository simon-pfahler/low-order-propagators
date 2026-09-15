"""Train a model.

Usage:
    train.py --layers=<n> --model_type=<name> --action=<name> --lattice_size=<str> --mass=<mass> [--training-steps=<n>] [--lr=<f>] [--seed=<n>]

Options:
    --layers=<n>            Number of layers
    --model_type=<name>     Model type ("HC", "HL", "restricted")
    --action=<name>         Action name ("WilsonQuenched", "WilsonDynamic" or "Haar")
    --lattice_size=<str>    Lattice size string (e.g. "8c16")
    --mass=<mass>           Mass parameter value
    --training-steps=<n>    Number of training steps [default: 10000]
    --lr=<f>                Learning rate of optimizer [default: 1e-3]
    --seed=<n>              Seed of the training
"""

import os
import sys

import numpy as np
import qcd_ml
import torch
from docopt import docopt
from model import *
from utility import get_config_paths, get_hopping_weights, model_paths

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
layers = int(args["--layers"])
model_type = args["--model_type"]
action = args["--action"]
lattice_size_str = args["--lattice_size"]
if model_type not in ["restricted", "HL", "HC"]:
    raise ValueError(f"Model type '{model_type}' not supported!")
mass = float(args["--mass"])
training_steps = int(args["--training-steps"])
lr = float(args["--lr"])
seed = int(args["--seed"]) if args["--seed"] else None

model_name = f"{layers}layers_{action}_{lattice_size_str}_{model_type}"
lattice_size = (
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[1]),
)

train_config_paths = get_config_paths(action, lattice_size_str)
test_config_paths = get_config_paths(action, lattice_size_str, test=True)

print(f"Starting training of model {model_name} with mass={mass:.2f}")

# Get config paths and load configs
train_Us = [torch.load(p, weights_only=True) for p in train_config_paths]
test_Us = [torch.load(p, weights_only=True) for p in test_config_paths]
train_ws = [qcd_ml.qcd.dirac.dirac_wilson(U, mass) for U in train_Us]
test_ws = [qcd_ml.qcd.dirac.dirac_wilson(U, mass) for U in test_Us]

# Define model and initialize weights
match model_type:
    case "restricted":
        model = Model_restricted(layers, model_paths)
        for p in model.parameters():
            p.data *= 1e-4
        if mass > -0.5:
            lr *= 0.1
            kappa = 1 / (2 * mass + 8)
            model.overall_factor.data += (
                2 * kappa * torch.ones_like(model.overall_factor)
            )
            for wi in range(model.weights.shape[0]):
                model.weights.data[wi, 0] += kappa
                model.weights.data[wi, 1] += kappa
    case "HC":
        model = Model_HC(layers, model_paths)
        for p in model.parameters():
            p.data *= 1e-4
        if mass > -0.5:
            lr *= 0.1
            kappa = 1 / (2 * mass + 8)
            model.weights[0].data[0, :, 0] += 2 * kappa
            model.weights[0].data[0, 0, 0] -= 1
            for wi in range(1, len(model.weights)):
                for mu in range(4):
                    model.weights[wi].data[2 * mu + 1, :, 0] += kappa
                    model.weights[wi].data[2 * mu + 1, :, mu + 1] += kappa
                    model.weights[wi].data[2 * mu + 2, :, 0] += kappa
                    model.weights[wi].data[2 * mu + 2, :, mu + 1] += -kappa
    case _:
        model = Model_HL(layers, model_paths)
        for p in model.parameters():
            p.data *= 1e-4
        if mass > 0:
            lr *= 0.1
            kappa = 1 / (2 * mass + 8)
            model.weights[0].data[0, :] += 2 * kappa * generators[0]
            for wi in range(1, len(model.weights)):
                for mu in range(4):
                    model.weights[wi].data[2 * mu + 1, :] += kappa * (
                        generators[0] + generators[mu + 1]
                    )
                    model.weights[wi].data[2 * mu + 2, :] += kappa * (
                        generators[0] - generators[mu + 1]
                    )

# Create output filenames
weights_path = f"data/weights/weights_{model_name}_m{mass:.2f}.pt"
history_path = f"data/histories/history_{model_name}_m{mass:.2f}.txt"
if seed is not None:
    weights_path = (
        f"data/weights/seeded_weights_{model_name}_m{mass:.2f}_seed{seed}.pt"
    )
    history_path = (
        f"data/histories/seeded_history_{model_name}_m{mass:.2f}_seed{seed}.txt"
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
v_tests = [
    torch.randn(*lattice_size, 4, 3, dtype=torch.cdouble)
    for _ in range(len(test_Us))
]
v_tests = [v / qcd_ml.util.linear_algebra.norm(v) for v in v_tests]
torch.random.set_rng_state(old_state)

for t in range(training_steps + 1):
    vs = [
        torch.randn(*lattice_size, 4, 3, dtype=torch.cdouble)
        for _ in range(len(train_Us))
    ]
    vs = [v / qcd_ml.util.linear_algebra.norm(v) for v in vs]

    diffs = [
        model.forward(w(v), U) - v for v, w, U in zip(vs, train_ws, train_Us)
    ]
    cost = sum(
        qcd_ml.util.linear_algebra.innerproduct(diff, diff).real
        for diff in diffs
    ) / len(train_config_paths)

    optimizer.zero_grad()
    cost.backward()

    # Save training cost, and check for convergence
    if t % 100 == 0:
        # Every 100 iterations, also compute and write test cost
        with torch.no_grad():
            diffs_test = [
                model.forward(w(v_test), U) - v_test
                for v_test, w, U in zip(v_tests, test_ws, test_Us)
            ]
            new_test_cost = sum(
                qcd_ml.util.linear_algebra.innerproduct(diff, diff).real
                for diff in diffs_test
            ) / len(test_config_paths)

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
