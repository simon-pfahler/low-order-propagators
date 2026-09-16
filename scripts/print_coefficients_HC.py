"""Print the formula obtained from restricted model weights.

Usage:
    print_coefficients_HC.py --layers=<n> --model_type=<name> --action=<name> --lattice_size=<str> --mass=<mass> --path=<str>

Options:
    --layers=<n>            Number of layers
    --model_type=<name>     Model type ("HC", "HL", "restricted")
    --action=<name>         Action name ("WilsonQuenched", "WilsonDynamic" or "Haar")
    --lattice_size=<str>    Lattice size string (e.g. "8c16")
    --mass=<mass>           Mass parameter value
    --path=<str>            Path to print coefficients of
"""

import ast
import json
import os
import sys

import docopt
import torch
from docopt import docopt
from utility import canonicalize_path, format_pdg, generator_mapping

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
layers = int(args["--layers"])
model_type = args["--model_type"]
action = args["--action"]
lattice_size_str = args["--lattice_size"]
mass = args["--mass"]
path = canonicalize_path(ast.literal_eval(args["--path"]))[0]

nrseeds = 5

model_name = f"{layers}layers_{action}_{lattice_size_str}_{model_type}"

# Load coefficients
coefficients = [
    [
        (k, v)
        for k, v in torch.load(
            f"data/coefficients/seeded_coefficients_{model_name}_m{mass}_seed{seed}.pt",
            weights_only=True,
        ).items()
        if canonicalize_path(k)[0] == path
    ]
    for seed in range(nrseeds)
]
can_coef = [[] for _ in range(nrseeds)]
for seed in range(nrseeds):
    for k, v in coefficients[seed]:
        _, new_indices, new_signs = canonicalize_path(k)
        new_generator_indices, new_generator_signs = generator_mapping(
            new_indices, new_signs
        )
        can_coef[seed].append(
            [
                new_generator_signs[i] * v[new_generator_indices[i]]
                for i in range(16)
            ]
        )

hopping_coefficients = [
    (k, v)
    for k, v in torch.load(
        f"data/coefficients/coefficients_4layers_hopping_m{mass}.pt",
        weights_only=True,
    ).items()
    if canonicalize_path(k)[0] == path
]
can_hcoef = []
for k, v in hopping_coefficients:
    _, new_indices, new_signs = canonicalize_path(k)
    new_generator_indices, new_generator_signs = generator_mapping(
        new_indices, new_signs
    )
    can_hcoef.append(
        [
            new_generator_signs[i] * v[new_generator_indices[i]]
            for i in range(16)
        ]
    )

can_coef_mean = torch.mean(torch.tensor(can_coef), dim=[0, 1])
can_coef_std_real = torch.std(torch.tensor(can_coef).real, dim=[0, 1])
can_coef_std_imag = torch.std(torch.tensor(can_coef).imag, dim=[0, 1])
can_hcoef_mean = torch.mean(torch.tensor(can_hcoef), dim=[0])
can_hcoef_std_real = (
    torch.std(torch.tensor(can_hcoef).real, dim=[0])
    if len(can_hcoef) > 1
    else torch.zeros(16)
)
can_hcoef_std_imag = (
    torch.std(torch.tensor(can_hcoef).imag, dim=[0])
    if len(can_hcoef) > 1
    else torch.zeros(16)
)

print(f"Hopping Expansion")
for i in range(16):
    cr, zr = format_pdg(
        can_hcoef_mean[i].real.item(), can_hcoef_std_real[i].item()
    )
    ci, zi = format_pdg(
        can_hcoef_mean[i].imag.item(), can_hcoef_std_imag[i].item()
    )
    if not (zr and zi):
        c = ""
        if not zr:
            c += cr
        if not zi:
            c += f", {ci}i"
        print(f"Coefficient of gamma structure {i}: {c}")

print(f"{model_type} Model")
for i in range(16):
    cr, zr = format_pdg(
        can_coef_mean[i].real.item(), can_coef_std_real[i].item()
    )
    ci, zi = format_pdg(
        can_coef_mean[i].imag.item(), can_coef_std_imag[i].item()
    )
    if not (zr and zi):
        c = ""
        if not zr:
            c += cr
        if not zi:
            c += f", {ci}i"
        print(f"Coefficient of gamma structure {i}: {c}")
