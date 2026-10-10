"""Calculate propagator errors of a trained model or a baseline.

Usage:
    get_propagator_error.py --layers=<n> --type=<name> --action=<name> --lattice_size=<str> [--model_action=<name>] [--model_lattice_size=<str>] --mass=<m>

Options:
    --layers=<n>                Number of layers of the model, or number of steps of the hopping expansion/GMRES
    --type=<name>               Model type ("HC", "HL", "restricted"), "hopping" or "GMRES"
    --action=<name>             Action name ("WilsonQuenched", "WilsonDynamic" or "Haar")
    --lattice_size=<str>        Lattice size string (e.g. "8c16")
    --model_action=<name>       Action the model was trained on
    --model_lattice_size=<str>  Lattice size the model was trained on
    --mass=<mass>               Mass parameter value
"""

import os
import sys

import qcd_ml
import torch
from docopt import docopt
from model import *
from utility import get_config_paths, model_paths

sys.path.insert(0, "scripts")

torch.set_grad_enabled(False)

# Parse docopt arguments
args = docopt(__doc__)
layers = int(args["--layers"])
model_type = args["--type"]
action = args["--action"]
lattice_size_str = args["--lattice_size"]
model_action = args["--model_action"]
model_lattice_size_str = args["--model_lattice_size"]
if model_type not in ["restricted", "HL", "HC", "hopping", "GMRES"]:
    raise ValueError(f"Type '{model_type}' not supported!")
mass = float(args["--mass"])

model_name = ""
if model_type in ["hopping", "GMRES"]:
    model_name = f"{layers}layers_{model_type}"
else:
    model_name = (
        f"{layers}layers_{model_action}_{model_lattice_size_str}_{model_type}"
    )
lattice_size = (
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[1]),
)

test_config_paths = get_config_paths(action, lattice_size_str, test=True)

test_Us = [torch.load(p, weights_only=True) for p in test_config_paths]
test_ws = [qcd_ml.qcd.dirac.dirac_wilson(U, mass) for U in test_Us]

# Handle hopping expansion at m=-4:
if model_type == "hopping" and mass == -4:
    timesliced_errors = torch.nan * torch.ones(
        len(test_ws), 4, 3, lattice_size[-1], dtype=torch.double
    )
    os.makedirs("data/propagators", exist_ok=True)
    outpath = f"data/propagators/timesliced_errors_{model_name}_{action}_{lattice_size_str}_m{mass:.2f}.pt"
    torch.save(timesliced_errors, outpath)
    quit()

# Define model
match model_type:
    case "restricted":
        model = Model_restricted(layers, model_paths)
        weights_path = f"data/weights/weights_{model_name}_m{mass:.2f}.pt"
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        func = lambda x, w: model(x, w.U)
    case "HL":
        model = Model_HL(layers, model_paths)
        weights_path = f"data/weights/weights_{model_name}_m{mass:.2f}.pt"
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        func = lambda x, w: model(x, w.U)
    case "GMRES":
        func = lambda x, w: qcd_ml.util.solver.GMRES(
            w, x.clone(), torch.zeros_like(x), eps=1e-16, maxiter=layers
        )[0]
    case "hopping":

        def hopping(v, w):
            kappa = 1 / (w.mass_parameter + 4)

            def H(x):
                res = torch.zeros_like(x)
                for mu in range(4):
                    res -= w.apply_pos_hop(x, mu)
                    res -= w.apply_neg_hop(x, mu)
                return res

            term = kappa * v
            res = term.clone()
            for _ in range(layers):
                term = kappa * H(term)
                res += term
            return res

        func = lambda x, w: hopping(x, w)
    case _:
        model = Model_HC(layers, model_paths)
        weights_path = f"data/weights/weights_{model_name}_m{mass:.2f}.pt"
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        func = lambda x, w: model(x, w.U)


# Ensure output directory exists
os.makedirs("data/propagators", exist_ok=True)

timesliced_errors = torch.zeros(
    len(test_ws), 4, 3, lattice_size[-1], dtype=torch.double
)

for idx, test_w in enumerate(test_ws):

    def op(x):
        x = test_w(x)
        test_w.dag = True
        x = test_w(x)
        test_w.dag = False
        return x

    for spin_idx in range(4):
        for color_idx in range(3):
            exact_propagator = torch.load(
                f"data/propagators/exact_propagator_{action}_{lattice_size_str}_U{idx}_s{spin_idx}_c{color_idx}_m{mass:.2f}.pt",
                weights_only=True,
            )
            source = torch.zeros(*lattice_size, 4, 3, dtype=torch.cdouble)
            source[0, 0, 0, 0, spin_idx, color_idx] = 1

            res = func(source, test_w)

            diff = res - exact_propagator

            timesliced_errors[idx, spin_idx, color_idx] = (
                torch.einsum("xyztsc,xyztsc->t", diff.conj(), diff).real ** 0.5
            )

outpath = f"data/propagators/timesliced_errors_{model_name}_{action}_{lattice_size_str}_m{mass:.2f}.pt"
torch.save(timesliced_errors, outpath)
