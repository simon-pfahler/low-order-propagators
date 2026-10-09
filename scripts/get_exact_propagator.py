"""Get the exact propagators for a mass, for later comparison with approximations.

Usage:
    get_exact_propagator.py --action=<name> --lattice_size=<str> --mass=<m>

Options:
    --action=<name>             Action name ("WilsonQuenched", "WilsonDynamic" or "Haar")
    --lattice_size=<str>        Lattice size string (e.g. "8c16")
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
action = args["--action"]
lattice_size_str = args["--lattice_size"]
mass = float(args["--mass"])

lattice_size = (
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[0]),
    int(lattice_size_str.split("c")[1]),
)

test_config_paths = get_config_paths(action, lattice_size_str, test=True)

test_Us = [torch.load(p, weights_only=True) for p in test_config_paths]
test_ws = [qcd_ml.qcd.dirac.dirac_wilson(U, mass) for U in test_Us]

# Ensure output directory exists
os.makedirs("data/propagators", exist_ok=True)

for idx, test_w in enumerate(test_ws):

    def op(x):
        x = test_w(x)
        test_w.dag = True
        x = test_w(x)
        test_w.dag = False
        return x

    for spin_idx in range(4):
        for color_idx in range(3):
            output_path = f"data/propagators/exact_propagator_{action}_{lattice_size_str}_U{idx}_s{spin_idx}_c{color_idx}_m{mass:.2f}.pt"

            if os.path.exists(output_path):
                continue

            source = torch.zeros(*lattice_size, 4, 3, dtype=torch.cdouble)
            source[0, 0, 0, 0, spin_idx, color_idx] = 1

            test_w.dag = True
            source = test_w(source)
            test_w.dag = False

            res = qcd_ml.util.solver.GMRES(
                op,
                source.clone(),
                torch.zeros_like(source),
                eps=1e-12,
                maxiter=10000,
                inner_iter=400,
            )[0]

            torch.save(res, output_path)
