"""Perform test solves using the hopping expansion as perconditioner.

Usage:
    get_iteration_count.py --nsteps=<n> --volume=<vol> --mass=<m>

Options:
    --nsteps=<n>            Number of steps in the hopping expansion
    --volume=<vol>          Volume string
    --mass=<mass>           Mass parameter value
"""

import json
import os
import sys

import qcd_ml
import torch
from docopt import docopt
from utility import get_config_path

sys.path.insert(0, "scripts")

torch.set_grad_enabled(False)

# Parse docopt arguments
args = docopt(__doc__)
nsteps = int(args["--nsteps"])
vol = args["--volume"]
Lx, Lt = map(int, args["--volume"].split("c"))
lattice_size = [Lx, Lx, Lx, Lt]
if vol == "8c16":
    test_configs = [1000, 2000, 2347, 786]
elif vol == "16c32":
    test_configs = [282, 472, 329, 591]
else:
    raise ValueError(f"Volume {vol} not supported!")
mass = float(args["--mass"])

print(
    f"Calculating iteration_counts for hopping expansion with {nsteps} steps,"
    f"mass={mass:.2f}, volume {args['--volume']}"
)

# Handle m=-4 case:
if mass == -4:
    # Ensure output directory exists
    os.makedirs("data/iteration_counts", exist_ok=True)

    iteration_counts = torch.nan * torch.ones(4, 100)

    # Save output
    torch.save(
        iteration_counts,
        f"data/iteration_counts/iteration_counts_{nsteps}steps_{vol}_hopping_m{mass:.2f}.pt",
    )

test_Us = [
    torch.load(get_config_path(lattice_size, c), weights_only=True)
    for c in test_configs
]
test_ws = [qcd_ml.qcd.dirac.dirac_wilson(U, mass) for U in test_Us]


# Define Hopping expansion
def hopping_expansion(v, w):
    kappa = 1 / (w.mass_parameter + 4)

    def H(x):
        res = torch.zeros_like(x)
        for mu in range(4):
            res -= w.apply_pos_hop(x, mu)
            res -= w.apply_neg_hop(x, mu)
        return res

    term = kappa * v
    res = term.clone()
    for _ in range(nsteps):
        term = kappa * H(term)
        res += term
    return res


# Define Hopping expansion
def hopping_expansion(v, w):
    kappa = 1 / (w.mass_parameter + 4)

    def H(x):
        res = torch.zeros_like(x)
        for mu in range(4):
            res -= w.apply_pos_hop(x, mu)
            res -= w.apply_neg_hop(x, mu)
        return res

    term = kappa * v
    res = term.clone()
    for _ in range(nsteps):
        term = kappa * H(term)
        res += term
    return res


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
                preconditioner=lambda x: hopping_expansion(x, w),
            )[1]["history"]
        )

# Ensure output directory exists
os.makedirs("data/iteration_counts", exist_ok=True)

# Save output
torch.save(
    iteration_counts,
    f"data/iteration_counts/iteration_counts_{nsteps}steps_{vol}_hopping_m{mass:.2f}.pt",
)
