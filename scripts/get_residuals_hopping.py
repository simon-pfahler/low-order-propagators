"""Calculate residuals for the hopping expansion.

Usage:
    get_residuals_hopping.py --nsteps=<n> --volume=<vol> --mass=<m>

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
    f"Calculating residuals for hopping expansion with {nsteps} steps,"
    f"mass={mass:.2f}, volume {args['--volume']}"
)

# Handle m=-4 case:
if mass == -4:
    # Ensure output directory exists
    os.makedirs("data/residuals", exist_ok=True)

    residuals = torch.nan * torch.ones(4, 100)

    # Save output
    torch.save(
        residuals,
        f"data/residuals/residuals_{nsteps}steps_{vol}_hopping_m{mass:.2f}.pt",
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


def residual(w, winv_approx, x):
    return qcd_ml.util.linear_algebra.norm(w(winv_approx(x)) - x)


residuals = torch.zeros(len(test_ws), 100)
for sample_idx in range(100):
    v = torch.randn(*lattice_size, 4, 3, dtype=torch.cdouble)
    v /= qcd_ml.util.linear_algebra.norm(v)
    for test_idx, w in enumerate(test_ws):
        residuals[test_idx, sample_idx] = (
            residual(w, lambda x: hopping_expansion(x, w), v.clone()) ** 2
        )

# Ensure output directory exists
os.makedirs("data/residuals", exist_ok=True)

# Save output
torch.save(
    residuals,
    f"data/residuals/residuals_{nsteps}steps_{vol}_hopping_m{mass:.2f}.pt",
)
