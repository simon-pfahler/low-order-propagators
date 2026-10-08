"""Print a LaTeX (booktabs) table of restricted model weights.

The table contains the overall factor w and the per-layer weights
w_I and w_gamma for all restricted models trained on the WilsonQuenched
action and the 8c16 lattice at the given mass.

Usage:
    print_latex_table_restricted.py --mass=<mass>

Options:
    --mass=<mass>           Mass parameter value (e.g. "0.00" or "-1.00")
"""

import glob
import os
import re
import sys

import torch
from docopt import docopt
from utility import format_pdg

# Parse docopt arguments
args = docopt(__doc__)
mass = float(args["--mass"])
mass_str = f"{mass:.2f}"

model_type = "restricted"
action = "WilsonQuenched"
lattice_size_str = "8c16"

# Exact values of the overall factor (kappa) and the layer weights (kappa / 2)
kappa = 1 / (mass + 4)


def format_exact(value):
    """Format an exact value without redundant trailing zeros."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def fold_exponent(s):
    """Fold the exponent of a PDG-formatted number into its mantissa,
    e.g. 2.31536(19)e-1 -> 0.231536(19)."""
    if "e" not in s:
        return s
    num, exp = s.split("e")
    exp = int(exp)
    match = re.fullmatch(r"(-?)(\d+)(?:\.(\d+))?(\(\d+\))?", num)
    sign, int_part, frac_part, unc = (
        match.group(1),
        match.group(2),
        match.group(3) or "",
        match.group(4) or "",
    )
    digits = int_part + frac_part
    point = len(int_part) + exp
    if point <= 0:
        num = "0." + "0" * (-point) + digits
    elif point >= len(digits):
        num = digits + "0" * (point - len(digits))
    else:
        num = digits[:point] + "." + digits[point:]
    return sign + num + unc


def format_pdg_plain(mean, std):
    """Format (mean, std) in PDG notation without an exponential."""
    s, zero = format_pdg(mean, std)
    return fold_exponent(s), zero


def format_complex(mean, std_real, std_imag):
    """Format a complex mean with real and imaginary uncertainties,
    omitting an imaginary part consistent with zero."""
    mr = format_pdg_plain(mean.real.item(), std_real.item())[0]
    mi, z = format_pdg_plain(mean.imag.item(), std_imag.item())
    if z:
        return mr
    return f"{mr}{mi}i" if mi.startswith("-") else f"{mr}+{mi}i"


def find_runs():
    """Find the available (layers, seeds) runs for the given mass."""
    pattern = re.compile(
        rf"seeded_weights_(\d+)layers_"
        rf"{re.escape(f'{action}_{lattice_size_str}_{model_type}')}"
        rf"_m{re.escape(mass_str)}_seed(\d+)\.pt"
    )
    runs = {}
    for path in glob.glob(f"data/weights/seeded_weights_*_m{mass_str}_seed*.pt"):
        match = pattern.match(os.path.basename(path))
        if match:
            runs.setdefault(int(match.group(1)), []).append(
                int(match.group(2))
            )
    for seeds in runs.values():
        seeds.sort()
    return {layers: runs[layers] for layers in sorted(runs)}


runs = find_runs()
if not runs:
    sys.exit(f"No restricted WilsonQuenched 8c16 models found for m={mass_str}.")

# Build the table body: one block per layer count, one row per layer index.
rows = []
for layers, seeds in runs.items():
    weightss = [
        torch.load(
            f"data/weights/seeded_weights_{layers}layers_"
            f"{action}_{lattice_size_str}_{model_type}"
            f"_m{mass_str}_seed{seed}.pt",
            weights_only=True,
        )
        for seed in seeds
    ]

    overall_factors = torch.stack([w["overall_factor"] for w in weightss])
    layer_weights = torch.stack([w["weights"] for w in weightss])

    overall_factor_mean = torch.mean(overall_factors)
    overall_factor_std_real = torch.std(overall_factors.real)
    overall_factor_std_imag = torch.std(overall_factors.imag)
    layer_weights_mean = torch.mean(layer_weights, dim=0)
    layer_weights_std_real = torch.std(layer_weights.real, dim=0)
    layer_weights_std_imag = torch.std(layer_weights.imag, dim=0)

    for ell in range(layers):
        rows.append(
            [
                str(layers) if ell == 0 else "",
                str(ell + 1),
                format_complex(
                    overall_factor_mean,
                    overall_factor_std_real,
                    overall_factor_std_imag,
                )
                if ell == 0
                else "",
                format_complex(
                    layer_weights_mean[ell, 0],
                    layer_weights_std_real[ell, 0],
                    layer_weights_std_imag[ell, 0],
                ),
                format_complex(
                    layer_weights_mean[ell, 1],
                    layer_weights_std_real[ell, 1],
                    layer_weights_std_imag[ell, 1],
                ),
            ]
        )

# Assemble the LaTeX table with padded columns.
header = [
    "$n$",
    "$\\ell$",
    f"\\multicolumn{{1}}{{c}}{{$w$ ({format_exact(kappa)})}}",
    f"\\multicolumn{{1}}{{c}}"
    f"{{$w_\\mI^{{(\\ell)}}$ ({format_exact(kappa / 2)})}}",
    f"\\multicolumn{{1}}{{c}}"
    f"{{$w_\\gamma^{{(\\ell)}}$ ({format_exact(kappa / 2)})}}",
]

widths = [max(len(row[i]) for row in rows + [header]) for i in range(5)]

lines = ["\\toprule", "\t\t" + " & ".join(header) + " \\\\", "\t\t\\midrule"]
for i, row in enumerate(rows):
    if i > 0 and row[0]:
        lines.append("\t\t\\midrule")
    lines.append(
        "\t\t"
        + " & ".join(cell.ljust(width) for cell, width in zip(row, widths))
        + " \\\\"
    )
lines.append("\\bottomrule")

print("\n".join(lines))
