"""Plot chi2/dof of rational function fits against the number of fit parameters.

With --extrapolation, also plots the extrapolation error at a validation mass
for the best fits.

Usage:
    plot_chi2_vs_nr_params.py --layers=<n> --path=<str> --gamma_index=<n> [--extrapolation]

Options:
    --layers=<n>            Number of layers in the HC models
    --path=<str>            Path to consider
    --gamma_index=<n>       Gamma structure index to consider
    --extrapolation         Also plot the extrapolation error of the best fits
"""

import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

plt.style.use("scripts/iclr2027.mplstyle")

args = docopt(__doc__)
layers = int(args["--layers"])
path = args["--path"]
gamma_index = int(args["--gamma_index"])
extrapolation = args["--extrapolation"]

val_mass = -9.0  # Validation mass extrapolated to, cf. fit_rational_function.py
nr_masses = 36  # Number of masses fitted, cf. fit_rational_function.py

with open(
    f"data/chi2/chi2s_{layers}layers_path{path}_g{gamma_index}.json"
) as f:
    chi2dict = json.load(f)

# Parse the powers of (mass+4) and the chi2/dof for each fit
fits = []
for key, chi2 in chi2dict.items():
    powers_num_str, powers_denom_str = key.split("/")
    powers_num = [int(e) for e in powers_num_str.split(",") if e != ""]
    powers_denom = [int(e) for e in powers_denom_str.split(",") if e != ""]
    fits.append((powers_num, powers_denom, chi2, key))

# Find the optimal fit for each number of parameters
best_fits = dict()
for powers_num, powers_denom, chi2, key in fits:
    nr_params = len(powers_num) + len(powers_denom)
    if chi2 < best_fits.get(nr_params, (None, None, float("inf"), None))[2]:
        best_fits[nr_params] = (powers_num, powers_denom, chi2, key)

best_keys = {(tuple(e[0]), tuple(e[1])) for e in best_fits.values()}
other_fits = [
    fit for fit in fits if (tuple(fit[0]), tuple(fit[1])) not in best_keys
]


def get_extrap_error(powers_num, powers_denom):
    powers_num_str = ",".join(map(str, powers_num))
    powers_denom_str = ",".join(map(str, powers_denom))
    key = powers_num_str + "/" + powers_denom_str
    print(f"Computing extrapolation error for {powers_num}/{powers_denom}")
    stdout = subprocess.run(
        [
            "python",
            "scripts/fit_rational_function.py",
            f"--path={path}",
            f"--gamma_index={gamma_index}",
            f"--layers={layers}",
            f"--powers_num={powers_num_str}",
            f"--powers_denom={powers_denom_str}",
        ],
        capture_output=True,
        text=True,
    ).stdout
    return float(stdout.split("Extrapolation error at mass")[1].split("\n")[1])


# Extrapolation errors of the best fits
best_extraps = dict()
if extrapolation:
    for nr_params in sorted(best_fits):
        try:
            best_extraps[nr_params] = get_extrap_error(
                *best_fits[nr_params][:2]
            )
        except:
            print(
                f"Extrapolation error for {nr_params} params could not be computed"
            )

# Create plot
if extrapolation:
    fig, (ax_chi2, ax_extrap) = plt.subplots(
        2, 1, figsize=(4.5, 6), sharex=True
    )
else:
    fig, ax_chi2 = plt.subplots(figsize=(4.5, 3))
    ax_chi2.set_xlabel("Number of parameters")

ax_chi2.scatter(
    [len(e[0]) + len(e[1]) for e in other_fits],
    [e[2] for e in other_fits],
    marker="o",
    facecolor="none",
    edgecolor="#0077bb",
    label="Rational function fits",
)
ax_chi2.scatter(
    sorted(best_fits),
    [best_fits[nr_params][2] for nr_params in sorted(best_fits)],
    marker="*",
    color="#ee3377",
    label="Best fit",
)

ax_chi2.set_ylabel("Chi2/DOF")
ax_chi2.set_yscale("log")
ax_chi2.legend(loc="upper right")
ax_chi2.grid(True, which="both", linestyle="--", alpha=0.5)

if extrapolation:
    ax_extrap.scatter(
        sorted(best_extraps),
        [best_extraps[nr_params] for nr_params in sorted(best_extraps)],
        marker="*",
        color="#ee3377",
        label="Best fit",
    )

    ax_extrap.set_xlabel("Number of parameters")
    ax_extrap.set_ylabel(f"Extrapolation error at m = {val_mass}")
    ax_extrap.set_yscale("log")
    ax_extrap.legend(loc="upper right")
    ax_extrap.grid(True, which="both", linestyle="--", alpha=0.5)

ax_chi2.set_title(
    "Chi2/DOF of rational function fits vs number of parameters\n"
    f"{layers}-layer HC models, path {path}, gamma index {gamma_index}\n"
    f"{len(chi2dict)} fits"
)
fig.tight_layout()
plt.show()

# BIC = chi2 + nr_params * ln(nr_masses), where the cached reduced
# chi2 is chi2 / (nr_masses - nr_params)
bics = {
    nr_params: best_fits[nr_params][2] * (nr_masses - nr_params)
    + nr_params * np.log(nr_masses)
    for nr_params in best_fits
}
best_bic_params = min(bics, key=bics.get)

for nr_params in sorted(best_fits):
    powers_num, powers_denom, chi2, key = best_fits[nr_params]
    marker = " <- BIC optimal" if nr_params == best_bic_params else ""
    if extrapolation:
        extrap_error = best_extraps.get(nr_params, "n/a")
        print(
            f"{nr_params} params: {powers_num}/{powers_denom} -> chi2/DOF {chi2}, "
            f"BIC {bics[nr_params]:.1f}, extrapolation error {extrap_error}{marker}"
        )
    else:
        print(
            f"{nr_params} params: {powers_num}/{powers_denom} -> chi2/DOF {chi2}, "
            f"BIC {bics[nr_params]:.1f}{marker}"
        )
