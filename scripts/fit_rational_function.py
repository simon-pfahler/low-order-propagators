"""Fit a rational function to the mass dependence.

Usage:
    figure_coefficient_mass_dependence_appendix.py --path=<str> --gamma_index=<n> --layers=<n> --powers_num=<n> --powers_denom=<n> [--plot]

Options:
    --path=<str>            Path to plot coefficients of
    --gamma_index=<n>       Gamma structure index to plot coefficients of
    --layers=<n>            Number of layers of the model
    --powers_num=<n>        Powers of (m+4) to use in numerator
    --powers_denom=<n>      Powers of (m+4) to use in denominator
    --plot                  Plot the resulting fit
"""

import ast
import os
import re
import sys
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from docopt import docopt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.optimize import least_squares, leastsq
from utility import (
    canonicalize_path,
    consolidate_path,
    format_pdg,
    generator_mapping,
    generators,
    get_path_length,
    model_paths,
)


def extract_mass(filename):
    try:
        mass = float(re.search(r"_m([+-]?\d+\.?\d*)\.pt", filename).group(1))
    except:
        raise ValueError(f"Mass not extractable from filename '{filename}'!")
    return mass


def get_reference_curve(order, prefactor):
    """Generate reference curve for hopping expansion."""
    mass_range = torch.linspace(-3.98, 2.5, 1000)
    line_values = prefactor / ((2**order) * (mass_range + 4) ** (order + 1))
    return mass_range, line_values


plt.style.use("./scripts/iclr2027.mplstyle")

sys.path.insert(0, "scripts")

# Parse docopt arguments
args = docopt(__doc__)
model_type = "HC"
action = "WilsonQuenched"
lattice_size_str = "8c16"
layers = int(args["--layers"])
path = ast.literal_eval(args["--path"])
path_length = get_path_length(path)
if get_path_length(path) > 3:
    raise ValueError(
        f"Only coefficients for paths up to length 3 are available!"
    )
if path != canonicalize_path(path)[0]:
    raise ValueError(
        f"Please pass a canonical path! This path's canonical path is {canonicalize_path(path)[0]}"
    )
gamma_index = int(args["--gamma_index"])
powers_num = [int(e) for e in args["--powers_num"].split(",")]
powers_denom = [int(e) for e in args["--powers_denom"].split(",")]

masses = torch.from_numpy(np.arange(-5, 2.2, 0.2))
for idx, mass in enumerate(masses):
    if -1e-3 < mass < 0:
        masses[idx] = 0.0

coef_means = torch.zeros(len(masses), dtype=torch.double)
coef_stds = torch.zeros(len(masses), dtype=torch.double)

val_mass = -9.0


def get_coefs(mass):
    filenames = [
        f"data/coefficients/coefficients_{layers}layers_WilsonQuenched_8c16_HC_m{mass:.2f}.pt",
        *[
            f"data/coefficients/seeded_coefficients_{layers}layers_WilsonQuenched_8c16_HC_m{mass:.2f}_seed{seed}.pt"
            for seed in range(5)
        ],
    ]
    coefs = list()
    for filename in filenames:
        if not os.path.exists(filename):
            continue
        coefficients = {
            k: v
            for k, v in torch.load(
                filename,
                weights_only=True,
            ).items()
            if get_path_length(k) == path_length
        }

        for curr_path, curr_coef in coefficients.items():
            can_curr_path, new_order, new_signs = canonicalize_path(curr_path)
            if can_curr_path == path:
                new_generator_indices, new_generator_signs = generator_mapping(
                    new_order, new_signs
                )
                can_curr_coef = torch.stack(
                    [
                        new_generator_signs[gamma_index]
                        * curr_coef[new_generator_indices[gamma_index]]
                        for gamma_index in range(16)
                    ]
                )
                coefs.append(can_curr_coef[gamma_index].real)

    return torch.tensor(coefs)


for mass_idx, mass in enumerate(masses):
    coefs = get_coefs(mass)
    coef_means[mass_idx] = torch.mean(coefs)
    coef_stds[mass_idx] = torch.std(coefs)
    if len(coefs) == 1:
        coef_stds[mass_idx] = 0.1 * coef_means[mass_idx].abs()

coefs_val = get_coefs(val_mass)
coef_val_mean = torch.mean(coefs_val)
coef_val_std = torch.std(coefs_val)

# Fit a rational function
nr_fit_params = len(powers_num) + len(powers_denom)


def linearized_fit(params, masses, coefs):
    res = torch.zeros_like(coefs)
    res -= coefs
    idx = 0
    for n in powers_num:
        res += params[idx] * (masses + 4) ** n
        idx += 1
    for m in powers_denom:
        res += params[idx] * (masses + 4) ** m * coefs
        idx += 1
    return res


def rational_function(mass, *params):
    num = 0
    idx = 0
    for n in powers_num:
        num += params[idx] * (mass + 4) ** n
        idx += 1
    denom = 1
    for m in powers_denom:
        denom += params[idx] * (mass + 4) ** m
        idx += 1
    return num / denom


best_fit_params = torch.zeros(nr_fit_params)
best_fit_param_errors = torch.zeros(nr_fit_params)
best_chisq = np.inf
best_converged = False
best_optimizer = None
nr_tries = 0
while not best_converged:
    best_fit_params = torch.zeros(nr_fit_params)
    best_chisq = np.inf
    best_converged = False
    for fit_run in range(20):
        # Fit parameters A_0, A_1, ..., A_{order_num}, B_1, ..., B_{order_denom}
        fit_params = torch.zeros(nr_fit_params)

        if fit_run != 0:
            # First, a least squares fit as a starting point
            samples_subset = np.random.choice(
                list(range(len(masses))), size=nr_fit_params, replace=False
            )
            masses_subset = masses[samples_subset]
            coef_means_subset = coef_means[samples_subset]
            fit_params, _ = leastsq(
                linearized_fit,
                fit_params,
                args=(masses_subset, coef_means_subset),
            )

        # And then a nonlinear fit
        least_squares = LeastSquares(
            masses, coef_means, coef_stds, rational_function
        )

        optimizer = Minuit(least_squares, *fit_params)
        optimizer.migrad(use_simplex=True, iterate=20)

        chisq = optimizer.fmin.reduced_chi2
        if chisq < best_chisq:
            best_chisq = chisq
            best_fit_params = np.array([*optimizer.values])
            best_fit_param_errors = np.array([*optimizer.errors])
            best_converged = optimizer.valid
            best_optimizer = optimizer

    if not best_converged:
        print(f"No fit converged, retrying")
    nr_tries += 1
    if nr_tries == 10:
        break

# Plot the fit on top of the values
masses_fit = torch.linspace(-10, 2, 100)
coefs_fit = torch.stack(
    [rational_function(mass, *best_fit_params) for mass in masses_fit]
)

print(f"Fit function parameters:")
print(
    f"Numerator: {", ".join([format_pdg(p,e)[0] for p, e in zip(best_fit_params[:len(powers_num)], best_fit_param_errors[:len(powers_num)])])}"
)
print(
    f"Denominator: {", ".join([format_pdg(p,e)[0] for p, e in zip(best_fit_params[len(powers_num):], best_fit_param_errors[len(powers_num):])])}"
)
print(f"Chisq/DOF:\n{best_chisq}")

if not args["--plot"]:
    quit()

# Bootstrapped fit error band
rng = np.random.default_rng(1)
par_b = rng.multivariate_normal(
    best_optimizer.values, best_optimizer.covariance, size=1000
)
y_b = np.array(
    [[rational_function(mass, *p) for p in par_b] for mass in masses_fit]
)
yerr_boot = np.std(y_b, axis=1)

plt.figure(figsize=(4.5, 3.5))

plt.errorbar(
    masses,
    coef_means,
    yerr=coef_stds,
    linestyle="none",
    capsize=5,
    marker="o",
    markerfacecolor="none",
    label="Data",
)
plt.errorbar(
    val_mass,
    coef_val_mean,
    coef_val_std,
    linestyle="none",
    capsize=5,
    marker="o",
    c="C2",
    markerfacecolor="none",
    label="Validation",
)
plt.plot(
    masses_fit,
    coefs_fit,
    label=f"Fit ({powers_num}, {powers_denom})",
    color="C1",
)
plt.fill_between(
    masses_fit,
    coefs_fit - yerr_boot,
    coefs_fit + yerr_boot,
    facecolor="C1",
    alpha=0.5,
)

plt.title(
    f"Coefficient evolution for {layers}-layer HC models\n"
    f"path {path}, gamma index {gamma_index}"
)
plt.xlabel("Mass")
plt.ylabel("Real part of coefficient")
plt.legend()
plt.tight_layout()
plt.show()
