"""Calculate chi2/dof for rational function fits with different numbers of parameters

Usage:
    get_chi2_vs_nr_params.py --layers=<n> --path=<str> --gamma_index=<n> --max_nr_params=<n> [--nr_steps=<n>] [--initialization=<str>]

Options:
    --layers=<n>            Number of layers in the HC models
    --path=<str>            Path to consider
    --gamma_index=<n>       Gamma structure index to consider
    --max_nr_params=<n>     Maximum number of parameters in the fit
    --nr_steps=<n>          Number of steps to run [default: 100]
    --initialization=<str>  Semicolon-separated initial guesses, e.g.
                            "0/;1/2;3,4/8;0,2/2,4;0,2,4/2,4". Each guess has the
                            form num/denom with comma-separated powers of (m+4).
                            A guess replaces the default initial guess (the
                            polynomial 0,...,i) for its number of parameters;
                            parameter counts without a guess keep the default.
"""

import ast
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from fit_rational_function import fit_rational_function, get_coef_data
from matplotlib.colors import LogNorm

args = docopt(__doc__)
layers = int(args["--layers"])
path = args["--path"]
gamma_index = int(args["--gamma_index"])
max_nr_params = int(args["--max_nr_params"])
nr_steps = int(args["--nr_steps"])

masses, coef_means, coef_stds, _, _ = get_coef_data(
    layers, ast.literal_eval(path), gamma_index
)

rng = np.random.default_rng(42)

chi2dict = dict()
try:
    with open(
        f"data/chi2/chi2s_{layers}layers_path{path}_g{gamma_index}.json"
    ) as f:
        chi2dict = json.load(f)
except:
    pass


def get_fit(powers_num, powers_denom):
    """Fit a rational function with the given powers, returning the best fit
    parameters, their errors and the reduced chi2."""
    best_fit_params, best_fit_param_errors, chi2, _ = fit_rational_function(
        powers_num, powers_denom, masses, coef_means, coef_stds
    )
    return best_fit_params, best_fit_param_errors, chi2


def get_chi2(powers_num, powers_denom):
    powers_num_str = ",".join(map(str, powers_num))
    powers_denom_str = ",".join(map(str, powers_denom))
    key = powers_num_str + "/" + powers_denom_str
    if key in chi2dict.keys():
        return chi2dict[key]
    try:
        best_fit_params, best_fit_param_errors, chi2 = get_fit(
            powers_num, powers_denom
        )
    except:
        raise ValueError(f"Fit failed for {powers_num}/{powers_denom}")
    chi2dict[key] = float(chi2)
    # If a parameter agrees with 0 within statistics, also fit without
    # that parameter and update the best fits if the pruned fit is better
    for i in range(len(powers_num) + len(powers_denom)):
        if abs(best_fit_params[i]) < best_fit_param_errors[i]:
            new_powers_num = list(powers_num)
            new_powers_denom = list(powers_denom)
            if i < len(powers_num):
                new_powers_num.pop(i)
            else:
                new_powers_denom.pop(i - len(powers_num))
            if not new_powers_num:
                # The numerator needs to contain at least one power
                continue
            new_nr_params = len(new_powers_num) + len(new_powers_denom)
            new_chi2 = get_chi2(new_powers_num, new_powers_denom)
            if new_chi2 < best_chi2s[new_nr_params - 1]:
                best_chi2s[new_nr_params - 1] = new_chi2
                best_paramss[new_nr_params - 1] = (
                    new_powers_num,
                    new_powers_denom,
                )
    return chi2dict[key]


def print_status(best_paramss, best_chi2s):
    lines = [
        f"\t{i+1} params: {','.join(map(str,best_paramss[i][0]))}/{','.join(map(str,best_paramss[i][1]))}"
        for i in range(max_nr_params)
    ]
    width = max(map(len, lines))
    for line, chi2 in zip(lines, best_chi2s):
        print(f"{line:<{width}} -> {chi2}")


def mutate(params):
    h = rng.uniform()
    powers_num, powers_denom = list(params[0]), list(params[1])
    if h < 0.2:
        # Add a power previously not there
        if len(powers_num) + len(powers_denom) >= max_nr_params:
            return params
        lists = [powers_num, powers_denom]
        idx = rng.integers(2)
        candidates = [
            p
            for p in (
                range(0, max(powers_num + powers_denom) + 5)
                if idx == 0
                else range(1, max(powers_num + powers_denom) + 5)
            )
            if p not in lists[idx]
        ]
        if not candidates:
            return params
        lists[idx].append(int(rng.choice(candidates)))
        return (sorted(lists[0]), sorted(lists[1]))
    if h < 0.4:
        # Remove a random power
        if len(powers_num) == 1:
            # Keep at least one power in the numerator
            candidates = [(powers_denom, p) for p in powers_denom]
        else:
            candidates = [(powers_num, p) for p in powers_num] + [
                (powers_denom, p) for p in powers_denom
            ]
        if not candidates:
            return params
        powers_list, power = candidates[int(rng.integers(len(candidates)))]
        powers_list.remove(power)
        return (sorted(powers_num), sorted(powers_denom))
    if h < 0.6:
        # Replace a random power by a different power not in that list
        candidates = powers_num + powers_denom
        if not candidates:
            return params
        power = int(rng.choice(candidates))
        lists = [powers_num, powers_denom]
        idx = 0 if power in powers_num else 1
        replacements = [
            p
            for p in (
                range(0, max_nr_params + 1)
                if idx == 0
                else range(1, max_nr_params + 1)
            )
            if p not in lists[idx]
        ]
        if not replacements:
            return params
        lists[idx][lists[idx].index(power)] = int(rng.choice(replacements))
        return (sorted(lists[0]), sorted(lists[1]))
    # Shuffle the powers between both lists, but keep powers that are in both lists as they are.
    shared = [p for p in powers_num if p in powers_denom]
    unique = [p for p in powers_num + powers_denom if p not in shared]
    new_num, new_denom = list(shared), list(shared)
    for p in unique:
        (new_num if p == 0 or rng.uniform() < 0.5 else new_denom).append(p)
    if len(new_num) == 0:
        return params
    return (sorted(new_num), sorted(new_denom))


def parse_initialization(initialization):
    """Parse an initialization string like "0/;1/2;3,4/8" into a list of
    (powers_num, powers_denom) initial guesses."""
    guesses = list()
    for part in initialization.split(";"):
        part = part.strip()
        if not part:
            continue
        num_str, _, denom_str = part.partition("/")
        powers_num = sorted(
            int(e) for e in num_str.split(",") if e.strip() != ""
        )
        powers_denom = sorted(
            int(e) for e in denom_str.split(",") if e.strip() != ""
        )
        if not powers_num:
            raise ValueError(
                f"Invalid initialization part '{part}':"
                " the numerator needs at least one power!"
            )
        nr_params = len(powers_num) + len(powers_denom)
        if nr_params > max_nr_params:
            raise ValueError(
                f"Invalid initialization part '{part}':"
                f" more than {max_nr_params} parameters!"
            )
        guesses.append((powers_num, powers_denom))
    if not guesses:
        raise ValueError("The initialization contains no initial guesses!")
    return guesses


best_paramss = [(list(range(i + 1)), []) for i in range(max_nr_params)]
best_chi2s = [np.inf] * max_nr_params
if args["--initialization"]:
    # Slot the provided guesses by their number of parameters (the last
    # guess for a given number of parameters wins); parameter counts
    # without a guess keep the default polynomial guess.
    slots = {
        len(powers_num) + len(powers_denom): (powers_num, powers_denom)
        for powers_num, powers_denom in parse_initialization(
            args["--initialization"]
        )
    }
    for i in range(max_nr_params):
        if i + 1 in slots:
            best_paramss[i] = slots[i + 1]
for i, params in enumerate(best_paramss):
    chi2 = get_chi2(*params)
    if chi2 < best_chi2s[i]:
        best_chi2s[i] = chi2
        best_paramss[i] = params

for step in range(nr_steps):
    print(f"Step {step}")
    print_status(best_paramss, best_chi2s)
    for params in best_paramss:
        for _ in range(5):
            # Mutate once in 50% of cases, twice in 50% of the
            # remaining cases, three times in 50% of those, etc.
            new_params = mutate(params)
            while rng.uniform() < 0.5:
                new_params = mutate(new_params)
            chi2 = get_chi2(new_params[0], new_params[1])
            nr_params = len(new_params[0]) + len(new_params[1])
            if chi2 < best_chi2s[nr_params - 1]:
                best_chi2s[nr_params - 1] = chi2
                best_paramss[nr_params - 1] = new_params
    with open(
        f"data/chi2/chi2s_{layers}layers_path{path}_g{gamma_index}.json", "w"
    ) as f:
        json.dump(chi2dict, f)
