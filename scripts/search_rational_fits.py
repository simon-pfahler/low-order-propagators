"""Search for rational function fits with different numbers of parameters

Usage:
    search_rational_fits.py --layers=<n> --path=<str> --gamma_index=<n> --max_nr_params=<n> [--nr_steps=<n>]

Options:
    --layers=<n>            Number of layers in the HC models
    --path=<str>            Path to consider
    --gamma_index=<n>       Gamma structure index to consider
    --max_nr_params=<n>     Maximum number of parameters in rational function models
    --nr_steps=<n>          Number of steps to run [default: 10]
"""

import ast
import json

import numpy as np
from docopt import docopt
from fit_rational_function import fit_rational_function, get_coef_data
from utility import get_path_length

args = docopt(__doc__)
layers = int(args["--layers"])
path = args["--path"]
path_length = get_path_length(ast.literal_eval(path))
gamma_index = int(args["--gamma_index"])
max_nr_params = int(args["--max_nr_params"])
nr_steps = int(args["--nr_steps"])

masses, coef_means, coef_stds, _, _ = get_coef_data(
    layers, ast.literal_eval(path), gamma_index
)

rng = np.random.default_rng()

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
    if key not in chi2dict.keys():
        try:
            best_fit_params, best_fit_param_errors, chi2 = get_fit(
                powers_num, powers_denom
            )
        except:
            raise ValueError(f"Fit failed for {powers_num}/{powers_denom}")
        chi2dict[key] = float(chi2)
    return chi2dict[key]


def print_status(pop_paramss, pop_chi2s):
    lines = [
        (
            f"\t{i+1} params: "
            f"{','.join(map(str,pop_paramss[i][0]))}/{','.join(map(str,pop_paramss[i][1]))}"
            if pop_paramss[i] is not None
            else f"\t{i+1} params: -"
        )
        for i in range(len(pop_paramss))
    ]
    width = max(map(len, lines))
    for line, chi2 in zip(lines, pop_chi2s):
        print(f"{line:<{width}} -> {chi2}")


def mutate(params):
    h = rng.uniform()
    powers_num, powers_denom = list(params[0]), list(params[1])
    new_num = powers_num.copy()
    new_denom = powers_denom.copy()
    if h < 0.25:
        # Add a power to the denominator
        new_power = int(
            rng.choice(
                [
                    p
                    for p in range(max(powers_denom) + 5)
                    if p not in powers_denom
                ]
            )
        )
        new_denom.append(new_power)
        new_denom = sorted(new_denom)
        if new_power == new_denom[-1]:
            new_num.append(new_power - path_length - 1)
    elif h < 0.5:
        # Add a power to the numerator
        new_power = int(
            rng.choice(
                [p for p in range(max(powers_num) + 5) if p not in powers_num]
            )
        )
        new_num.append(new_power)
        new_num = sorted(new_num)
        if new_power == new_num[-1]:
            new_denom.append(new_power + path_length + 1)
    elif h < 0.75:
        # Remove a power from the denominator
        if len(powers_denom) > 1:
            removed_power = int(rng.choice(powers_denom))
            new_denom.remove(removed_power)
            while (
                len(new_num) > 0
                and new_num[-1] >= new_denom[-1] - path_length - 1
            ):
                del new_num[-1]
            new_num.append(new_denom[-1] - path_length - 1)
    else:
        # Remove a power from the numerator
        if len(powers_num) > 1:
            removed_power = int(rng.choice(powers_num))
            new_num.remove(removed_power)
            while (
                len(new_denom) > 0
                and new_denom[-1] >= new_num[-1] + path_length + 1
            ):
                del new_denom[-1]
            new_denom.append(new_num[-1] + path_length + 1)
    if new_num[-1] < 0 or new_denom[-1] < 0:
        # Reject forms with negative powers
        return params
    return new_num, new_denom


hopping_params = ([0], [path_length + 1])
hopping_chi2 = get_chi2(*hopping_params)

pop_paramss = max_nr_params * [None]
pop_chi2s = max_nr_params * [np.inf]
pop_paramss[1] = hopping_params
pop_chi2s[1] = hopping_chi2

for step in range(nr_steps):
    print(f"Step {step}")
    print_status(pop_paramss, pop_chi2s)
    new_paramss = pop_paramss.copy()
    new_chi2s = pop_chi2s.copy()
    for params in pop_paramss:
        if params is None:
            continue
        for _ in range(5):
            new_params = mutate(params)
            while rng.uniform() < 0.3:
                new_params = mutate(new_params)
            new_nr_params = len(new_params[0]) + len(new_params[1])
            if new_nr_params > max_nr_params:
                continue
            new_chi2 = get_chi2(new_params[0], new_params[1])
            if new_chi2 < new_chi2s[new_nr_params - 1]:
                new_chi2s[new_nr_params - 1] = new_chi2
                new_paramss[new_nr_params - 1] = new_params
    pop_paramss = new_paramss
    pop_chi2s = new_chi2s
    with open(
        f"data/chi2/chi2s_{layers}layers_path{path}_g{gamma_index}.json", "w"
    ) as f:
        json.dump(chi2dict, f)
