"""Calculate chi2/dof for rational function fits with different numbers of parameters

Usage:
    get_chi2_vs_nr_params.py --layers=<n> --path=<str> --gamma_index=<n> --max_nr_params=<n>

Options:
    --layers=<n>            Number of layers in the HC models
    --path=<str>            Path to consider
    --gamma_index=<n>       Gamma structure index to consider
    --max_nr_params=<n>     Maximum number of parameters in the fit
"""

import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from matplotlib.colors import LogNorm

args = docopt(__doc__)
layers = int(args["--layers"])
path = args["--path"]
gamma_index = int(args["--gamma_index"])
max_nr_params = int(args["--max_nr_params"])

nr_masses = (
    36  # Number of masses fitted, cf. masses in fit_rational_function.py
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

extrapdict = dict()
try:
    with open(
        f"data/chi2/extraps_{layers}layers_path{path}_g{gamma_index}.json"
    ) as f:
        extrapdict = json.load(f)
except:
    pass


def get_chi2(powers_num, powers_denom):
    powers_num_str = ",".join(map(str, powers_num))
    powers_denom_str = ",".join(map(str, powers_denom))
    key = powers_num_str + "/" + powers_denom_str
    if key in chi2dict.keys():
        return chi2dict[key]
    try:
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
        result = float(stdout.split("\n")[-2])
    except:
        raise ValueError(f"Fit failed for {powers_num}/{powers_denom}")
    try:
        extrapdict[key] = float(
            stdout.split("Extrapolation error at mass")[1].split("\n")[1]
        )
    except:
        pass
    chi2dict[key] = result
    return result


def print_status(best_paramss, best_chi2s):
    lines = [
        f"\t{i+1} params: {','.join(map(str,best_paramss[i][0]))}/{','.join(map(str,best_paramss[i][1]))}"
        for i in range(max_nr_params)
    ]
    # BIC = chi2 + nr_params * ln(nr_masses), where the cached reduced
    # chi2 is chi2 / (nr_masses - nr_params)
    bics = [
        best_chi2s[i] * (nr_masses - i - 1) + (i + 1) * np.log(nr_masses)
        for i in range(max_nr_params)
    ]
    best_bic_idx = int(np.argmin(bics))
    width = max(map(len, lines))
    for i, (line, chi2, bic) in enumerate(zip(lines, best_chi2s, bics)):
        marker = " <- BIC optimal" if i == best_bic_idx else ""
        print(f"{line:<{width}} -> {chi2},\tBIC {bic:.1f}{marker}")


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
                range(0, max_nr_params + 1)
                if idx == 0
                else range(1, max_nr_params + 1)
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


best_paramss = [(list(range(i + 1)), []) for i in range(max_nr_params)]
best_chi2s = [get_chi2(e[0], e[1]) for e in best_paramss]

for step in range(100):
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
    with open(
        f"data/chi2/extraps_{layers}layers_path{path}_g{gamma_index}.json", "w"
    ) as f:
        json.dump(extrapdict, f)
