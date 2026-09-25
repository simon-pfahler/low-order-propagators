import json
import math
from copy import deepcopy

import torch
from qcd_ml.qcd.static import gamma

gamma5 = gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]

generators = torch.zeros(16, 4, 4, dtype=torch.cdouble)
generators[0] = torch.eye(4, dtype=torch.cdouble)
generators[1:5] = torch.stack(gamma)
idx = 5
for mu in range(4):
    for nu in range(mu + 1, 4):
        generators[idx] = (
            1j / 2 * (gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu])
        )
        idx += 1
for mu in range(4):
    generators[idx] = 1j * gamma[mu] @ gamma5
    idx += 1
generators[-1] = gamma5


model_paths = [[]] + [[(mu, d)] for mu in range(4) for d in [1, -1]]


def get_coefficients(W):
    """Calculate the coefficients of the Clifford algebra generators for some
    4x4 matrix W."""
    return 0.25 * torch.einsum("njk,kj->n", generators, W)


def get_coefficients_from_weights(weights, path_length_min, path_length_max):
    contributions = {((), 0): torch.eye(4, dtype=torch.cdouble)}

    for n in range(len(weights)):
        weights_layer = torch.einsum(
            "njk,ion->iojk", generators, weights[f"weights.{n}"]
        )

        new_contributions = {}
        for (path, last_path_idx), contribution in contributions.items():
            for next_path_idx in range(weights_layer.shape[1]):
                new_segment = model_paths[next_path_idx]

                consolidated_new_path = consolidate_path_incremental(
                    path, new_segment
                )

                new_path_length = get_path_length_consolidated(
                    consolidated_new_path
                )
                if new_path_length > 7:
                    continue
                remaining_layers = len(weights) - 1 - n
                if (
                    new_path_length < path_length_min - remaining_layers
                    or new_path_length > path_length_max + remaining_layers
                ):
                    continue

                new_contrib = torch.einsum(
                    "ij,jk->ik",
                    weights_layer[last_path_idx, next_path_idx],
                    contribution,
                )

                # Accumulate contributions
                key = (consolidated_new_path, next_path_idx)
                if key in new_contributions:
                    new_contributions[key] += new_contrib
                else:
                    new_contributions[key] = new_contrib

        contributions = new_contributions

    coefficients = {}
    for (path, _), contribution in contributions.items():
        coefficients[path] = get_coefficients(contribution)

    return coefficients


def get_hopping_weights(mass, nlayers):
    kappa = 1 / (mass + 4) if mass != -4 else torch.nan
    hopping_weights = {
        "weights.0": torch.zeros(1, len(model_paths), 16, dtype=torch.cdouble)
    }
    hopping_weights["weights.0"][0, :, 0] = kappa
    for n in range(1, nlayers + 1):
        if n == nlayers:
            hopping_weights[f"weights.{n}"] = torch.zeros(
                len(model_paths), 1, 16, dtype=torch.cdouble
            )
        else:
            hopping_weights[f"weights.{n}"] = torch.zeros(
                len(model_paths), len(model_paths), 16, dtype=torch.cdouble
            )
        hopping_weights[f"weights.{n}"][0, 0, 0] = 1
        for mu in range(4):
            hopping_weights[f"weights.{n}"][2 * mu + 1, :, 0] = kappa / 2
            hopping_weights[f"weights.{n}"][2 * mu + 1, :, mu + 1] = kappa / 2
            hopping_weights[f"weights.{n}"][2 * mu + 2, :, 0] = kappa / 2
            hopping_weights[f"weights.{n}"][2 * mu + 2, :, mu + 1] = -kappa / 2
    return hopping_weights


def get_weights_from_HC(weights_dict_HC):
    """Transform HC model weights to generic format."""
    weights = dict()

    for n in range(len(weights_dict_HC)):
        weights[f"weights.{n}"] = weights_dict_HC[f"weights.{n}"]
        weights[f"weights.{n}"][0, 0, 0] += 1

    return weights


def get_weights_from_HL(weights_dict_HL):
    """Transform HL model weights to generic format."""
    weights = dict()

    for n in range(len(weights_dict_HL)):
        weights[f"weights.{n}"] = torch.zeros(
            *weights_dict_HL[f"weights.{n}"].shape[:2], 16, dtype=torch.cdouble
        )
        for i in range(weights[f"weights.{n}"].shape[0]):
            for o in range(weights[f"weights.{n}"].shape[1]):
                weights[f"weights.{n}"][i, o] = get_coefficients(
                    weights_dict_HL[f"weights.{n}"][i, o]
                )
        weights[f"weights.{n}"][0, 0, 0] += 1

    return weights


def get_weights_from_restricted(weights_dict_restricted):
    """Transform restricted model weights to generic format."""
    weights = dict()

    # Layer 0 weights
    weights["weights.0"] = torch.zeros(
        1, len(model_paths), 16, dtype=torch.cdouble
    )
    weights["weights.0"][0, :, 0] = weights_dict_restricted["overall_factor"]

    nlayers = weights_dict_restricted["weights"].shape[0]
    # Other layer weights
    for n in range(1, nlayers + 1):
        if n < nlayers:
            weights[f"weights.{n}"] = torch.zeros(
                len(model_paths), len(model_paths), 16, dtype=torch.cdouble
            )
        else:
            weights[f"weights.{nlayers}"] = torch.zeros(
                len(model_paths), 1, 16, dtype=torch.cdouble
            )
        weights[f"weights.{n}"][0, :, 0] = 1
        for mu in range(4):
            weights[f"weights.{n}"][2 * mu + 1, :, 0] = weights_dict_restricted[
                "weights"
            ][n - 1, 0]
            weights[f"weights.{n}"][2 * mu + 1, :, mu + 1] = (
                weights_dict_restricted["weights"][n - 1, 1]
            )
            weights[f"weights.{n}"][2 * mu + 2, :, 0] = weights_dict_restricted[
                "weights"
            ][n - 1, 0]
            weights[f"weights.{n}"][2 * mu + 2, :, mu + 1] = (
                -weights_dict_restricted["weights"][n - 1, 1]
            )

    return weights


def get_config_paths(action, lattice_size_str, test=False):
    """Get the path to a gauge config, given its parameters."""
    with open("parameters.json", "r") as f:
        parameters = json.load(f)

    config_string = "train_configs"
    if test:
        config_string = "test_configs"

    return [
        f"configs/{action}/{lattice_size_str}/{config}.pt"
        for config in parameters[action][lattice_size_str][config_string]
    ]


def consolidate_path(path):
    """Consolidate a path to the form with the minimal number of steps."""

    if len(path) == 0:
        return path

    old_path = deepcopy(path)
    while True:
        # Remove leading segment of length 0
        while old_path[0][1] == 0:
            old_path = deepcopy(old_path[1:])

            # Exit if empty path remains
            if len(old_path) == 0:
                return old_path

        # Add first non-zero-length segment to new path
        new_path = [old_path[0]]

        # Go through segments of path
        for segment_idx in range(1, len(old_path)):
            if old_path[segment_idx][0] == new_path[-1][0]:
                # combine with last segment if they have the same direction
                new_path[-1] = (
                    new_path[-1][0],
                    old_path[segment_idx][1] + new_path[-1][1],
                )
            else:
                # add as a new segment (only if its length is non-zero)
                if old_path[segment_idx][1] != 0:
                    new_path.append(old_path[segment_idx])

        # Break if nothing changed
        if old_path == new_path:
            break

        old_path = deepcopy(new_path)

    return new_path


def consolidate_path_incremental(cpath, new_segment):
    """Incrementally consolidate a path by adding a segment."""

    if not new_segment:
        return cpath

    mu_new, d_new = new_segment[0]

    if d_new == 0:
        return cpath

    if not cpath:
        return ((mu_new, d_new),)

    last_mu, last_d = cpath[-1]

    if mu_new == last_mu:
        new_d = last_d + d_new
        if new_d == 0:
            return cpath[:-1]
        else:
            return cpath[:-1] + ((mu_new, new_d),)

    return cpath + ((mu_new, d_new),)


def canonicalize_path(path):
    """Canonicalize a path so hops are in increasing dimension, and the first
    hop in each dimension is positive."""

    new_order = list()
    new_signs = list()

    for i in range(len(path)):
        if path[i][0] not in new_order:
            new_order.append(path[i][0])
            new_signs.append(1 if path[i][1] > 0 else -1)
    for i in range(4):
        if i not in new_order:
            new_order.append(i)
            new_signs.append(1)

    new_path = list()
    for h, d in path:
        new_h = new_order.index(h)
        new_path.append((new_h, new_signs[new_h] * d))

    return new_path, new_order, new_signs


def generator_mapping(new_order, new_signs):
    """Generate the mapping of generators under the standard ordering to
    generators under an alternative ordering."""

    new_order_generators = [i for i in range(16)]
    new_signs_generators = [1 for _ in range(16)]

    def generator_index_from_gammas(i, j):
        match tuple(sorted((i, j))):
            case (0, 1):
                return 5
            case (0, 2):
                return 6
            case (0, 3):
                return 7
            case (1, 2):
                return 8
            case (1, 3):
                return 9
            case (2, 3):
                return 10
        return -1

    gamma5_sign = 1

    for i in range(4):
        new_order_generators[i + 1] = new_order[i] + 1
        new_signs_generators[i + 1] = new_signs[i]
        for j in range(i + 1, 4):
            gen_idx = generator_index_from_gammas(i, j)
            new_gen_idx = generator_index_from_gammas(
                new_order[i], new_order[j]
            )
            new_order_generators[gen_idx] = new_gen_idx
            new_signs_generators[gen_idx] = (
                new_signs_generators[new_order[i]]
                * new_signs_generators[new_order[j]]
            )
            if new_order[i] > new_order[j]:
                new_signs_generators[gen_idx] *= -1
                gamma5_sign *= -1
    for i in range(4):
        new_order_generators[i + 11] = new_order[i] + 11
        new_signs_generators[i + 11] = new_signs[i] * gamma5_sign
    new_signs_generators[15] = gamma5_sign

    return new_order_generators, new_signs_generators


def get_path_length(path):
    """Get length of a path."""
    cpath = consolidate_path(path)

    res = 0
    for mu, d in cpath:
        res += abs(d)

    return res


def get_path_length_consolidated(cpath):
    """Get length of a consolidated path."""
    res = 0
    for mu, d in cpath:
        res += abs(d)

    return res


def get_gauge_field(lattice_sizes, N_gauge, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return torch.stack(
            [get_SUN_field(lattice_sizes, N_gauge) for _ in range(4)]
        )


def get_SUN_field(lattice_sizes, N_gauge):
    # following https://arxiv.org/pdf/math-ph/0609050
    X = torch.randn(*lattice_sizes, N_gauge, N_gauge, dtype=torch.cdouble)

    Q, R = torch.linalg.qr(X)

    d = torch.diagonal(Q, dim1=-2, dim2=-1)

    ph = d / torch.abs(d)

    Q = torch.einsum("abcdij,abcdj->abcdij", Q, ph)

    det_Q = torch.linalg.det(Q)
    Q /= det_Q.pow(1 / N_gauge).view(*Q.shape[:4], 1, 1).expand(Q.shape)

    return Q


def format_pdg(mean, std):
    """Format (mean, std) in particle physics notation.

    e.g. 3.27(8)e-3 means (3.27 +/- 0.08)e-3, where the bracketed
    number is the uncertainty in the last displayed digit(s).
    Uses 2 significant figures for the uncertainty when the leading
    digit is 1 or 2, otherwise 1 significant figure (PDG convention).
    """
    if std == 0:
        if mean == 0:
            return "0", True
        exp = math.floor(math.log10(abs(mean)))
        norm = mean / 10**exp
        s = f"{norm:.2f}"
        return f"{s}e{exp:+d}" if exp != 0 else s, False

    std_exp = math.floor(math.log10(std))
    std_norm = round(std / 10**std_exp, 10)

    if std_norm < 3:
        last_digit_exp = std_exp - 1
    else:
        last_digit_exp = std_exp

    scale = 10**last_digit_exp

    if mean == 0:
        exp = last_digit_exp
    else:
        exp = math.floor(math.log10(abs(mean)))
        if exp < last_digit_exp:
            exp = last_digit_exp

    decimals = exp - last_digit_exp

    mean_rounded = round(mean / scale)
    std_rounded = math.ceil(std / scale)
    zero = abs(mean_rounded) <= std_rounded

    mean_rounded *= scale

    mean_norm = mean_rounded / 10**exp
    result = f"{mean_norm:.{decimals}f}({int(std_rounded)})"
    if exp != 0:
        result += f"e{exp:+d}"
    return result, zero
