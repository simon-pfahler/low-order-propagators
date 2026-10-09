"""Create the point-source propagator error per time slice plot of the paper.

Usage:
    figure_propagator_error_timeslice.py --mass=<f>

Options:
    --mass=<mass>           Mass parameter value
"""

import os
import re

import matplotlib.pyplot as plt
import torch
from docopt import docopt

plt.style.use("scripts/iclr2027.mplstyle")

# Parse docopt arguments
args = docopt(__doc__)
mass = float(args["--mass"])
layerss = [1, 4, 16]

action = "WilsonQuenched"
lattice_size_str = "8c16"
model_action = "WilsonQuenched"
model_lattice_size_str = "8c16"

fig, ax = plt.subplots(1, 3, figsize=(9, 2.3))
plt.subplots_adjust(wspace=0.3, hspace=0.12, top=0.68)

for ax_idx, layers in enumerate(layerss):
    gmres_path = f"data/propagators/timesliced_errors_{layers}layers_GMRES_WilsonQuenched_8c16_m{mass:.2f}.pt"
    hopping_path = f"data/propagators/timesliced_errors_{layers}layers_hopping_WilsonQuenched_8c16_m{mass:.2f}.pt"
    hc_path = f"data/propagators/timesliced_errors_{layers}layers_WilsonQuenched_8c16_HC_WilsonQuenched_8c16_m{mass:.2f}.pt"
    restricted_path = f"data/propagators/timesliced_errors_{layers}layers_WilsonQuenched_8c16_restricted_WilsonQuenched_8c16_m{mass:.2f}.pt"

    try:
        data_gmres = torch.load(gmres_path, weights_only=True)
        data_hopping = torch.load(hopping_path, weights_only=True)
        data_hc = torch.load(hc_path, weights_only=True)
        data_restricted = torch.load(restricted_path, weights_only=True)
    except:
        continue

    for idx in range(4):
        for spin_idx in range(4):
            for color_idx in range(3):
                exact_propagator = torch.load(
                    f"data/propagators/exact_propagator_{action}_{lattice_size_str}_U{idx}_s{spin_idx}_c{color_idx}_m{mass:.2f}.pt",
                    weights_only=True,
                )

                timesliced_norm = (
                    torch.einsum(
                        "xyztsc,xyztsc->t",
                        exact_propagator.conj(),
                        exact_propagator,
                    ).real
                    ** 0.5
                )
                data_gmres[idx, spin_idx, color_idx] /= timesliced_norm
                data_hopping[idx, spin_idx, color_idx] /= timesliced_norm
                data_hc[idx, spin_idx, color_idx] /= timesliced_norm
                data_restricted[idx, spin_idx, color_idx] /= timesliced_norm

    data_gmres_mean = torch.mean(data_gmres, [0, 1, 2])
    data_gmres_std = torch.std(data_gmres, [0, 1, 2])
    data_hopping_mean = torch.mean(data_hopping, [0, 1, 2])
    data_hopping_std = torch.std(data_hopping, [0, 1, 2])
    data_hc_mean = torch.mean(data_hc, [0, 1, 2])
    data_hc_std = torch.std(data_hc, [0, 1, 2])
    data_restricted_mean = torch.mean(data_restricted, [0, 1, 2])
    data_restricted_std = torch.std(data_restricted, [0, 1, 2])

    ax[ax_idx].errorbar(
        torch.arange(data_gmres_mean.shape[0] + 1),
        list(data_gmres_mean) + [data_gmres_mean[0]],
        yerr=list(data_gmres_std) + [data_gmres_std[0]],
        linestyle="none",
        color="#ee3377",
        capsize=4,
        marker="^",
        markersize=4,
        markerfacecolor="none",
        label="GMRES",
        zorder=2,
    )
    ax[ax_idx].errorbar(
        torch.arange(data_restricted_mean.shape[0] + 1),
        list(data_restricted_mean) + [data_restricted_mean[0]],
        yerr=list(data_restricted_std) + [data_restricted_std[0]],
        linestyle="none",
        color="#0077bb",
        capsize=5,
        marker="D",
        markerfacecolor="none",
        label="Restricted model",
        zorder=3,
    )
    ax[ax_idx].errorbar(
        torch.arange(data_hc_mean.shape[0] + 1),
        list(data_hc_mean) + [data_hc_mean[0]],
        yerr=list(data_hc_std) + [data_hc_std[0]],
        linestyle="none",
        color="#ee7733",
        capsize=5,
        marker="s",
        markerfacecolor="none",
        label="HC model",
        zorder=4,
    )
    ax[ax_idx].set_yscale("log")
    ylim = ax[ax_idx].get_ylim()
    ax[ax_idx].errorbar(
        torch.arange(data_hopping_mean.shape[0] + 1),
        list(data_hopping_mean) + [data_hopping_mean[0]],
        yerr=list(data_hopping_std) + [data_hopping_std[0]],
        linestyle="none",
        color="#33bbee",
        capsize=5,
        marker="o",
        markerfacecolor="none",
        label="Hopping expansion",
        zorder=1.9,
    )

    ax[ax_idx].set_xlabel("Time separation on the lattice")
    if ax_idx == 0:
        ax[ax_idx].set_ylabel("Relative propagator error")
        ax[ax_idx].set_title(f"{layers} layer")
    else:
        ax[ax_idx].set_title(f"{layers} layers")
    ax[ax_idx].set_xticks([0, 4, 8, 12, 16])
    ax[ax_idx].set_ylim(ylim)

fig.suptitle(
    f"Relative propagator error vs time separation at $m={mass}$", y=0.98
)

handles, labels = [], []
for axis in ax:
    axis_handles, axis_labels = axis.get_legend_handles_labels()
    if axis_labels:
        handles, labels = axis_handles, axis_labels
        break
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    ncol=4,
    frameon=False,
)

os.makedirs("plots/propagator/", exist_ok=True)

plt.savefig(
    f"plots/propagator/propagator_timesliced_error_m{mass:.2f}.pdf",
    bbox_inches="tight",
)
