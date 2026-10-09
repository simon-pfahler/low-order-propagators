"""Calculate chi2/dof-matrix for rational function fits

Usage:
    get_chi2_matrix.py --layers=<n> --path=<str> --gamma_index=<n> --n_max=<n> --m_max=<n>

Options:
    --layers=<n>        Number of layers in the HC models
    --path=<str>        Path to consider
    --gamma_index=<n>   Gamma structure index to consider
    --n_max=<n>         Maximal power of (mass+4) in the numerator
    --m_max=<n>         Maximal power of (mass+4) in the denominator
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from matplotlib.colors import LogNorm

args = docopt(__doc__)
layers = int(args["--layers"])
path = args["--path"]
gamma_index = int(args["--gamma_index"])
n_max = int(args["--n_max"])
m_max = int(args["--m_max"])

chi2_matrix = np.zeros((m_max, n_max))

for n in range(n_max):
    for m in range(m_max):
        result = subprocess.run(
            [
                "python",
                "scripts/fit_rational_function.py",
                f"--path={path}",
                f"--gamma_index={gamma_index}",
                f"--layers={layers}",
                f"--powers_num={",".join(map(str,range(n+1)))}",
                f"--powers_denom={",".join(map(str,range(m+1)))}",
            ],
            capture_output=True,
            text=True,
        ).stdout
        try:
            chi2_matrix[m, n] = float(result.split("\n")[-2])
        except:
            chi2_matrix[m, n] = np.inf
        print(f"({n}, {m}): {chi2_matrix[m,n]}")

plt.imshow(
    chi2_matrix,
    cmap="viridis_r",
    norm=LogNorm(vmin=chi2_matrix.min(), vmax=np.nan_to_num(chi2_matrix).max()),
)
for n in range(n_max):
    for m in range(m_max):
        plt.text(n, m, f"{chi2_matrix[m,n]:.2e}", ha="center", va="center")
plt.colorbar()
plt.xlabel("Numerator order")
plt.ylabel("Denominator order")
plt.title(
    f"Chi^2/DOF for rational fits of different orders, {layers}-layer HC models\n"
    f"Path {path}, Gamma index {gamma_index}"
)
plt.tight_layout()
plt.show()
