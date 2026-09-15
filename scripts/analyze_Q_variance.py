import os

import torch

filenames = os.listdir(f"data/Qs/")
filenames = [f for f in filenames if "hopping" not in f and "GMRES" not in f]

Qs = torch.stack(
    [torch.load(f"data/Qs/{f}", weights_only=True) for f in filenames]
)

variances = Qs.var(dim=(1, 2))
stds = torch.sqrt(variances)
means = Qs.mean(dim=(1, 2))
relerrs = stds / means

max_relerr_idx = torch.argmax(relerrs)

print(
    f"Maximum coefficient of variation observed for {filenames[max_relerr_idx]}"
)
print(
    f"This file has a coefficient of variation of {relerrs[max_relerr_idx]:.2e} = {stds[max_relerr_idx]:.2e} / {means[max_relerr_idx]:.2e}"
)
print(
    f"The average coefficient of variation across all calculations is {relerrs.mean():.2e}"
)
