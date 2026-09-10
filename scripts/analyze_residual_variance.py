import os

import torch

filenames = os.listdir(f"data/residuals/")
filenames = [f for f in filenames if "hopping" not in f]

Qs = torch.stack(
    [torch.load(f"data/residuals/{f}", weights_only=True) for f in filenames]
)

variances = Qs.var(dim=(1, 2))
stds = torch.sqrt(variances)
means = Qs.mean(dim=(1, 2))
relerrs = stds / means

max_relerr_idx = torch.argmax(relerrs)

print(f"Maximum relative error observed for {filenames[max_relerr_idx]}")
print(
    f"This file has a relative error of {relerrs[max_relerr_idx]:.2e} = {stds[max_relerr_idx]:.2e} / {means[max_relerr_idx]:.2e}"
)
print(
    f"The average relative error across all calculations is {relerrs.mean():.2e}"
)
