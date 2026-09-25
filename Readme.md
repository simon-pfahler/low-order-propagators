# Learning Low-Order Approximations of the Quark Propagator in Lattice QCD

## Figures
Figures from the paper can be obtained with the `scripts/figure_*.py` scripts.
Running `snakemake` creates all plots and the data needed for them. Be aware that this takes significant time and computational resources.
There are individual rules for each of the plots, which also generate all data necessary for the plot:
```
snakemake Qs_main
snakemake Qs_vs_layers
snakemake coefficient_mass_dependence
snakemake coefficients_vs_layers_main
snakemake history_comparison
snakemake Qs_action_dependence
snakemake Qs_volume_dependence
snakemake coefficient_mass_dependence_pathlength
snakemake convergence_rate
snakemake coefficients_vs_layers
```

## Individual steps
### Model Training
The script `scripts/train.py` trains a model.
You should provide the number of layers (`--layers`), model type (`--model_type`), action (`--action`), lattice size (`--lattice_size`) and mass (`--mass`).
To train seeded model (e.g., to obtain formulas with error bars), provide an integer seed via the `--seed` argument.

### Approximation Quality
The script `scripts/get_approximation_quality.py` calculates the approximation quality of a model, the hopping expansion, or GMRES.
You should provide the number of layers (`--layers`), type (`--type`), action (`--action`), lattice size (`--lattice_size`), action and lattice size trained on (`--model_action`, `--model_lattice_size`) and mass (`--mass`).

### Approximation Coefficients
The script `scripts/get_coefficients.py` extracts the coefficients of a model or the hopping expansion.
You should provide the number of layers (`--layers`), type (`--type`), mass (`--mass`), and if extracting model coefficients, the action (`--model_action`) and lattice size (`--model_lattice_size`).
