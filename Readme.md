# Optimal Low-order Approximations of Quark Propagators

## Model Training
The script `scripts/train.py` trains a model.
You should provide the number of layers (`--layers`), model type (`--model_type`), action (`--action`), lattice size (`--lattice_size`) and mass (`--mass`).

To train all HC- and restricted models with 1,2,3,4,6 and 8 layers for masses between -5 and 2, with quenched Wilson action gauge fields on an 8^3x16 lattice, run

```
snakemake train_main
```

To train seeded model (e.g., to obtain formulas with error bars), provide an integer seed via the `--seed` argument.
To train seeded HC- and restricted models with 1,2,3 and 4 layers for masses between -5 and 2, with quenched Wilson action gauge fields on an 8^3x16 lattice for seeds 1,2,3,4,5, run

```
snakemake train_seeded_main
```

## Approximation Quality
The script `scripts/get_approximation_quality.py` calculates the approximation quality of a model, the hopping expansion, or GMRES.
You should provide the number of layers (`--layers`), type (`--type`), action (`--action`), lattice size (`--lattice_size`), action and lattice size trained on (`--model_action`, `--model_lattice_size`) and mass (`--mass`).

To calculate all approximation qualities for HC- and restricted models (trained on 8^3x16 quenched Wilson action gauge fields), hopping expansions and GMRES with 1,2,3,4,6 and 8 layers for masses between -5 and 2, using quenched Wilson action gauge fields on a 16^3x32 lattice, run

```
snakemake Qs_main
```

## Approximation Coefficients
The script `scripts/get_coefficients.py` extracts the coefficients of a model or the hopping expansion.
You should provide the number of layers (`--layers`), type (`--type`), mass (`--mass`), and if extracting model coefficients, the action (`--model_action`) and lattice size (`--model_lattice_size`).

To extract all coefficients up to paths of length 4 for HC- and restricted models (trained on 8^3x16 quenched Wilson action gauge fields) with 1,2,3,4,6 and 8 layers, and for the hopping expansion with 4 layers, for masses between -5 and 2, run

```
snakemake coefficients_main
```

## Approximation Quality Plots
The script `scripts/plot_approximation_quality.py` creates plots of the mass-dependence of the approximation quality.
You should provide the number of layers (`--layers`), action and lattice size used for the approximation quality calculation (`--action`, `--lattice_size`) and action and lattice size used for training (`--model_action`, `--model_lattice_size`).

To create plots of the approximation quality for models with 1,2,3,4,5,6 and 8 layers for the 16^3x32 quenched Wilson action gauge fields, with models trained on the same action but the 8^3x16 volume, run

```
snakemake plot_Q_mass_dependence_main
```

The script `scripts/plot_approximation_quality_vs_layers.py` creates plots of the layer-dependence of the approximation quality.
You should provide the action and lattice size used for the approximation quality (`--action`, `--lattice_size`) and for training (`--model_action`, `--model_lattice_size`), and the mass parameter (`--mass`).

To create plots of the approximation quality for HC- and restricted models for the 16^3x32 quenched Wilson action gauge fields, with models trained on the same action but the 8^3x16 volume, for masses between -5 and 2, run

```
snakemake plot_Q_layer_dependence_main
```

## Coefficients Plots
The script `scripts/plot_mass_dependence_coefficients.py` creates plots of the mass-dependence of the coefficients.
You should provide the number of layers (`--layers`), model type (`--model_type`), action and lattice size trained on (`--action`, `--lattice_size`), and the path length (`--path_length`).

To create plots of the coefficients for models with 1,2,3,4,5,6 and 8 layers trained on 8^3x16 quenched Wilson action gauge fields, for path lengths 0 to 4, run

```
snakemake plot_coefficients_mass_dependence_main
```

## Formulas from restricted models
The script `scripts/print_formula_restricted.py` prints the formula learned by a restricted model.
You should provide the number of layers (`--layers`), action and lattice size trained on (`--action`, `--lattice_size`), and the mass parameter (`--mass`).

## Coefficients from HC-models
The script `scripts/print_coefficients_HC.py` prints the coefficients learned by an HC-model.
You should provide the number of layers (`--layers`), action and lattice size trained on (`--action`, `--lattice_size`), the mass parameter (`--mass`), and the path to print the coefficients of (`--path`).

The script `scripts/plot_coefficients_vs_layers.py` plots how the coefficients learned by an HC- or restricted model change with model depth.
You should provide the number of layers (`--model_type`), action and lattice size used in training (`--action`, `--lattice_size`), mass parameter (`--mass`), a path in its canonical form (`--path`) and the index of the gamma structure whose coefficient should be plotted (`--gamma_index`).

To create plots for all canonical paths of length 0, 1 and 2, run
```
snakemake plot_coefficient_vs_layers_main
```

## Additional plots
There are additional rules in `Snakefile` that can be used to generate data and plots for other analyses:

```
snakemake train_action_dependence
snakemake Qs_volume_dependence
snakemake Qs_action_dependence
snakemake plot_history_comparison_main
snakemake plot_convergence_rate_main
snakemake plot_Qs_mass_dependence_volume_dependence
snakemake plot_Qs_mass_dependence_action_dependence
```
