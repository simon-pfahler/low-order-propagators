import numpy as np

# >>> Parameters
LAYERS = [1,2,3,4,6,8]

MODEL_TYPES=["Clifford", "restricted"]
VOLUME="8c16"

MASSES = [f"{m:.2f}" for m in sorted([round(float(m), 2) for m in np.arange(-3.8, 2.2, 0.2).tolist()] + [-0.9, -0.75, -0.7, -0.5])]
# <<< Parameters

# >>> Helper functions
MODEL_NAMES = []
for model_type in MODEL_TYPES:
    for layer in LAYERS:
        MODEL_NAMES.append(f"{layer}layers_{VOLUME}_{model_type}")
# <<< Helper functions

rule all:
    input:
        # Training history plots
        expand("plots/png/histories/history_{model_name}_m{mass}.png", model_name=MODEL_NAMES, mass=MASSES),
        expand("plots/pdf/histories/history_{model_name}_m{mass}.pdf", model_name=MODEL_NAMES, mass=MASSES),
        # Training history comparison plots
        expand("plots/png/histories/history_comparison_{layers}layers_8c16_m-0.75.png", layers=LAYERS),
        expand("plots/pdf/histories/history_comparison_{layers}layers_8c16_m-0.75.pdf", layers=LAYERS),
        # Coefficients plots
        expand("plots/png/coefficients/coefficients_{model_name}_m{mass}.png", model_name=MODEL_NAMES, mass=MASSES),
        expand("plots/pdf/coefficients/coefficients_{model_name}_m{mass}.pdf", model_name=MODEL_NAMES, mass=MASSES),
        expand("plots/png/coefficients/comparison_hopping_{model_name}_m{mass}.png", model_name=MODEL_NAMES, mass=MASSES),
        expand("plots/pdf/coefficients/comparison_hopping_{model_name}_m{mass}.pdf", model_name=MODEL_NAMES, mass=MASSES),
        # Categories plots
        expand("plots/png/categories/categories_{model_name}_m{mass}.png", model_name=MODEL_NAMES, mass=MASSES),
        expand("plots/pdf/categories/categories_{model_name}_m{mass}.pdf", model_name=MODEL_NAMES, mass=MASSES),
        expand("plots/png/categories/comparison_hopping_{model_name}_m{mass}.png", model_name=MODEL_NAMES, mass=MASSES),
        expand("plots/pdf/categories/comparison_hopping_{model_name}_m{mass}.pdf", model_name=MODEL_NAMES, mass=MASSES),
        # Residuals plots
        expand("plots/png/residuals/residuals_{nsteps}steps_{volume}.png", nsteps=LAYERS, volume=[VOLUME, "16c32"]),
        expand("plots/pdf/residuals/residuals_{nsteps}steps_{volume}.pdf", nsteps=LAYERS, volume=[VOLUME, "16c32"]),
        expand("plots/png/residuals/residuals_vs_layers_m{mass}_{volume}.png", mass=MASSES, volume=[VOLUME, "16c32"]),
        expand("plots/pdf/residuals/residuals_vs_layers_m{mass}_{volume}.pdf", mass=MASSES, volume=[VOLUME, "16c32"]),
        # Mass dependence plots
        expand(f"plots/png/mass_dependence/mass_dependence_{{layers}}layers_{VOLUME}_{{model_type}}_pathlength{{path_length}}.png", layers=LAYERS, model_type=MODEL_TYPES, path_length=list(range(5))),
        expand(f"plots/pdf/mass_dependence/mass_dependence_{{layers}}layers_{VOLUME}_{{model_type}}_pathlength{{path_length}}.pdf", layers=LAYERS, model_type=MODEL_TYPES, path_length=list(range(5))),
        # Convergence speed plots
        expand("plots/png/convergence/convergence_speed_{volume}.png", volume=[VOLUME, "16c32"]),
        expand("plots/pdf/convergence/convergence_speed_{volume}.pdf", volume=[VOLUME, "16c32"]),
        # Extra seeded trainings
        expand("data/weights/seeded_weights_{layers}layers_8c16_restricted_m1.00_seed{seed}.pt", layers=LAYERS, seed=range(5)),
        expand("data/histories/seeded_history_{layers}layers_8c16_restricted_m1.00_seed{seed}.txt", layers=LAYERS, seed=range(5))

rule train:
    threads: 8
    resources:
        cores = 8
    input:
        "scripts/train.py",
        "models/{model_name}.json"
    output:
        "data/weights/weights_{model_name}_m{mass}.pt",
        "data/histories/history_{model_name}_m{mass}.txt"
    shell:
        "python scripts/train.py --model {wildcards.model_name} --mass {wildcards.mass}"

rule train_seeded:
    threads: 8
    resources:
        cores = 8
    input:
        "scripts/train.py",
        "models/{model_name}.json"
    output:
        "data/weights/seeded_weights_{model_name}_m{mass}_seed{seed}.pt",
        "data/histories/seeded_history_{model_name}_m{mass}_seed{seed}.txt"
    shell:
        "python scripts/train.py --model {wildcards.model_name} --mass {wildcards.mass} --seed {wildcards.seed}"

rule plot_training_history:
    threads: 1
    resources:
        cores = 1
    input:
        "scripts/plot_history.py",
        "data/histories/history_{model_name}_m{mass}.txt"
    output:
        "plots/png/histories/history_{model_name}_m{mass}.png",
        "plots/pdf/histories/history_{model_name}_m{mass}.pdf"
    shell:
        "python scripts/plot_history.py --model {wildcards.model_name} --mass {wildcards.mass}"

rule plot_history_comparison:
    threads: 1
    resources:
        cores = 1
    input:
        "scripts/plot_history_comparison.py",
        "data/histories/history_{layers}layers_{vol}_Clifford_m{mass}.txt",
        "data/histories/history_{layers}layers_{vol}_4x4_m{mass}.txt"
    output:
        "plots/png/histories/history_comparison_{layers}layers_{vol}_m{mass}.png",
        "plots/pdf/histories/history_comparison_{layers}layers_{vol}_m{mass}.pdf"
    shell:
        "python scripts/plot_history_comparison.py --layers {wildcards.layers} --volume {wildcards.vol} --mass {wildcards.mass}"

rule get_model_residuals16c32:
    threads: 8
    resources:
        cores = 8
    input:
        "scripts/get_residuals16c32.py",
        "data/weights/weights_{nsteps}layers_8c16_{model_type}_m{mass}.pt"
    output:
        "data/residuals/residuals_{nsteps}layers_16c32_{model_type}_m{mass}.pt"
    shell:
        "python scripts/get_residuals16c32.py --model {wildcards.nsteps}layers_8c16_{wildcards.model_type} --mass {wildcards.mass}"

rule get_model_residuals:
    threads: 8
    resources:
        cores = 8
    input:
        "scripts/get_residuals.py",
        "data/weights/weights_{model_name}_m{mass}.pt"
    output:
        "data/residuals/residuals_{model_name}_m{mass}.pt"
    shell:
        "python scripts/get_residuals.py --model {wildcards.model_name} --mass {wildcards.mass}"

rule get_hopping_residuals:
    threads: 8
    resources:
        cores = 8
    input:
        "scripts/get_residuals_hopping.py"
    output:
        "data/residuals/residuals_{nsteps}steps_{volume}_hopping_m{mass}.pt"
    shell:
        "python scripts/get_residuals_hopping.py --nsteps {wildcards.nsteps} --volume {wildcards.volume} --mass {wildcards.mass}"

rule plot_residuals_vs_layers:
    threads: 1
    resources:
        cores = 1
    input:
        "scripts/plot_residuals_vs_layers.py",
        lambda wildcards: expand(
            "data/residuals/residuals_{nsteps}layers_{volume}_{model_type}_m{mass}.pt",
            mass=wildcards.mass, nsteps=LAYERS, volume=wildcards.volume, model_type=MODEL_TYPES
        ),
        lambda wildcards: expand(
            "data/residuals/residuals_{nsteps}steps_{volume}_hopping_m{mass}.pt",
            mass=wildcards.mass, nsteps=LAYERS, volume=wildcards.volume
        )
    output:
        "plots/png/residuals/residuals_vs_layers_m{mass}_{volume}.png",
        "plots/pdf/residuals/residuals_vs_layers_m{mass}_{volume}.pdf"
    shell:
        "python scripts/plot_residuals_vs_layers.py --mass {wildcards.mass} --volume {wildcards.volume}"

rule plot_residuals:
    threads: 1
    resources:
        cores = 1
    input:
        "scripts/plot_residuals.py",
        lambda wildcards: expand(
            "data/residuals/residuals_{nsteps}layers_{volume}_{model_type}_m{mass}.pt",
            mass=MASSES, nsteps=wildcards.nsteps, volume=wildcards.volume, model_type=MODEL_TYPES
        ),
        lambda wildcards: expand(
            "data/residuals/residuals_{nsteps}steps_{volume}_hopping_m{mass}.pt",
            mass=MASSES, nsteps=wildcards.nsteps, volume=wildcards.volume
        )
    output:
        "plots/png/residuals/residuals_{nsteps}steps_{volume}.png",
        "plots/pdf/residuals/residuals_{nsteps}steps_{volume}.pdf"
    shell:
        "python scripts/plot_residuals.py --nsteps {wildcards.nsteps} --volume {wildcards.volume}"

rule plot_mass_dependence:
    threads: 16
    resources:
        cores = 16
    input:
        "scripts/plot_mass_dependence_coefficients.py",
        lambda wildcards: expand(
            "data/weights/weights_{layers}layers_{VOLUME}_{model_type}_m{mass}.pt",
            layers=wildcards.layers, VOLUME=VOLUME, model_type=wildcards.model_type, mass=MASSES
        ),
    output:
        "plots/png/mass_dependence/mass_dependence_{layers}layers_{VOLUME}_{model_type}_pathlength{path_length}.png",
        "plots/pdf/mass_dependence/mass_dependence_{layers}layers_{VOLUME}_{model_type}_pathlength{path_length}.pdf"
    shell:
        "python scripts/plot_mass_dependence_coefficients.py --model_type {wildcards.model_type} --volume {VOLUME} --layers {wildcards.layers} --path_length {wildcards.path_length}"

rule plot_coefficients:
    threads: 16
    resources:
        cores = 16
    input:
        "scripts/plot_coefficients.py",
        "data/weights/weights_{model_name}_m{mass}.pt"
    output:
        "plots/png/coefficients/coefficients_{model_name}_m{mass}.png",
        "plots/pdf/coefficients/coefficients_{model_name}_m{mass}.pdf",
        "plots/png/coefficients/comparison_hopping_{model_name}_m{mass}.png",
        "plots/pdf/coefficients/comparison_hopping_{model_name}_m{mass}.pdf"
    shell:
        "python scripts/plot_coefficients.py --model {wildcards.model_name} --mass {wildcards.mass}"

rule plot_categories:
    threads: 16
    resources:
        cores = 16
    input:
        "scripts/plot_categories.py",
        "data/weights/weights_{model_name}_m{mass}.pt"
    output:
        "plots/png/categories/categories_{model_name}_m{mass}.png",
        "plots/pdf/categories/categories_{model_name}_m{mass}.pdf",
        "plots/png/categories/comparison_hopping_{model_name}_m{mass}.png",
        "plots/pdf/categories/comparison_hopping_{model_name}_m{mass}.pdf"
    shell:
        "python scripts/plot_categories.py --model {wildcards.model_name} --mass {wildcards.mass}"

rule plot_convergence_speed:
    threads: 1
    resources:
        cores = 1
    input:
        "scripts/plot_convergence_speed.py",
        lambda wildcards: expand(
            "data/weights/weights_{layers}layers_8c16_{model_type}_m{mass}.pt",
            layers=LAYERS, model_type=MODEL_TYPES, mass=MASSES
        ),
    output:
        "plots/png/convergence/convergence_speed_{volume}.png",
        "plots/pdf/convergence/convergence_speed_{volume}.pdf"
    shell:
        "python scripts/plot_convergence_speed.py --volume {wildcards.volume}"
