import numpy as np

# >>> Parameters
LAYERS = [1,2,3,4,6,8]

ALL_MODEL_TYPES=["HC", "HL", "restricted"]
MODEL_TYPES=["HC", "restricted"]
VOLUME="8c16"

MASSES = [f"{m:.2f}" for m in np.arange(-5, 2.2, 0.2)]

SEEDED_LAYERS=[1,2,3,4,6,8]
SEEDED_MASSES = [f"{m:.2f}" for m in range(-5,3)]
# <<< Parameters

# >>> Helper functions
MODEL_NAMES = []
for model_type in MODEL_TYPES:
    for layer in LAYERS:
        MODEL_NAMES.append(f"{layer}layers_{VOLUME}_{model_type}")
# <<< Helper functions

# >>> Global wildcard constraints
wildcard_constraints:
    layers=r"\d+",
# <<< Global wildcard constraints

# >>> Rules to aggregate results
rule train_main:
    input:
        expand(
            "data/weights/weights_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}.pt",
            layers=LAYERS,
            action="WilsonQuenched",
            lattice_size="8c16",
            model_type=MODEL_TYPES,
            mass=MASSES,
        ),

rule train_seeded_main:
    input:
        expand(
            "data/weights/seeded_weights_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}_seed{seed}.pt",
            layers=SEEDED_LAYERS,
            action="WilsonQuenched",
            lattice_size="8c16",
            model_type=MODEL_TYPES,
            mass=SEEDED_MASSES,
            seed=range(5),
        ),

rule train_action_dependence:
    input:
        expand(
            "data/weights/weights_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}.pt",
            layers=[2,4],
            action=["WilsonQuenched", "WilsonDynamic", "Haar"],
            lattice_size="8c16",
            model_type=MODEL_TYPES,
            mass=MASSES,
        )

rule Qs_main:
    input:
        expand(
            "data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size}_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=LAYERS,
            action="WilsonQuenched",
            lattice_size="16c32",
            model_action="WilsonQuenched",
            model_lattice_size="8c16",
            type=[*MODEL_TYPES],
            mass=MASSES,
        ),
        expand(
            "data/Qs/Qs_{layers}layers_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=LAYERS,
            type=["hopping", "GMRES"],
            action="WilsonQuenched",
            lattice_size="16c32",
            mass=MASSES,
        ),

rule Qs_volume_dependence:
    input:
        expand(
            "data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size}_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=3,
            action="WilsonQuenched",
            lattice_size=["8c16", "16c32"],
            model_action="WilsonQuenched",
            model_lattice_size="8c16",
            type=MODEL_TYPES,
            mass=MASSES,
        ),

rule Qs_action_dependence:
    input:
        expand(
            "data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size}_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=[2,4],
            action=["WilsonQuenched", "WilsonDynamic", "Haar"],
            lattice_size="8c16",
            model_action=["WilsonQuenched", "Haar"],
            model_lattice_size="8c16",
            type=MODEL_TYPES,
            mass=MASSES,
        ),

rule coefficients_main:
    input:
        expand(
            "data/coefficients/coefficients_{layers}layers_{model_action}_{model_lattice_size}_{type}_m{mass}.pt",
            layers=LAYERS,
            model_action="WilsonQuenched",
            model_lattice_size="8c16",
            type=MODEL_TYPES,
            mass=MASSES,
        ),
        expand(
            "data/coefficients/coefficients_{layers}layers_{type}_m{mass}.pt",
            layers=4,
            type="hopping",
            mass=MASSES,
        ),

rule plot_Q_mass_dependence_main:
    input:
        expand(
            "plots/Qs/Qs_{layers}layers_WilsonQuenched_16c32_WilsonQuenched_8c16.pdf",
            layers=LAYERS,
        ),

rule plot_Q_layer_dependence_main:
    input:
        expand(
            "plots/Qs/Qs_vs_layers_WilsonQuenched_16c32_WilsonQuenched_8c16_m{mass}.pdf",
            mass=MASSES,
        ),

rule plot_Q_mass_dependence_volume_dependence:
    input:
        expand(
            "plots/Qs/Qs_3layers_WilsonQuenched_{lattice_size}_WilsonQuenched_8c16.pdf",
            lattice_size=["8c16", "16c32"],
        ),

rule plot_Q_mass_dependence_action_dependence:
    input:
        expand(
            "plots/Qs/Qs_{layers}layers_{action}_8c16_{model_action}_8c16.pdf",
            layers=[2,4],
            action=["WilsonQuenched", "WilsonDynamic", "Haar"],
            model_action=["WilsonQuenched", "WilsonDynamic", "Haar"],
        ),

rule plot_coefficients_mass_dependence_main:
    input:
        expand(
            "plots/mass_dependence/mass_dependence_{layers}layers_{model_type}_{action}_{lattice_size}_pathlength{path_length}.pdf",
            layers=LAYERS,
            model_type=MODEL_TYPES,
            action="WilsonQuenched",
            lattice_size="8c16",
            path_length=range(5),
        )

rule plot_history_comparison_main:
    input:
        expand(
            "plots/histories/history_comparison_{layers}layers_WilsonQuenched_8c16_m-0.8.pdf",
            layers=LAYERS,
        ),

rule plot_convergence_rate_main:
    input:
        "plots/pdf/convergence/convergence_rate_WilsonQuenched_16c32_WilsonQuenched_8c16.pdf",

rule plot_coefficient_vs_layers_main:
    input:
        expand(
            "plots/coefficients/coefficients_vs_layers_{model_type}_{action}_{lattice_size}_p{path}_g{gamma_index}_m{mass}.pdf",
            model_type=["HC", "restricted"],
            action="WilsonQuenched",
            lattice_size="8c16",
            path="[]",
            gamma_index=[0],
            mass=SEEDED_MASSES,
        ),
        expand(
            "plots/coefficients/coefficients_vs_layers_{model_type}_{action}_{lattice_size}_p{path}_g{gamma_index}_m{mass}.pdf",
            model_type=["HC", "restricted"],
            action="WilsonQuenched",
            lattice_size="8c16",
            path=["[(0,1)]", "[(0,2)]"],
            gamma_index=[0,1],
            mass=SEEDED_MASSES,
        ),
        expand(
            "plots/coefficients/coefficients_vs_layers_{model_type}_{action}_{lattice_size}_p{path}_g{gamma_index}_m{mass}.pdf",
            model_type=["HC", "restricted"],
            action="WilsonQuenched",
            lattice_size="8c16",
            path="[(0,1),(1,1)]",
            gamma_index=[0,1,2],
            mass=SEEDED_MASSES,
        ),
# <<< Rules to aggregate results

rule train:
    threads: 8
    resources:
        cores = 8
    output:
        "data/weights/weights_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}.pt",
        "data/histories/history_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}.txt",
    shell:
        "python scripts/train.py --layers={wildcards.layers} --model_type={wildcards.model_type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass}"

rule train_seeded:
    threads: 8
    resources:
        cores = 8
    output:
        "data/weights/seeded_weights_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}_seed{seed}.pt",
        "data/histories/seeded_history_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}_seed{seed}.txt",
    shell:
        "python scripts/train.py --layers={wildcards.layers} --model_type={wildcards.model_type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass} --seed={wildcards.seed}"

rule get_Qs_model:
    threads: 8
    resources:
        cores = 8
    input:
        lambda wildcards:
            "data/weights/weights_{layers}layers_{model_action}_{model_lattice_size}_{type}_m{mass}.pt" if wildcards.type in ALL_MODEL_TYPES else []
    output:
        "data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size}_{type}_{action}_{lattice_size}_m{mass}.pt",
    wildcard_constraints:
        type="HC|HL|restricted",
    shell:
        "python scripts/get_approximation_quality.py --layers={wildcards.layers} --type={wildcards.type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --model_action={wildcards.model_action} --model_lattice_size={wildcards.model_lattice_size} --mass={wildcards.mass}"

rule get_Qs_baseline:
    threads: 8
    resources:
        cores = 8
    output:
        "data/Qs/Qs_{layers}layers_{type}_{action}_{lattice_size}_m{mass}.pt"
    wildcard_constraints:
        type="hopping|GMRES",
    shell:
        "python scripts/get_approximation_quality.py --layers={wildcards.layers} --type={wildcards.type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass}"

rule get_coefficients_model:
    threads: lambda wildcards: 2 * int(wildcards.layers)
    resources:
        cores = lambda wildcards: 2 * int(wildcards.layers)
    input:
        "data/weights/weights_{layers}layers_{action}_{lattice_size}_{type}_m{mass}.pt"
    output:
        "data/coefficients/coefficients_{layers}layers_{action}_{lattice_size}_{type}_m{mass}.pt"
    wildcard_constraints:
        type="HC|HL|restricted",
    shell:
        "python scripts/get_coefficients.py --layers={wildcards.layers} --type={wildcards.type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass}"

rule get_coefficients_model_seeded:
    threads: lambda wildcards: 2 * int(wildcards.layers)
    resources:
        cores = lambda wildcards: 2 * int(wildcards.layers)
    input:
        "data/weights/seeded_weights_{layers}layers_{action}_{lattice_size}_{type}_m{mass}_seed{seed}.pt"
    output:
        "data/coefficients/seeded_coefficients_{layers}layers_{action}_{lattice_size}_{type}_m{mass}_seed{seed}.pt"
    wildcard_constraints:
        type="HC|HL|restricted",
    shell:
        "python scripts/get_coefficients.py --layers={wildcards.layers} --type={wildcards.type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass} --seed={wildcards.seed}"

rule get_coefficients_hopping:
    threads: lambda wildcards: 2 * int(wildcards.layers)
    resources:
        cores = lambda wildcards: 2 * int(wildcards.layers)
    output:
        "data/coefficients/coefficients_{layers}layers_{type}_m{mass}.pt"
    wildcard_constraints:
        type="hopping",
    shell:
        "python scripts/get_coefficients.py --layers={wildcards.layers} --type={wildcards.type} --mass={wildcards.mass}"

rule plot_Q_mass_dependence:
    threads: 1
    resources:
        cores = 1
    input:
        lambda wildcards: expand(
            "data/Qs/Qs_{layers}layers_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=wildcards.layers,
            type=["hopping", "GMRES"],
            action=wildcards.action,
            lattice_size=wildcards.lattice_size,
            mass=MASSES,
        ),
        lambda wildcards: expand(
            "data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size}_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=wildcards.layers,
            type=MODEL_TYPES,
            model_action=wildcards.model_action,
            model_lattice_size=wildcards.model_lattice_size,
            action=wildcards.action,
            lattice_size=wildcards.lattice_size,
            mass=MASSES,
        ),
    output:
        "plots/Qs/Qs_{layers}layers_{action}_{lattice_size}_{model_action}_{model_lattice_size}.pdf",
    shell:
        "python scripts/plot_approximation_quality.py --layers={wildcards.layers} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --model_action={wildcards.model_action} --model_lattice_size={wildcards.model_lattice_size}"

rule plot_Q_layer_dependence:
    threads: 1
    resources:
        cores = 1
    input:
        lambda wildcards: expand(
            "data/Qs/Qs_{layers}layers_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=LAYERS,
            type=["hopping", "GMRES"],
            action=wildcards.action,
            lattice_size=wildcards.lattice_size,
            mass=wildcards.mass,
        ),
        lambda wildcards: expand(
            "data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size}_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=LAYERS,
            type=MODEL_TYPES,
            model_action=wildcards.model_action,
            model_lattice_size=wildcards.model_lattice_size,
            action=wildcards.action,
            lattice_size=wildcards.lattice_size,
            mass=wildcards.mass,
        ),
    output:
        "plots/Qs/Qs_vs_layers_{action}_{lattice_size}_{model_action}_{model_lattice_size}_m{mass}.pdf",
    shell:
        "python scripts/plot_approximation_quality_vs_layers.py --mass={wildcards.mass} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --model_action={wildcards.model_action} --model_lattice_size={wildcards.model_lattice_size}"
    
rule plot_mass_dependence_coefficients:
    threads: 1
    resources:
        cores = 1
    input:
        lambda wildcards: expand(
            "data/coefficients/coefficients_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}.pt",
            layers=LAYERS,
            action=wildcards.action,
            lattice_size=wildcards.lattice_size,
            model_type=wildcards.model_type,
            mass=MASSES,
        ),
        "data/coefficients/coefficients_4layers_hopping_m1.00.pt",
    output:
        "plots/mass_dependence/mass_dependence_{layers}layers_{model_type}_{action}_{lattice_size}_pathlength{path_length}.pdf",
    shell:
        "python scripts/plot_mass_dependence_coefficients.py --layers={wildcards.layers} --model_type={wildcards.model_type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --path_length={wildcards.path_length}"

rule plot_convergence_rate:
    threads: 4
    resources:
        cores = 4
    input:
        lambda wildcards: expand(
            "data/Qs/Qs_{layers}layers_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=LAYERS,
            type=["hopping", "GMRES"],
            action=wildcards.action,
            lattice_size=wildcards.lattice_size,
            mass=MASSES,
        ),
        lambda wildcards: expand(
            "data/Qs/Qs_{layers}layers_{model_action}_{model_lattice_size}_{type}_{action}_{lattice_size}_m{mass}.pt",
            layers=LAYERS,
            type=MODEL_TYPES,
            model_action=wildcards.model_action,
            model_lattice_size=wildcards.model_lattice_size,
            action=wildcards.action,
            lattice_size=wildcards.lattice_size,
            mass=MASSES,
        ),
    output:
        "plots/pdf/convergence/convergence_rate_{action}_{lattice_size}_{model_action}_{model_lattice_size}.pdf",
    shell:
        "python scripts/plot_convergence_rate.py --action={wildcards.action} --lattice_size={wildcards.lattice_size} --model_action={wildcards.model_action} --model_lattice_size={wildcards.model_lattice_size}"

rule plot_history_comparison:
    threads: 1
    resources:
        cores = 1
    input:
        "data/histories/history_{layers}layers_{action}_{lattice_size}_HC_m{mass}.txt",
        "data/histories/history_{layers}layers_{action}_{lattice_size}_HL_m{mass}.txt",
    output:
        "plots/histories/history_comparison_{layers}layers_{action}_{lattice_size}_m{mass}.pdf",
    shell:
        "python scripts/plot_history_comparsion.py --layers={wildcards.layers} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass}"

rule plot_coefficient_vs_layers:
    threads: 1
    resources:
        cores = 1
    input:
        lambda wildcards: expand(
            "data/coefficients/seeded_coefficients_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}_seed{seed}.pt",
            layers=SEEDED_LAYERS,
            action=wildcards.action,
            lattice_size=wildcards.lattice_size,
            model_type=wildcards.model_type,
            mass=wildcards.mass,
            seed=range(5),
        ),
        lambda wildcards: expand(
            "data/coefficients/coefficients_4layers_hopping_m{mass}.pt",
            mass=wildcards.mass,
        ),
    output:
        "plots/coefficients/coefficients_vs_layers_{model_type}_{action}_{lattice_size}_p{path}_g{gamma_index}_m{mass}.pdf",
    shell:
        "python scripts/plot_coefficient_vs_layers.py --model_type={wildcards.model_type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass} --path='{wildcards.path}' --gamma_index={wildcards.gamma_index}"
