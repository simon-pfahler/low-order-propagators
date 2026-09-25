import numpy as np

# >>> Parameters
MASSES = [f"{m:.2f}" for m in np.arange(-5, 2.2, 0.2)]
SMALL_MASSES = [f"{m:.2f}" for m in np.arange(-5, -0.4, 0.2)]
LARGE_MASSES = [f"{m:.2f}" for m in np.arange(-0.4, 2.2, 0.2)]
SEEDED_MASSES = [f"{m:.2f}" for m in range(-5,3)]
# <<< Parameters

# >>> Global wildcard constraints
wildcard_constraints:
    layers=r"\d+",
# <<< Global wildcard constraints

# >>> Rules to create figures
rule all:
    input:
        "plots/Qs/Qs_main.pdf",
        "plots/Qs/Qs_vs_layers.pdf",
        "plots/mass_dependence/coefficient_mass_dependence.pdf",
        "plots/coefficients/coefficients_vs_layers_main.pdf",
        "plots/histories/history_comparison.pdf",
        "plots/Qs/Qs_action_dependence.pdf",
        "plots/Qs/Qs_volume_dependence.pdf",
        "plots/mass_dependence/coefficient_mass_dependence_pathlength0.pdf",
        "plots/mass_dependence/coefficient_mass_dependence_pathlength1.pdf",
        "plots/mass_dependence/coefficient_mass_dependence_pathlength2.pdf",
        "plots/convergence/convergence_rate.pdf",
        "plots/coefficients/coefficients_vs_layers_appendix_path[]_g0.pdf",
        "plots/coefficients/coefficients_vs_layers_appendix_path[(0, 1)]_g0.pdf",
        "plots/coefficients/coefficients_vs_layers_appendix_path[(0, 1)]_g1.pdf",
        "plots/coefficients/coefficients_vs_layers_appendix_path[(0, 1), (1, 1), (0, -1)]_g0.pdf",
        "plots/coefficients/coefficients_vs_layers_appendix_path[(0, 3)]_g0.pdf",

rule Qs_main:
    input:
        expand(
            "data/Qs/Qs_{layers}layers_{type}_WilsonQuenched_16c32_m{mass}.pt",
            layers=[1,4],
            type=["hopping", "GMRES"],
            mass=MASSES,
        ),
        expand(
            "data/Qs/Qs_{layers}layers_WilsonQuenched_8c16_{type}_WilsonQuenched_16c32_m{mass}.pt",
            layers=[1,4],
            type=["HC", "restricted"],
            mass=MASSES,
        ),
    output:
        "plots/Qs/Qs_main.pdf",
    shell:
        "python scripts/figure_approximation_quality_main.py"

rule Qs_vs_layers:
    input:
        expand(
            "data/Qs/Qs_{layers}layers_{type}_WilsonQuenched_16c32_m1.40.pt",
            layers=[1,2,3,4,6],
            type=["hopping", "GMRES"],
        ),
        expand(
            "data/Qs/Qs_{layers}layers_{type}_WilsonQuenched_16c32_m-3.00.pt",
            layers=[1,2,3,4,6,8,12,16],
            type=["hopping", "GMRES"],
        ),
        expand(
            "data/Qs/Qs_{layers}layers_WilsonQuenched_8c16_{type}_WilsonQuenched_16c32_m1.40.pt",
            layers=[1,2,3,4,6],
            type=["HC", "restricted"],
        ),
        expand(
            "data/Qs/Qs_{layers}layers_WilsonQuenched_8c16_{type}_WilsonQuenched_16c32_m-3.00.pt",
            layers=[1,2,3,4,6,8,12,16],
            type=["HC", "restricted"],
        ),
    output:
        "plots/Qs/Qs_vs_layers.pdf",
    shell:
        "python scripts/figure_coefficient_vs_layers_main.py"

rule coefficient_mass_dependence:
    input:
        "data/coefficients/coefficients_4layers_hopping_m1.00.pt",
        expand(
            "data/coefficients/coefficients_4layers_WilsonQuenched_8c16_{type}_m{mass}.pt",
            type=["HC", "restricted"],
            mass=MASSES,
        ),
    output:
        "plots/mass_dependence/coefficient_mass_dependence.pdf",
    shell:
        "python scripts/figure_coefficient_mass_dependence_main.py"

rule coefficients_vs_layers_main:
    input:
        "data/coefficients/coefficients_4layers_hopping_m-3.00.pt",
        expand(
            "data/coefficients/seeded_coefficients_{layers}layers_WilsonQuenched_8c16_HC_m-3.00_seed{seed}.pt",
            layers=[1,2,3,4,6,8,12,16],
            seed=range(5),
        ),
    output:
        "plots/coefficients/coefficients_vs_layers_main.pdf",
    shell:
        "python scripts/figure_coefficient_vs_layers_main.py"

rule history_comparison:
    input:
        expand(
            "data/histories/history_{layers}layers_WilsonQuenched_8c16_{type}_m-0.80.txt",
            layers=[1,4],
            type=["HC", "HL"],
        ),
    output:
        "plots/histories/history_comparison.pdf",
    shell:
        "python scripts/figure_history_comparison.py"

rule Qs_action_dependence:
    input:
        expand(
            "data/Qs/Qs_4layers_{type}_{action}_8c16_m{mass}.pt",
            type=["hopping", "GMRES"],
            action=["WilsonQuenched", "WilsonDynamic", "Haar"],
            mass=MASSES,
        ),
        expand(
            "data/Qs/Qs_4layers_{type}_{action}_8c16_{type}_{model_action}_8c16_m{mass}.pt",
            type=["HC", "restricted"],
            action=["WilsonQuenched", "WilsonDynamic", "Haar"],
            model_action=["WilsonQuenched", "WilsonDynamic", "Haar"],
            mass=MASSES,
        ),
    output:
        "plots/Qs/Qs_action_dependence.pdf",
    shell:
        "python scripts/figure_approximation_quality_action_dependence.py"

rule Qs_volume_dependence:
    input:
        expand(
            "data/Qs/Qs_4layers_{type}_WilsonQuenched_{vol}_m{mass}.pt",
            type=["hopping", "GMRES"],
            vol=["8c16", "16c32"],
            mass=MASSES,
        ),
        expand(
            "data/Qs/Qs_4layers_{type}_WilsonQuenched_8c16_{type}_WilsonQuenched_{vol}_m{mass}.pt",
            type=["HC", "restricted"],
            vol=["8c16", "16c32"],
            mass=MASSES,
        ),
    output:
        "plots/Qs/Qs_volume_dependence.pdf",
    shell:
        "python scripts/figure_approximation_quality_volume_dependence.py"

rule coefficient_mass_dependence_pathlength:
    input:
        "data/coefficients/coefficients_4layers_hopping_m1.00.pt",
        expand(
            "data/coefficients/coefficients_{layers}layers_WilsonQuenched_8c16_HC_m{mass}.pt",
            layers=[1,2,3,4],
            mass=MASSES,
        ),
    output:
        "plots/mass_dependence/coefficient_mass_dependence_pathlength{pathlength}.pdf",
    shell:
        "python scripts/figure_coefficient_mass_dependence_appendix.py --path_length={wildcards.pathlength}"

rule convergence_rate:
    input:
        expand(
            "data/Qs/Qs_{layers}layers_{type}_WilsonQuenched_16c32_m{mass}.pt",
            layers=[1,2,3,4,6,8,12,16],
            mass=SMALL_MASSES,
            type=["hopping", "GMRES"],
        ),
        expand(
            "data/Qs/Qs_{layers}layers_{type}_WilsonQuenched_16c32_m{mass}.pt",
            layers=[1,2,3,4,6],
            mass=LARGE_MASSES,
            type=["hopping", "GMRES"],
        ),
        expand(
            "data/Qs/Qs_{layers}layers_{type}_WilsonQuenched_8c16_{type}_WilsonQuenched_16c32_m{mass}.pt",
            layers=[1,2,3,4,6,8,12,16],
            mass=SMALL_MASSES,
            type=["HC", "restricted"],
        ),
        expand(
            "data/Qs/Qs_{layers}layers_{type}_WilsonQuenched_8c16_{type}_WilsonQuenched_16c32_m{mass}.pt",
            layers=[1,2,3,4,6],
            mass=LARGE_MASSES,
            type=["HC", "restricted"],
        ),
    output:
        "plots/convergence/convergence_rate.pdf",
    shell:
        "python scripts/figure_convergence_rate.py"

rule coefficients_vs_layers:
    input:
        expand(
            "data/coefficients/seeded_coefficients_{layers}layers_WilsonQuenched_8c16_restricted_m{mass}_seed{seed}.pt",
            layers=[1,2,3,4,6],
            mass=SEEDED_MASSES,
            seed=range(5),
        ),
        expand(
            "data/coefficients/seeded_coefficients_{layers}layers_WilsonQuenched_8c16_HC_m{mass}_seed{seed}.pt",
            layers=[1,2,3,4,6,8,12,16],
            mass=SEEDED_MASSES,
            seed=range(5),
        ),
        expand(
            "data/coefficients/coefficients_4layers_hopping_m{mass}.pt",
            mass=SEEDED_MASSES,
        ),
    output:
        "plots/coefficients/coefficients_vs_layers_appendix_path{path}_g{gamma_index}.pdf",
    shell:
        "python scripts/figure_coefficient_vs_layers_appendix.py --path='{wildcards.path}' --gamma_index={wildcards.gamma_index}"
# <<< Rules to create figures

rule train:
    threads: 8
    resources:
        cores = 8
    output:
        "data/weights/weights_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}.pt",
        "data/histories/history_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}.txt",
    wildcard_constraints:
        model_type="HC|HL|restricted",
    shell:
        "python scripts/train.py --layers={wildcards.layers} --model_type={wildcards.model_type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass}"

rule train_seeded:
    threads: 8
    resources:
        cores = 8
    output:
        "data/weights/seeded_weights_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}_seed{seed}.pt",
        "data/histories/seeded_history_{layers}layers_{action}_{lattice_size}_{model_type}_m{mass}_seed{seed}.txt",
    wildcard_constraints:
        model_type="HC|HL|restricted",
    shell:
        "python scripts/train.py --layers={wildcards.layers} --model_type={wildcards.model_type} --action={wildcards.action} --lattice_size={wildcards.lattice_size} --mass={wildcards.mass} --seed={wildcards.seed}"

rule get_Qs_model:
    threads: 8
    resources:
        cores = 8
    input:
        "data/weights/weights_{layers}layers_{model_action}_{model_lattice_size}_{type}_m{mass}.pt"
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
