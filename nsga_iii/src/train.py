from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .evaluate_policy import evaluate_genome
from .metrics import monte_carlo_hypervolume, normalize_for_max, pareto_front_mask, spacing_metric
from .nsga3 import Individual, nsga3_select, uniform_reference_points
from .operators import polynomial_mutation, sbx_crossover
from .policy import build_policy_spec, random_genome
from .utils import infer_reward_dim, load_config, make_env, save_json, set_seed, smallest_multiple_of_four_at_least


def make_offspring(population: List[Individual], n_offspring: int, config: Dict, rng: np.random.Generator) -> List[Individual]:
    low = float(config["gene_low"])
    high = float(config["gene_high"])
    eta_c = float(config["eta_c"])
    eta_m = float(config["eta_m"])
    crossover_prob = float(config["crossover_prob"])
    genome_size = len(population[0].genome)
    mut_prob = config.get("mutation_prob", "auto")
    mutation_prob = 1.0 / genome_size if mut_prob == "auto" else float(mut_prob)

    children: List[Individual] = []
    while len(children) < n_offspring:
        p1, p2 = rng.choice(population, size=2, replace=True)
        c1, c2 = sbx_crossover(p1.genome, p2.genome, rng, low, high, eta_c, crossover_prob)
        c1 = polynomial_mutation(c1, rng, low, high, eta_m, mutation_prob)
        c2 = polynomial_mutation(c2, rng, low, high, eta_m, mutation_prob)
        children.append(Individual(c1))
        if len(children) < n_offspring:
            children.append(Individual(c2))
    return children


def evaluate_population(population: List[Individual], spec, config: Dict, seed_offset: int) -> None:
    env_id = config["env_id"]
    episodes = int(config["train_episodes"])
    max_episode_steps = config.get("max_episode_steps", None)
    for i, ind in enumerate(population):
        if ind.fitness is None:
            fit, length = evaluate_genome(
                ind.genome,
                spec,
                env_id=env_id,
                episodes=episodes,
                seed=int(config["seed"]) + seed_offset + i * 17,
                max_episode_steps=max_episode_steps,
            )
            ind.fitness = fit
            ind.episode_length = length


def summarize_generation(population: List[Individual], generation: int, eval_count: int, hv_samples: int, seed: int) -> Dict:
    points = np.asarray([ind.fitness for ind in population], dtype=np.float64)
    front_mask = pareto_front_mask(points)
    norm, ideal, nadir = normalize_for_max(points)
    return {
        "generation": generation,
        "evaluations": eval_count,
        "nondominated_count": int(front_mask.sum()),
        "mean_episode_length": float(np.mean([ind.episode_length for ind in population])),
        "mean_sum_return": float(points.sum(axis=1).mean()),
        "best_sum_return": float(points.sum(axis=1).max()),
        "normalized_hv_mc": monte_carlo_hypervolume(norm, samples=hv_samples, seed=seed + generation),
        "spacing": spacing_metric(norm),
        **{f"mean_obj_{i}": float(points[:, i].mean()) for i in range(points.shape[1])},
        **{f"best_obj_{i}": float(points[:, i].max()) for i in range(points.shape[1])},
    }


def save_population(run_dir: Path, population: List[Individual], spec, config: Dict, ref_points: np.ndarray) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    genomes = np.asarray([ind.genome for ind in population], dtype=np.float32)
    fitness = np.asarray([ind.fitness for ind in population], dtype=np.float64)
    np.savez_compressed(
        ckpt_dir / "final_population.npz",
        genomes=genomes,
        fitness=fitness,
        ref_points=ref_points,
        obs_dim=spec.obs_dim,
        action_dim=spec.action_dim,
        hidden_sizes=np.asarray(spec.hidden_sizes, dtype=np.int64),
        action_low=spec.action_low,
        action_high=spec.action_high,
        env_id=np.asarray([config["env_id"]]),
        max_episode_steps=np.asarray([config.get("max_episode_steps", -1)]),
    )

    rows = []
    for i, ind in enumerate(population):
        row = {
            "individual": i,
            "rank": int(ind.rank),
            "ref_index": int(ind.ref_index),
            "ref_distance": float(ind.ref_distance),
            "episode_length": float(ind.episode_length),
        }
        for j, value in enumerate(ind.fitness):
            row[f"train_obj_{j}"] = float(value)
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "final_population.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena políticas con NSGA-III en MO-Gymnasium/MuJoCo.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    run_dir = Path(config["results_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config_used.json", config)

    env = make_env(config["env_id"], seed=seed, max_episode_steps=config.get("max_episode_steps", None))
    spec = build_policy_spec(env, config.get("hidden_sizes", [32]))
    reward_dim = infer_reward_dim(env)
    env.close()

    ref_points = uniform_reference_points(reward_dim, int(config["reference_divisions"]))
    if config.get("population_size", "auto") == "auto":
        population_size = smallest_multiple_of_four_at_least(len(ref_points))
    else:
        population_size = int(config["population_size"])
    if population_size < len(ref_points):
        population_size = smallest_multiple_of_four_at_least(len(ref_points))

    print(f"Ambiente: {config['env_id']}")
    print(f"Resultados: {run_dir}")
    print(f"Dimensión de recompensa: {reward_dim}")
    print(f"Puntos de referencia: {len(ref_points)}")
    print(f"Tamaño de población: {population_size}")
    print(f"Parámetros de política: {spec.n_params}")

    population = [
        Individual(random_genome(spec, rng, float(config["init_scale"]), float(config["gene_low"]), float(config["gene_high"])))
        for _ in range(population_size)
    ]

    logs = []
    eval_count = 0
    evaluate_population(population, spec, config, seed_offset=0)
    eval_count += population_size * int(config["train_episodes"])
    population = nsga3_select(population, population_size, ref_points, rng)
    logs.append(summarize_generation(population, 0, eval_count, int(config.get("hv_samples", 50000)), seed))
    pd.DataFrame(logs).to_csv(run_dir / "generation_log.csv", index=False)

    for generation in tqdm(range(1, int(config["generations"]) + 1), desc="NSGA-III"):
        offspring = make_offspring(population, population_size, config, rng)
        evaluate_population(offspring, spec, config, seed_offset=generation * 10000)
        eval_count += population_size * int(config["train_episodes"])
        population = nsga3_select(population + offspring, population_size, ref_points, rng)
        logs.append(summarize_generation(population, generation, eval_count, int(config.get("hv_samples", 50000)), seed))
        pd.DataFrame(logs).to_csv(run_dir / "generation_log.csv", index=False)
        save_population(run_dir, population, spec, config, ref_points)

    save_population(run_dir, population, spec, config, ref_points)
    print("Entrenamiento terminado.")
    print(f"Log: {run_dir / 'generation_log.csv'}")
    print(f"Población final: {run_dir / 'checkpoints' / 'final_population.npz'}")


if __name__ == "__main__":
    main()
