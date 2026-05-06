from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena PGMORL de morl-baselines como baseline comparable.")
    parser.add_argument("--env-id", type=str, default="mo-hopper-v5")
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="nsga_iii/results/morl_baselines_pgmorl_hopper")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--pop-size", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=4)
    parser.add_argument("--evolutionary-iterations", type=int, default=2)
    parser.add_argument("--steps-per-iteration", type=int, default=1024)
    args = parser.parse_args()

    try:
        import torch
        import mo_gymnasium as mo_gym
        from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar morl-baselines. Instala con: pip install morl-baselines. "
            f"Error original: {exc!r}"
        ) from exc

    set_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_env = mo_gym.make(args.env_id)
    # Punto peor que retornos razonables para Hopper: avance y salto cerca de 0,
    # costo de control bastante negativo. Se usa solo para evaluación interna.
    ref_point = np.array([0.0, 0.0, -1000.0], dtype=np.float32)
    origin = ref_point.copy()

    agent = PGMORL(
        env_id=args.env_id,
        origin=origin,
        num_envs=args.num_envs,
        pop_size=args.pop_size,
        warmup_iterations=args.warmup_iterations,
        evolutionary_iterations=args.evolutionary_iterations,
        steps_per_iteration=args.steps_per_iteration,
        seed=args.seed,
        device=device,
        log=False,
        num_minibatches=8,
        update_epochs=5,
    )
    agent.train(
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
        num_eval_weights_for_eval=25,
    )

    if hasattr(agent, "archive") and hasattr(agent.archive, "evaluations"):
        evaluations = np.asarray(agent.archive.evaluations, dtype=np.float64)
    else:
        raise RuntimeError("No se encontró agent.archive.evaluations en la versión instalada de morl-baselines.")

    if evaluations.ndim == 1:
        evaluations = evaluations.reshape(1, -1)
    df = pd.DataFrame(evaluations, columns=[f"mean_obj_{i}" for i in range(evaluations.shape[1])])
    df.to_csv(out_dir / "pgmorl_evaluations.csv", index=False)
    save_json(out_dir / "config_used.json", vars(args))
    print(f"Evaluaciones guardadas en {out_dir / 'pgmorl_evaluations.csv'}")


if __name__ == "__main__":
    main()
