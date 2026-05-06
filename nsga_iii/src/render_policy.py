from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

# Debe fijarse antes de importar gymnasium/mo_gymnasium/mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
import numpy as np
import pandas as pd

try:
    from .policy import PolicySpec, act
    from .utils import flatten_obs
except ImportError:
    from nsga_iii.src.policy import PolicySpec, act
    from nsga_iii.src.utils import flatten_obs


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_config(run_dir: Path) -> dict:
    config_path = run_dir / "config_used.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_population(run_dir: Path):
    ckpt_path = run_dir / "checkpoints" / "final_population.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No se encontró {ckpt_path}. Primero ejecuta el entrenamiento de NSGA-III."
        )

    data = np.load(ckpt_path, allow_pickle=True)
    spec = PolicySpec(
        obs_dim=int(data["obs_dim"]),
        action_dim=int(data["action_dim"]),
        hidden_sizes=[int(x) for x in data["hidden_sizes"].tolist()],
        action_low=data["action_low"].astype(np.float32),
        action_high=data["action_high"].astype(np.float32),
    )
    return data, spec


def objective_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("mean_obj_")]
    if cols:
        return cols
    cols = [c for c in df.columns if c.startswith("Objetivo ")]
    if cols:
        return cols
    cols = [c for c in df.columns if c.startswith("obj_")]
    return cols


def choose_individual(
    run_dir: Path,
    n_genomes: int,
    selection: str,
    objective_index: int,
    individual: Optional[int],
    seed: int,
) -> tuple[int, Optional[pd.Series]]:
    if individual is not None:
        if individual < 0 or individual >= n_genomes:
            raise ValueError(f"individual debe estar entre 0 y {n_genomes - 1}.")
        return int(individual), None

    eval_path = run_dir / "evaluation_summary.csv"
    if not eval_path.exists():
        print(
            f"No se encontró {eval_path}. Se usará el primer individuo de la población. "
            "Para seleccionar mejor, ejecuta antes nsga_iii.src.evaluate."
        )
        return 0, None

    df = pd.read_csv(eval_path)

    if "individual" not in df.columns:
        raise ValueError("evaluation_summary.csv debe contener la columna 'individual'.")

    if "is_nondominated" in df.columns:
        candidates = df[df["is_nondominated"].astype(bool)].copy()
    elif "No dominada" in df.columns:
        candidates = df[df["No dominada"].astype(bool)].copy()
    else:
        candidates = df.copy()

    if len(candidates) == 0:
        candidates = df.copy()

    obj_cols = objective_columns(candidates)
    if "mean_sum_return" in candidates.columns:
        sum_col = "mean_sum_return"
    elif "Suma retornos" in candidates.columns:
        sum_col = "Suma retornos"
    elif obj_cols:
        candidates = candidates.copy()
        candidates["_sum_return"] = candidates[obj_cols].sum(axis=1)
        sum_col = "_sum_return"
    else:
        sum_col = None

    if selection == "best_sum":
        if sum_col is None:
            selected = candidates.iloc[0]
        else:
            selected = candidates.sort_values(sum_col, ascending=False).iloc[0]
    elif selection == "middle":
        if sum_col is None:
            selected = candidates.iloc[len(candidates) // 2]
        else:
            ordered = candidates.sort_values(sum_col).reset_index(drop=True)
            selected = ordered.iloc[len(ordered) // 2]
    elif selection == "random":
        rng = np.random.default_rng(seed)
        selected = candidates.iloc[int(rng.integers(0, len(candidates)))]
    elif selection == "best_objective":
        if not obj_cols:
            raise ValueError("No se encontraron columnas de objetivos para usar best_objective.")
        if objective_index < 0 or objective_index >= len(obj_cols):
            raise ValueError(f"objective_index debe estar entre 0 y {len(obj_cols) - 1}.")
        selected = candidates.sort_values(obj_cols[objective_index], ascending=False).iloc[0]
    else:
        raise ValueError(f"Modo de selección no reconocido: {selection}")

    return int(selected["individual"]), selected


def print_selection_summary(individual: int, row: Optional[pd.Series]) -> None:
    print("Individuo seleccionado:", individual)
    if row is None:
        return

    if "is_nondominated" in row.index:
        print("No dominada:", bool(row["is_nondominated"]))
    elif "No dominada" in row.index:
        print("No dominada:", bool(row["No dominada"]))

    if "mean_sum_return" in row.index:
        print("Suma de retornos evaluada:", float(row["mean_sum_return"]))
    elif "Suma retornos" in row.index:
        print("Suma de retornos evaluada:", float(row["Suma retornos"]))

    obj_cols = [c for c in row.index if c.startswith("mean_obj_") or c.startswith("Objetivo ")]
    if obj_cols:
        print("Retornos por objetivo evaluados:")
        for c in obj_cols:
            print(f"  {c}: {float(row[c]):.3f}")


def render_episode(
    genome: np.ndarray,
    spec: PolicySpec,
    env_id: str,
    seed: int,
    max_episode_steps: Optional[int],
):
    import gymnasium as gym
    import mo_gymnasium as mo_gym

    env = mo_gym.make(env_id, render_mode="rgb_array")
    if max_episode_steps is not None and max_episode_steps > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    obs, _ = env.reset(seed=seed)
    frames = []
    done = False
    steps = 0
    episode_return = None

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_vec = flatten_obs(obs)
        action = act(genome, obs_vec, spec)
        obs, reward_vec, terminated, truncated, _ = env.step(action)

        reward_vec = np.asarray(reward_vec, dtype=np.float64).reshape(-1)
        if episode_return is None:
            episode_return = np.zeros_like(reward_vec, dtype=np.float64)
        episode_return += reward_vec

        done = bool(terminated or truncated)
        steps += 1

    frame = env.render()
    if frame is not None:
        frames.append(frame)
    env.close()

    if episode_return is None:
        episode_return = np.zeros(0, dtype=np.float64)
    return frames, episode_return, steps


def maybe_display_video(video_path: Path, embed: bool) -> None:
    try:
        from IPython.display import Video, display

        display(Video(str(video_path), embed=embed))
    except Exception as exc:
        print("No se pudo mostrar el video automáticamente:", repr(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Renderiza una política entrenada con NSGA-III.")
    parser.add_argument("--run-dir", type=str, default="nsga_iii/results/report_hopper_robust")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--video-name", type=str, default="nsga3_hopper_demo")
    parser.add_argument(
        "--selection",
        type=str,
        default="best_sum",
        choices=["best_sum", "middle", "random", "best_objective"],
    )
    parser.add_argument("--objective-index", type=int, default=0)
    parser.add_argument("--individual", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif", action="store_true", help="También guarda un GIF.")
    parser.add_argument("--display", action="store_true", help="Muestra el MP4 si se ejecuta dentro de notebook.")
    parser.add_argument("--embed", action="store_true", help="Inserta el video embebido al usar --display.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = _read_config(run_dir)
    data, spec = load_population(run_dir)
    genomes = data["genomes"]

    env_id = _as_str(data["env_id"][0]) if "env_id" in data else str(config.get("env_id", "mo-hopper-v5"))

    if args.max_episode_steps is not None:
        max_episode_steps = int(args.max_episode_steps)
    elif "max_episode_steps" in config and config["max_episode_steps"] is not None:
        max_episode_steps = int(config["max_episode_steps"])
    elif "max_episode_steps" in data:
        max_episode_steps = int(data["max_episode_steps"][0])
        if max_episode_steps <= 0:
            max_episode_steps = None
    else:
        max_episode_steps = None

    selected_idx, row = choose_individual(
        run_dir=run_dir,
        n_genomes=len(genomes),
        selection=args.selection,
        objective_index=args.objective_index,
        individual=args.individual,
        seed=args.seed,
    )
    print_selection_summary(selected_idx, row)
    print("Ambiente:", env_id)
    print("Máximo de pasos:", max_episode_steps)

    frames, episode_return, steps = render_episode(
        genome=genomes[selected_idx],
        spec=spec,
        env_id=env_id,
        seed=args.seed,
        max_episode_steps=max_episode_steps,
    )

    if not frames:
        raise RuntimeError("No se generaron frames. Revisa que el ambiente soporte render_mode='rgb_array'.")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{args.video_name}.mp4"
    gif_path = out_dir / f"{args.video_name}.gif"

    imageio.mimsave(video_path, frames, fps=args.fps)

    print("Pasos del episodio:", steps)
    print("Retorno obtenido:", episode_return)
    print("Suma de retornos:", float(np.sum(episode_return)))
    print("Número de frames:", len(frames))
    print("Video guardado en:", video_path)

    if args.gif:
        gif_frames = frames[:: max(1, args.fps // 15)]
        imageio.mimsave(gif_path, gif_frames, fps=min(15, args.fps))
        print("GIF guardado en:", gif_path)

    if args.display:
        maybe_display_video(video_path, embed=args.embed)


if __name__ == "__main__":
    main()
