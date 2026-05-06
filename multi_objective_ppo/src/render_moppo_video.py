from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# En Colab debe fijarse antes de importar mujoco/gymnasium/mo_gymnasium.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

from .networks import ActorCritic
from .utils import flatten_obs, select_device


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    return torch.load(path, map_location=device, weights_only=False)


def parse_weight_value(value) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=np.float64)
    raise ValueError(f"No se pudo interpretar el peso: {value!r}")


def checkpoint_name_from_summary(run_dir: Path, policy_mode: str) -> Optional[str]:
    summary_path = run_dir / "evaluation_summary.csv"
    if not summary_path.exists():
        return None

    df = pd.read_csv(summary_path)
    if df.empty or "checkpoint" not in df.columns:
        return None

    if policy_mode == "max_obj0" and "mean_obj_0" in df.columns:
        idx = int(df["mean_obj_0"].idxmax())
        return str(df.loc[idx, "checkpoint"])

    if policy_mode == "max_obj1" and "mean_obj_1" in df.columns:
        idx = int(df["mean_obj_1"].idxmax())
        return str(df.loc[idx, "checkpoint"])

    if policy_mode == "best_scalar" and "mean_scalarized_return" in df.columns:
        idx = int(df["mean_scalarized_return"].idxmax())
        return str(df.loc[idx, "checkpoint"])

    if policy_mode == "middle" and "weight" in df.columns:
        weights = np.stack([parse_weight_value(w) for w in df["weight"].tolist()])
        target = np.ones(weights.shape[1], dtype=np.float64) / weights.shape[1]
        idx = int(np.argmin(np.linalg.norm(weights - target, axis=1)))
        return str(df.loc[idx, "checkpoint"])

    return None


def checkpoint_name_from_weights(run_dir: Path, policy_mode: str, device: torch.device) -> str:
    ckpt_paths = sorted((run_dir / "checkpoints").glob("*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No se encontraron checkpoints en {run_dir / 'checkpoints'}")

    if policy_mode == "first":
        return ckpt_paths[0].name

    weights: List[np.ndarray] = []
    for path in ckpt_paths:
        ckpt = load_checkpoint(path, device)
        weights.append(np.asarray(ckpt.get("weight", []), dtype=np.float64))

    if policy_mode == "max_obj0":
        idx = int(np.argmax([w[0] if len(w) > 0 else -np.inf for w in weights]))
        return ckpt_paths[idx].name

    if policy_mode == "max_obj1":
        idx = int(np.argmax([w[1] if len(w) > 1 else -np.inf for w in weights]))
        return ckpt_paths[idx].name

    if policy_mode == "middle":
        valid = [w for w in weights if len(w) > 0]
        if not valid:
            return ckpt_paths[len(ckpt_paths) // 2].name
        dim = len(valid[0])
        target = np.ones(dim, dtype=np.float64) / dim
        distances = [np.linalg.norm(w - target) if len(w) == dim else np.inf for w in weights]
        idx = int(np.argmin(distances))
        return ckpt_paths[idx].name

    return ckpt_paths[0].name


def select_checkpoint(run_dir: Path, checkpoint: Optional[str], policy_mode: str, device: torch.device) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ckpt_dir / checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {ckpt_path}")
        return ckpt_path

    selected = checkpoint_name_from_summary(run_dir, policy_mode)
    if selected is None:
        selected = checkpoint_name_from_weights(run_dir, policy_mode, device)

    ckpt_path = ckpt_dir / selected
    if not ckpt_path.exists():
        raise FileNotFoundError(f"El CSV seleccionó {selected}, pero no existe en {ckpt_dir}")
    return ckpt_path


def make_render_env(env_id: str, seed: int, max_episode_steps: Optional[int], gl_backend: str):
    os.environ["MUJOCO_GL"] = gl_backend
    os.environ["PYOPENGL_PLATFORM"] = gl_backend

    import gymnasium as gym
    import mo_gymnasium as mo_gym

    env = mo_gym.make(env_id, render_mode="rgb_array")
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Renderiza un episodio de una política Multi-objective PPO.")
    parser.add_argument("--run-dir", type=str, required=True, help="Directorio de resultados del entrenamiento.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Nombre o ruta del checkpoint .pt a renderizar.")
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="max_obj0",
        choices=["max_obj0", "max_obj1", "middle", "best_scalar", "first"],
        help="Criterio para seleccionar la política si no se pasa --checkpoint.",
    )
    parser.add_argument("--video-name", type=str, default="moppo_halfcheetah_demo.mp4")
    parser.add_argument("--gif-name", type=str, default=None)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Sobrescribe el máximo de pasos del episodio.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gl", type=str, default="egl", choices=["egl", "osmesa", "glfw"], help="Backend OpenGL para MuJoCo.")
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = args.gl
    os.environ["PYOPENGL_PLATFORM"] = args.gl

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    ckpt_path = select_checkpoint(run_dir, args.checkpoint, args.policy_mode, device)
    ckpt = load_checkpoint(ckpt_path, device)

    config = {}
    config_path = run_dir / "config_used.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    env_id = ckpt.get("env_id", config.get("env_id", "mo-halfcheetah-v5"))
    seed = int(args.seed if args.seed is not None else config.get("seed", 0) + 12345)
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = int(config["max_episode_steps"]) if "max_episode_steps" in config else None

    model = ActorCritic(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        reward_dim=int(ckpt["reward_dim"]),
        action_low=np.asarray(ckpt["action_low"], dtype=np.float32),
        action_high=np.asarray(ckpt["action_high"], dtype=np.float32),
        hidden_sizes=ckpt.get("hidden_sizes", [64, 64]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = make_render_env(env_id, seed=seed, max_episode_steps=max_steps, gl_backend=args.gl)
    obs, _ = env.reset(seed=seed)

    frames = []
    episode_return = np.zeros(int(ckpt["reward_dim"]), dtype=np.float64)
    done = False
    step = 0

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_vec = flatten_obs(obs)
        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model.act_deterministic(obs_tensor).squeeze(0).cpu().numpy()

        obs, reward_vec, terminated, truncated, _ = env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float64).reshape(-1)
        episode_return += reward_vec
        done = bool(terminated or truncated)
        step += 1

        if args.max_steps is not None and step >= args.max_steps:
            done = True

    frame = env.render()
    if frame is not None:
        frames.append(frame)
    env.close()

    if not frames:
        raise RuntimeError("No se generaron frames. Revisa el backend OpenGL con --gl egl u --gl osmesa.")

    video_path = out_dir / args.video_name
    imageio.mimsave(video_path, frames, fps=args.fps)

    gif_path = None
    if args.make_gif:
        gif_name = args.gif_name or video_path.with_suffix(".gif").name
        gif_path = out_dir / gif_name
        imageio.mimsave(gif_path, frames[:: max(args.gif_stride, 1)], fps=args.gif_fps)

    print("Checkpoint:", ckpt_path)
    print("Peso:", ckpt.get("weight", "no disponible"))
    print("Política seleccionada:", args.policy_mode)
    print("Pasos del episodio:", step)
    print("Retorno vectorial:", episode_return)
    print("Frames:", len(frames))
    print("Video guardado en:", video_path)
    if gif_path is not None:
        print("GIF guardado en:", gif_path)


if __name__ == "__main__":
    main()
