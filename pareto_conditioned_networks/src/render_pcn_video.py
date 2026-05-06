from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

# En Colab no hay ventana/X11. Esta variable debe fijarse antes de importar MuJoCo/Gymnasium.
def configure_mujoco_backend(backend: str) -> None:
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = backend


def flatten_obs(obs: Any):
    import numpy as np

    return np.asarray(obs, dtype=np.float32).reshape(-1)


def normalize_obs(obs, normalizer: Dict):
    return (obs - normalizer["obs_mean"]) / normalizer["obs_std"]


def normalize_condition(condition, normalizer: Dict):
    return (condition - normalizer["cond_mean"]) / normalizer["cond_std"]


def torch_load_checkpoint(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def select_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA no está disponible. Se usará CPU.")
        return torch.device("cpu")
    return requested


def returns_array(episodes):
    import numpy as np

    if not episodes:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([np.asarray(ep["return_vec"], dtype=np.float32) for ep in episodes], axis=0)


def pareto_mask(points):
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    dominated = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        others = np.delete(points, i, axis=0)
        if len(others) == 0:
            continue
        dominated[i] = bool(np.any(np.all(others >= p, axis=1) & np.any(others > p, axis=1)))
    return ~dominated


def select_target(episodes, mode: str, target_index: int | None, num_targets: int) -> Tuple[Any, int, int, Any]:
    import numpy as np

    points = returns_array(episodes)
    if len(points) == 0:
        raise ValueError("El checkpoint no contiene trayectorias para seleccionar targets.")

    lengths = np.asarray([int(ep["length"]) for ep in episodes], dtype=np.int64)
    nd_idx = np.where(pareto_mask(points))[0]
    if len(nd_idx) == 0:
        nd_idx = np.arange(len(points))

    nd_idx = nd_idx[np.argsort(points[nd_idx, 0])]
    if len(nd_idx) > num_targets:
        chosen = np.linspace(0, len(nd_idx) - 1, num_targets).round().astype(int)
        nd_idx = nd_idx[chosen]

    targets = points[nd_idx].astype(np.float32)
    horizons = lengths[nd_idx].astype(np.int64)

    if target_index is not None:
        idx = int(target_index)
        if idx < 0:
            idx = len(targets) + idx
        idx = max(0, min(idx, len(targets) - 1))
    elif mode == "max_obj0":
        idx = int(np.argmax(targets[:, 0]))
    elif mode == "max_obj1":
        idx = int(np.argmax(targets[:, 1]))
    elif mode == "min_obj1":
        idx = int(np.argmin(targets[:, 1]))
    else:
        idx = len(targets) // 2

    return targets[idx], int(horizons[idx]), idx, targets


def make_render_env(env_id: str, seed: int, max_episode_steps: int, width: int | None, height: int | None):
    import gymnasium as gym
    import mo_gymnasium as mo_gym

    kwargs = {"render_mode": "rgb_array"}
    if width is not None:
        kwargs["width"] = int(width)
    if height is not None:
        kwargs["height"] = int(height)

    try:
        env = mo_gym.make(env_id, **kwargs)
    except TypeError:
        env = mo_gym.make(env_id, render_mode="rgb_array")

    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera video/GIF de una política PCN local en mo-halfcheetah-v5.")
    parser.add_argument("--run-dir", type=str, required=True, help="Carpeta de resultados que contiene checkpoints/pcn_final.pt.")
    parser.add_argument("--out-dir", type=str, default=None, help="Carpeta de salida. Por defecto: <run-dir>/videos.")
    parser.add_argument("--video-name", type=str, default="pcn_halfcheetah_demo.mp4")
    parser.add_argument("--gif-name", type=str, default="pcn_halfcheetah_demo.gif")
    parser.add_argument("--make-gif", action="store_true", help="También guarda un GIF submuestreado.")
    parser.add_argument("--target-mode", type=str, default="middle", choices=["middle", "max_obj0", "max_obj1", "min_obj1"], help="Criterio para elegir target de evaluación.")
    parser.add_argument("--target-index", type=int, default=None, help="Índice explícito del target. Tiene prioridad sobre --target-mode.")
    parser.add_argument("--num-targets", type=int, default=None, help="Número de targets no dominados candidatos.")
    parser.add_argument("--seed-offset", type=int, default=12345)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gl", type=str, default="egl", choices=["egl", "osmesa", "glfw"], help="Backend OpenGL. En Colab usa egl; fallback: osmesa.")
    args = parser.parse_args()

    configure_mujoco_backend(args.gl)

    import imageio
    import numpy as np
    import torch

    from pareto_conditioned_networks.src.action_bank import action_from_index
    from pareto_conditioned_networks.src.networks import PCNPolicy

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / args.video_name
    gif_path = out_dir / args.gif_name
    device = select_device(args.device)

    ckpt_path = run_dir / "checkpoints" / "pcn_final.pt"
    ckpt = torch_load_checkpoint(ckpt_path, device)
    config = ckpt["config"]

    model = PCNPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        condition_dim=int(ckpt["reward_dim"]) + 1,
        num_actions=len(ckpt["action_bank"]),
        embedding_dim=int(config.get("embedding_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 64)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_return, target_horizon, selected_idx, targets = select_target(
        episodes=ckpt["episodes"],
        mode=args.target_mode,
        target_index=args.target_index,
        num_targets=int(args.num_targets or config.get("num_eval_targets", 10)),
    )

    print("Targets candidatos:")
    for i, t in enumerate(targets):
        print(f"  {i}: {t}")
    print("\nTarget seleccionado:", selected_idx, target_return)
    print("Horizonte:", target_horizon)

    max_episode_steps = int(config["max_episode_steps"])
    render_seed = int(config.get("seed", 0)) + int(args.seed_offset)
    env = make_render_env(config["env_id"], render_seed, max_episode_steps, args.width, args.height)
    obs, _ = env.reset(seed=render_seed)

    frames = []
    desired_return = np.asarray(target_return, dtype=np.float32).copy()
    horizon = max(int(target_horizon), 1)
    done = False
    step = 0
    episode_return = None

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_vec = flatten_obs(obs)
        cond = np.concatenate([[float(horizon)], desired_return]).astype(np.float32)
        obs_in = normalize_obs(obs_vec, ckpt["normalizer"])
        cond_in = normalize_condition(cond, ckpt["normalizer"])

        obs_tensor = torch.as_tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
        cond_tensor = torch.as_tensor(cond_in, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_idx = int(model.act(obs_tensor, cond_tensor, stochastic=False).item())
        action = action_from_index(ckpt["action_bank"], action_idx)

        obs, reward_vec, terminated, truncated, _ = env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32).reshape(-1)
        if episode_return is None:
            episode_return = np.zeros_like(reward_vec, dtype=np.float64)
        episode_return += reward_vec
        desired_return = desired_return - reward_vec
        horizon = max(horizon - 1, 1)
        done = bool(terminated or truncated)
        step += 1

    frame = env.render()
    if frame is not None:
        frames.append(frame)
    env.close()

    if not frames:
        raise RuntimeError("No se generaron frames. Revisa el backend OpenGL con --gl egl u --gl osmesa.")

    imageio.mimsave(video_path, frames, fps=int(args.fps))
    print("\nPasos del episodio:", step)
    print("Retorno obtenido:", episode_return)
    print("Número de frames:", len(frames))
    print("Video guardado en:", video_path)

    if args.make_gif:
        stride = max(1, int(args.gif_stride))
        imageio.mimsave(gif_path, frames[::stride], fps=int(args.gif_fps))
        print("GIF guardado en:", gif_path)


if __name__ == "__main__":
    main()
