from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .buffer import RolloutBuffer
from .networks import ActorCritic
from .utils import flatten_obs, load_config, make_env, parse_weights, safe_weight_name, save_json, select_device, set_seed


def infer_reward_dim(env) -> int:
    """Obtiene la dimensión de la recompensa vectorial."""
    if hasattr(env, "reward_space"):
        return len(np.asarray(env.reward_space.sample()).reshape(-1))
    obs, _ = env.reset()
    action = env.action_space.sample()
    _, reward_vec, _, _, _ = env.step(action)
    env.reset()
    return len(np.asarray(reward_vec).reshape(-1))


def train_one_weight(config: Dict, weight: np.ndarray, run_dir: Path, base_seed: int) -> List[Dict]:
    """Entrena PPO para un vector de preferencias específico."""
    env = make_env(config["env_id"], base_seed)
    obs, _ = env.reset(seed=base_seed)
    obs = flatten_obs(obs)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = infer_reward_dim(env)
    obs, _ = env.reset(seed=base_seed)
    obs = flatten_obs(obs)

    device = select_device(config.get("device", "auto"))
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        action_low=env.action_space.low.reshape(-1),
        action_high=env.action_space.high.reshape(-1),
        hidden_sizes=config.get("hidden_sizes", [64, 64]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), eps=1e-5)
    buffer = RolloutBuffer(
        num_steps=int(config["num_steps"]),
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        device=device,
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        weight=weight,
    )

    total_timesteps = int(config["total_timesteps_per_weight"])
    num_steps = int(config["num_steps"])
    num_updates = max(1, total_timesteps // num_steps)
    batch_size = num_steps
    num_minibatches = int(config["num_minibatches"])
    minibatch_size = max(1, batch_size // num_minibatches)
    weight_name = safe_weight_name(weight)

    logs: List[Dict] = []
    global_step = 0
    episode_return_vec = np.zeros(reward_dim, dtype=np.float64)
    episode_length = 0
    episode_idx = 0

    pbar = tqdm(range(1, num_updates + 1), desc=f"Entrenando {weight_name}")
    for update in pbar:
        buffer.reset()

        for _ in range(num_steps):
            global_step += 1
            episode_length += 1
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                raw_action, logprob, _, value_vec, clipped_action = model.get_action_and_value(obs_tensor)

            action_np = clipped_action.squeeze(0).cpu().numpy()
            next_obs, reward_vec, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            reward_vec = np.asarray(reward_vec, dtype=np.float32).reshape(-1)

            buffer.add(obs, raw_action.squeeze(0), logprob.squeeze(0), reward_vec, done, value_vec)

            episode_return_vec += reward_vec
            obs = flatten_obs(next_obs)

            if done:
                logs.append({
                    "weight": weight.tolist(),
                    "weight_name": weight_name,
                    "global_step": global_step,
                    "update": update,
                    "episode": episode_idx,
                    "episode_length": episode_length,
                    "scalarized_return": float(np.dot(weight, episode_return_vec)),
                    **{f"return_obj_{i}": float(v) for i, v in enumerate(episode_return_vec)},
                })
                episode_idx += 1
                episode_return_vec = np.zeros(reward_dim, dtype=np.float64)
                episode_length = 0
                obs, _ = env.reset()
                obs = flatten_obs(obs)

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_value_vec = model.get_value(next_obs_tensor).squeeze(0)
        batch = buffer.compute_returns_and_advantages(next_value_vec, next_done=False)

        b_inds = np.arange(batch_size)
        clip_coef = float(config["clip_coef"])
        update_epochs = int(config["update_epochs"])
        vf_coef = float(config["vf_coef"])
        ent_coef = float(config["ent_coef"])
        max_grad_norm = float(config["max_grad_norm"])
        norm_adv = bool(config.get("norm_adv", True))

        for _epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_values_vec, _ = model.get_action_and_value(batch.obs[mb_inds], batch.actions[mb_inds])
                ratio = (newlogprob - batch.logprobs[mb_inds]).exp()

                mb_advantages = batch.advantages_scalar[mb_inds]
                if norm_adv and mb_advantages.numel() > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = F.mse_loss(new_values_vec, batch.returns_vec[mb_inds])
                entropy_loss = entropy.mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if logs:
            recent = logs[-1]
            pbar.set_postfix({"scalar_ret": f"{recent['scalarized_return']:.1f}", "ep_len": int(recent["episode_length"])})

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{weight_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "weight": weight.tolist(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "reward_dim": reward_dim,
        "action_low": env.action_space.low.reshape(-1).tolist(),
        "action_high": env.action_space.high.reshape(-1).tolist(),
        "hidden_sizes": config.get("hidden_sizes", [64, 64]),
        "env_id": config["env_id"],
    }, ckpt_path)
    env.close()
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena Multi-objective PPO en MO-Gymnasium/MuJoCo.")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo YAML de configuración.")
    args = parser.parse_args()

    config = load_config(args.config)
    weights = parse_weights(config["weights"])
    seed = int(config.get("seed", 0))
    set_seed(seed)

    run_dir = Path(config["results_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config_used.json", config)

    print(f"Ambiente: {config['env_id']}")
    print(f"Resultados: {run_dir}")
    print(f"Dispositivo: {select_device(config.get('device', 'auto'))}")
    print(f"Pesos: {weights.tolist()}")

    all_logs: List[Dict] = []
    for idx, weight in enumerate(weights):
        logs = train_one_weight(config, weight, run_dir, base_seed=seed + 1000 * idx)
        all_logs.extend(logs)
        pd.DataFrame(all_logs).to_csv(run_dir / "training_log.csv", index=False)

    print("Entrenamiento terminado.")
    print(f"Log guardado en: {run_dir / 'training_log.csv'}")
    print(f"Checkpoints guardados en: {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
