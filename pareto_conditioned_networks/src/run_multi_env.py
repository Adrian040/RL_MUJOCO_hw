from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

from .utils import env_dir_name, load_config, save_yaml

DEFAULT_MUJOCO_V5_ENVS = [
    "mo-reacher-v5",
    "mo-hopper-v5",
    "mo-hopper-2obj-v5",
    "mo-halfcheetah-v5",
    "mo-walker2d-v5",
    "mo-ant-v5",
    "mo-ant-2obj-v5",
    "mo-swimmer-v5",
    "mo-humanoid-v5",
]


def run_command(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_env_config(base_config: dict, env_id: str, root_dir: Path, seed_offset: int) -> dict:
    cfg = dict(base_config)
    cfg["env_id"] = env_id
    cfg["seed"] = int(base_config.get("seed", 0)) + seed_offset
    cfg["experiment_name"] = f"{base_config.get('experiment_name', 'multi_env')}_{env_dir_name(env_id)}"
    cfg["results_dir"] = str(root_dir / env_dir_name(env_id))
    cfg.pop("env_ids", None)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta PCN local y MORL-Baselines PCN en varios ambientes MO-Gymnasium MuJoCo.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env-ids", nargs="*", default=None)
    parser.add_argument("--max-envs", type=int, default=10)
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)
    env_ids = args.env_ids or base_config.get("env_ids", DEFAULT_MUJOCO_V5_ENVS)
    env_ids = list(env_ids)[: int(args.max_envs)]
    root_dir = Path(base_config["results_dir"])
    root_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = root_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for idx, env_id in enumerate(env_ids):
        env_name = env_dir_name(env_id)
        env_dir = root_dir / env_name
        cfg = build_env_config(base_config, env_id, root_dir, seed_offset=1000 * idx)
        cfg_path = generated_dir / f"{env_name}.yaml"
        save_yaml(cfg_path, cfg)

        print("\n" + "=" * 80)
        print(f"Ambiente {idx + 1}/{len(env_ids)}: {env_id}")
        print("=" * 80)

        if not args.skip_local:
            run_command([sys.executable, "-m", "pareto_conditioned_networks.src.train", "--config", str(cfg_path)])
            run_command([sys.executable, "-m", "pareto_conditioned_networks.src.evaluate", "--run-dir", str(env_dir)])

        if not args.skip_baseline:
            run_command([sys.executable, "-m", "pareto_conditioned_networks.src.train_morl_baselines_pcn", "--config", str(cfg_path)])

        compare_dir = env_dir / "comparison_morl_baselines"
        if not args.skip_compare and not args.skip_local and not args.skip_baseline:
            local_eval = env_dir / "evaluation_summary.csv"
            baseline_eval = env_dir / "morl_baselines_pcn" / "evaluation_summary.csv"
            if local_eval.exists() and baseline_eval.exists():
                run_command([
                    sys.executable,
                    "-m",
                    "pareto_conditioned_networks.src.compare_morl_baselines",
                    "--local-run",
                    str(env_dir),
                    "--baseline-run",
                    str(env_dir / "morl_baselines_pcn"),
                    "--out-dir",
                    str(compare_dir),
                ])

        comparison_csv = compare_dir / "comparison_local_vs_morl_baselines.csv"
        if comparison_csv.exists():
            df = pd.read_csv(comparison_csv)
            for _, row in df.iterrows():
                summary = row.to_dict()
                summary["Ambiente"] = env_id
                summary_rows.append(summary)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        cols = ["Ambiente"] + [c for c in summary.columns if c != "Ambiente"]
        summary = summary[cols]
        summary.to_csv(root_dir / "multi_env_comparison_summary.csv", index=False)
        print("\nResumen multi-ambiente:")
        print(summary.round(3).to_string(index=False))
        try:
            run_command([sys.executable, "-m", "pareto_conditioned_networks.src.aggregate_envs", "--root-dir", str(root_dir)])
        except subprocess.CalledProcessError:
            pass


if __name__ == "__main__":
    main()
