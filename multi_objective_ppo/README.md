# Camino: Multi-objective PPO

Esta carpeta contiene la implementación entregable del camino de descenso/continuación usando una versión práctica de Multi-objective PPO.

## Archivos principales

- `src/train.py`: entrena una política PPO por vector de preferencias.
- `src/evaluate.py`: evalúa los checkpoints y grafica el frente aproximado.
- `src/networks.py`: actor gaussiano y crítico vectorial.
- `src/buffer.py`: buffer on-policy con GAE vectorial.
- `src/pareto.py`: utilidades para puntos no dominados e hipervolumen 2D.
- `src/compare_morl_baselines.py`: script opcional para comparar contra `morl-baselines`.

## Ejecución mínima (prueba)

```bash
python -m multi_objective_ppo.src.train --config multi_objective_ppo/configs/quick_halfcheetah.yaml
python -m multi_objective_ppo.src.evaluate --run-dir multi_objective_ppo/results/quick_halfcheetah
```

## Ejecución final

```bash
python -m multi_objective_ppo.src.train --config multi_objective_ppo/configs/report_halfcheetah.yaml
python -m multi_objective_ppo.src.evaluate --run-dir multi_objective_ppo/results/report_halfcheetah
```
