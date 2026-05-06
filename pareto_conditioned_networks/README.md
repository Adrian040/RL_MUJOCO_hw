# Camino: Pareto Conditioned Networks

Implementación de Pareto Conditioned Networks para ambientes multi-objetivo de MuJoCo usando `MO-Gymnasium`.

PCN transforma el problema de aprendizaje por refuerzo multi-objetivo en un problema supervisado. Cada trayectoria recolectada se convierte en ejemplos de la forma:

\[
(s_t, h_t, R_t) \rightarrow a_t,
\]

donde `s_t` es el estado, `h_t` es el horizonte restante, `R_t` es el retorno vectorial restante y `a_t` es la acción tomada en la trayectoria observada. La política se condiciona por un retorno deseado y aprende a seleccionar acciones que reproducen trayectorias asociadas con distintos compromisos entre objetivos.

Para MuJoCo continuo, la implementación local usa un banco fijo de acciones prototipo: la red clasifica sobre ese banco y cada clase corresponde a una acción continua. La comparación oficial se realiza contra el PCN de `morl-baselines`, que incluye una variante para acciones continuas.

## Ambientes incluidos

La carpeta incluye ejecución multi-ambiente para los ambientes MuJoCo v5 disponibles en MO-Gymnasium:

```text
mo-reacher-v5
mo-hopper-v5
mo-hopper-2obj-v5
mo-halfcheetah-v5
mo-walker2d-v5
mo-ant-v5
mo-ant-2obj-v5
mo-swimmer-v5
mo-humanoid-v5
```

Si se agregan más ambientes en versiones futuras, el script permite limitar a los primeros 10 con `--max-envs 10`.

## Ejecución de un ambiente

```bash
python -m pareto_conditioned_networks.src.train \
  --config pareto_conditioned_networks/configs/report_halfcheetah.yaml

python -m pareto_conditioned_networks.src.evaluate \
  --run-dir pareto_conditioned_networks/results/report_halfcheetah

python -m pareto_conditioned_networks.src.train_morl_baselines_pcn \
  --config pareto_conditioned_networks/configs/report_halfcheetah.yaml

python -m pareto_conditioned_networks.src.compare_morl_baselines \
  --local-run pareto_conditioned_networks/results/report_halfcheetah \
  --baseline-run pareto_conditioned_networks/results/report_halfcheetah/morl_baselines_pcn \
  --out-dir pareto_conditioned_networks/results/report_halfcheetah/comparison_morl_baselines
```

## Ejecución multi-ambiente rápida

```bash
python -m pareto_conditioned_networks.src.run_multi_env \
  --config pareto_conditioned_networks/configs/multi_env_fast.yaml \
  --max-envs 10
```

## Ejecución multi-ambiente para reporte

```bash
python -m pareto_conditioned_networks.src.run_multi_env \
  --config pareto_conditioned_networks/configs/multi_env_report.yaml \
  --max-envs 10
```

Para correr solamente la implementación local y omitir el baseline:

```bash
python -m pareto_conditioned_networks.src.run_multi_env \
  --config pareto_conditioned_networks/configs/multi_env_report.yaml \
  --max-envs 10 \
  --skip-baseline
```

## Salidas principales

Por ambiente se generan carpetas del tipo:

```text
pareto_conditioned_networks/results/multi_env_report/mo_halfcheetah_v5/
pareto_conditioned_networks/results/multi_env_report/mo_hopper_v5/
...
```

Dentro de cada una:

```text
training_log.csv
dataset_coverage.csv
evaluation_summary.csv
metrics.json
pcn_coverage.png
pcn_evaluation_front.png
morl_baselines_pcn/evaluation_summary.csv
comparison_morl_baselines/comparison_local_vs_morl_baselines.csv
comparison_morl_baselines/comparison_local_vs_morl_baselines.png
```

La carpeta global contiene:

```text
multi_env_comparison_summary.csv
multi_env_hv_summary.png
```
