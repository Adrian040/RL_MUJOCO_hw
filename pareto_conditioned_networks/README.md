# Camino: Pareto Conditioned Networks

Implementación de Pareto Conditioned Networks para ambientes multi-objetivo de MuJoCo usando `MO-Gymnasium`.

PCN transforma el problema de aprendizaje por refuerzo multi-objetivo en un problema supervisado. Cada trayectoria recolectada se convierte en ejemplos de la forma:

\[
(s_t, h_t, R_t) \rightarrow a_t,
\]

donde `s_t` es el estado, `h_t` es el horizonte restante, `R_t` es el retorno vectorial restante y `a_t` es la acción tomada en la trayectoria observada. La política se condiciona por un retorno deseado y aprende a seleccionar acciones que reproduzcan trayectorias asociadas con distintos compromisos entre objetivos.

El artículo original formula PCN de manera natural para acciones discretas. Para usar MuJoCo continuo, esta versión emplea un banco fijo de acciones prototipo: la red clasifica sobre ese banco y cada clase corresponde a una acción continua. La comparación oficial se realiza contra el PCN de `morl-baselines`, que incluye una variante con acciones continuas.

## Archivos principales

- `src/train.py`: entrena la implementación local de PCN.
- `src/evaluate.py`: evalúa el checkpoint local condicionado en retornos no dominados.
- `src/train_morl_baselines_pcn.py`: entrena y evalúa el PCN oficial de `morl-baselines`.
- `src/compare_morl_baselines.py`: compara la implementación local contra el baseline oficial.
- `src/aggregate_seeds.py`: agrega resultados de varias semillas.
- `src/networks.py`: arquitectura condicionada por estado, retorno deseado y horizonte.
- `src/dataset.py`: almacenamiento de trayectorias, muestreo y poda del dataset.
- `src/pareto.py`: dominancia, frente no dominado, crowding distance e hipervolumen.

## Ejecución rápida

```bash
python -m pareto_conditioned_networks.src.train \
  --config pareto_conditioned_networks/configs/quick_halfcheetah.yaml

python -m pareto_conditioned_networks.src.evaluate \
  --run-dir pareto_conditioned_networks/results/quick_halfcheetah

python -m pareto_conditioned_networks.src.train_morl_baselines_pcn \
  --config pareto_conditioned_networks/configs/quick_halfcheetah.yaml

python -m pareto_conditioned_networks.src.compare_morl_baselines \
  --local-run pareto_conditioned_networks/results/quick_halfcheetah \
  --baseline-run pareto_conditioned_networks/results/quick_halfcheetah/morl_baselines_pcn \
  --out-dir pareto_conditioned_networks/results/quick_halfcheetah/comparison_morl_baselines
```

## Ejecución para reporte

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

## Salidas generadas

- `training_log.csv`
- `dataset_coverage.csv`
- `evaluation_summary.csv`
- `metrics.json`
- `pcn_coverage.png`
- `pcn_evaluation_front.png`
- `morl_baselines_pcn/evaluation_summary.csv`
- `comparison_morl_baselines/comparison_local_vs_morl_baselines.csv`
- `comparison_morl_baselines/comparison_local_vs_morl_baselines.png`
