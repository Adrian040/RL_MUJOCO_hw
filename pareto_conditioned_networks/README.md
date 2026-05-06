# Camino: Pareto Conditioned Networks

Implementación de Pareto Conditioned Networks para ambientes multi-objetivo de MuJoCo usando `MO-Gymnasium`.

PCN transforma el problema de aprendizaje por refuerzo multi-objetivo en un problema supervisado. Cada trayectoria recolectada se convierte en ejemplos de la forma:

\[
(s_t, h_t, R_t) \rightarrow a_t,
\]

donde `s_t` es el estado, `h_t` es el horizonte restante, `R_t` es el retorno vectorial restante y `a_t` es la acción que produjo ese retorno en la trayectoria observada. La política se condiciona por el retorno deseado y aprende a seleccionar acciones que reproduzcan trayectorias que alcanzan compromisos específicos entre objetivos.

El artículo original usa acciones discretas. Para usar MuJoCo continuo, esta versión emplea un banco fijo de acciones prototipo. La red clasifica sobre ese banco y cada clase corresponde a una acción continua. Esta adaptación conserva el entrenamiento supervisado de PCN, aunque limita la resolución del espacio de acciones.

## Archivos principales

- `src/train.py`: entrena PCN y guarda dataset, checkpoints y métricas.
- `src/evaluate.py`: evalúa el checkpoint condicionado en objetivos no dominados.
- `src/aggregate_seeds.py`: agrega resultados de varias semillas.
- `src/compare_moppo.py`: comparación opcional contra resultados ya generados de MOPPO.
- `src/networks.py`: arquitectura condicionada por estado, retorno deseado y horizonte.
- `src/dataset.py`: almacenamiento de trayectorias, muestreo y poda del dataset.
- `src/pareto.py`: dominancia, frente no dominado, crowding distance e hipervolumen.

## Ejecución rápida

```bash
python -m pareto_conditioned_networks.src.train \
  --config pareto_conditioned_networks/configs/quick_halfcheetah.yaml

python -m pareto_conditioned_networks.src.evaluate \
  --run-dir pareto_conditioned_networks/results/quick_halfcheetah
```

## Ejecución para reporte

```bash
python -m pareto_conditioned_networks.src.train \
  --config pareto_conditioned_networks/configs/report_halfcheetah.yaml

python -m pareto_conditioned_networks.src.evaluate \
  --run-dir pareto_conditioned_networks/results/report_halfcheetah
```

## Salidas generadas

- `training_log.csv`
- `dataset_coverage.csv`
- `evaluation_summary.csv`
- `metrics.json`
- `pcn_coverage.png`
- `pcn_evaluation_front.png`
- `checkpoints/pcn_final.pt`

## Comparación con MOPPO

Si ya existen resultados de MOPPO:

```bash
python -m pareto_conditioned_networks.src.compare_moppo \
  --pcn-run pareto_conditioned_networks/results/report_halfcheetah_seed_1 \
  --moppo-run multi_objective_ppo/results/report_halfcheetah_seed_1 \
  --out-dir pareto_conditioned_networks/results/comparison_pcn_vs_moppo
```
