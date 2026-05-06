# Camino: NSGA-III

Esta carpeta contiene una implementación de NSGA-III aplicada a un ambiente MuJoCo multi-objetivo. La política se representa como una red neuronal determinista pequeña y la población evoluciona directamente sobre sus parámetros.

## Ambiente

Se usa `mo-hopper-v5` para no repetir `mo-halfcheetah-v5`. Este ambiente es más ligero en dimensión de observación y acción, y su recompensa vectorial tiene tres componentes:

- objetivo 0: avance en el eje x;
- objetivo 1: altura/salto;
- objetivo 2: costo de control.

Como el objetivo 2 es un costo reportado como recompensa negativa, el algoritmo lo maximiza de forma natural al hacerlo menos negativo.

## Idea general

La población contiene políticas completas. Cada individuo se evalúa ejecutando episodios en el ambiente y acumulando el retorno vectorial. Después, NSGA-III selecciona la siguiente generación mediante:

1. ordenamiento no dominado;
2. normalización adaptativa de objetivos;
3. asociación a puntos de referencia;
4. preservación de nichos alrededor de esos puntos;
5. recombinación SBX y mutación polinomial.

El código usa el criterio de maximización propio de RL, pero internamente invierte el signo de los retornos para aplicar las reglas de NSGA-III en forma de minimización, como se presenta usualmente en optimización evolutiva.

## Archivos principales

- `src/train.py`: entrenamiento evolutivo con NSGA-III.
- `src/evaluate.py`: evaluación final y generación de figuras.
- `src/nsga3.py`: selección por ordenamiento no dominado y puntos de referencia.
- `src/policy.py`: política MLP determinista codificada como vector de parámetros.
- `src/operators.py`: SBX y mutación polinomial.
- `src/metrics.py`: frente no dominado, hipervolumen aproximado y normalización.
- `src/aggregate_seeds.py`: resumen de múltiples semillas.
- `src/compare_morl_baselines.py`: comparación opcional con PGMORL de `morl-baselines`.
- `src/compare_methods.py`: comparación visual y métricas entre NSGA-III y el baseline.

## Ejecución rápida

```bash
python -m nsga_iii.src.train --config nsga_iii/configs/quick_hopper.yaml
python -m nsga_iii.src.evaluate --run-dir nsga_iii/results/quick_hopper
```

## Ejecución recomendada para reporte

```bash
python -m nsga_iii.src.train --config nsga_iii/configs/report_hopper.yaml
python -m nsga_iii.src.evaluate --run-dir nsga_iii/results/report_hopper
```

Para múltiples semillas, se puede copiar el YAML cambiando `seed` y `results_dir`, o usar el notebook entregado.

## Comparación con MORL-Baselines

`morl-baselines` no incluye una implementación directa de NSGA-III. Por ello, se usa PGMORL como baseline comparable, ya que también mantiene una población de políticas y está diseñado para espacios continuos.

```bash
python -m nsga_iii.src.compare_morl_baselines \
  --env-id mo-hopper-v5 \
  --total-timesteps 120000 \
  --out-dir nsga_iii/results/morl_baselines_pgmorl_hopper \
  --pop-size 5 \
  --warmup-iterations 4 \
  --evolutionary-iterations 2 \
  --steps-per-iteration 1024 \
  --num-envs 4
```

Después de tener ambos resultados:

```bash
python -m nsga_iii.src.compare_methods \
  --nsga-path nsga_iii/results/report_hopper_seed_0/evaluation_summary.csv \
  --baseline-path nsga_iii/results/morl_baselines_pgmorl_hopper/pgmorl_evaluations.csv \
  --out-dir nsga_iii/results/comparison_nsga3_vs_pgmorl
```

## Configuración sugerida en Colab

Para mantener el tiempo total bajo control, la configuración `report_hopper.yaml` usa una población pequeña y pocas generaciones. En Colab, con GPU disponible para el baseline y CPU para la simulación de NSGA-III, el tiempo esperado para dos semillas de NSGA-III más PGMORL con 120k pasos suele quedar dentro de una ventana corta de experimentación. Si sobra tiempo, conviene aumentar `generations` de 8 a 12 o usar tres semillas.

## Resultados generados

Cada corrida guarda:

- `generation_log.csv`
- `final_population.csv`
- `evaluation_summary.csv`
- `metrics.json`
- `front_3d.png`
- `objective_pairplots.png`
- `value_path.png`
- `checkpoints/final_population.npz`
