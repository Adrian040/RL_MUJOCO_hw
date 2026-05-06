# RL_MUJOCO_hw

Implementaciones de aprendizaje por refuerzo multi-objetivo en ambientes MuJoCo usando `MO-Gymnasium`.

El repositorio está organizado por caminos metodológicos. Cada implementación (carpeta) contiene su propio `README`, configuraciones, scripts de entrenamiento y evaluación.

Nota: Las versiones finales ya están contenidas en la rama master, no es necesario cambiar a las otras ramas en caso de bajar el repositorio.

## Caminos implementados

### Descenso/Continuación
- Multi-objective PPO

Carpeta:
```text
multi_objective_ppo/
```

Incluye entrenamiento, evaluación, aproximación de frente de Pareto y comparación con `morl-baselines`.

---

### Basados en valor
- Pareto Conditioned Networks (PCN)

Carpeta:
```text
pareto_conditioned_networks/
```

Implementación de PCN adaptada a MuJoCo continuo mediante un banco de acciones prototipo.

Incluye entrenamiento condicionado por retornos deseados, evaluación multiobjetivo y comparación opcional contra MOPPO.

---

## Caminos reservados

```text
nsga_iii/
videos/
```

**(Hasta este punto estos no se han realizado/implementado)**.

---

## Ambiente principal

```text
mo-halfcheetah-v5
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Ejecución

Cada implementación tiene instrucciones específicas en su respectivo `README`.

Ejemplo general:

```bash
python -m <metodo>.src.train --config <config.yaml>
python -m <metodo>.src.evaluate --run-dir <results_dir>
```

---

## Notebooks

El repositorio incluye notebooks de reporte y reproducción de experimentos:

```text
Tarea5_RL_AJMO.ipynb
Tarea5_PCN_AJMO.ipynb
```

---

## Resultados

Cada implementación guarda resultados dentro de su carpeta `results/`, incluyendo:

- Logs de entrenamiento
- Evaluaciones
- Frentes de Pareto aproximados
- Hipervolumen
- Figuras y checkpoints

---

## Uso de GPU

Los scripts usan CUDA automáticamente cuando está disponible; en caso contrario se ejecutan en CPU.

---

## Limitaciones

- Entrenamientos realizados con presupuesto computacional moderado.
- Comparaciones con `morl-baselines` usadas como baseline aproximado.
- La implementación de PCN adapta un método originalmente discreto a acciones continuas mediante discretización por banco de acciones.
- NSGA-III queda pendiente como posible extensión.