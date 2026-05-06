# RL_MUJOCO_hw

Implementaciones para problemas MuJoCo multi-objetivo.

**Nota:** Las versiones finales ya están contenidas en la rama master, no es necesario cambiar a las otras ramas en caso de bajar el repositorio.

## Caminos implementados

- Descenso/Continuación: Multi-objective PPO
- Basados en valor: Pareto Conditioned Networks
- Basados en política/evolutivo: NSGA-III

## Ambientes usados

- `mo-halfcheetah-v5` para Multi-objective PPO y Pareto Conditioned Networks.
- `mo-hopper-v5` para NSGA-III.

## Organización

Cada camino tiene su propia carpeta con configuración, código fuente, resultados y README específico:

```text
multi_objective_ppo/
pareto_conditioned_networks/
nsga_iii/
videos/
```

## Reproducción general

1. Instalar dependencias.
2. Ejecutar el entrenamiento del camino correspondiente.
3. Evaluar checkpoints o población final.
4. Generar tablas, métricas y figuras.
5. Comparar con un baseline de `morl-baselines` cuando sea posible.

Las instrucciones concretas de ejecución están dentro del README de cada carpeta.


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

## Algunas imitaciones

- Entrenamientos realizados con presupuesto computacional relativamente moderado (Colab version gratuita).
- Comparaciones con `morl-baselines` usadas como baseline aproximado para NSGA-III y para Multi-objective PPO
- La implementación de PCN adapta un método originalmente discreto a acciones continuas mediante discretización por banco de acciones.