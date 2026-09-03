# Agerbot

Agerbot es un laboratorio educativo para entrenar un modelo de lenguaje pequeño
desde pesos aleatorios. No descarga ni reutiliza modelos preentrenados.

El runtime 0.2.0 mantiene compatibilidad con el tokenizador byte-level de v1 y
con el tokenizador de caracteres serializado de v2. Nunca reajusta el
tokenizador al cargar un checkpoint.

## Principios

- Código PyTorch portable entre MPS (Mac), CUDA (Windows/Linux) y CPU.
- Arquitectura pequeña, legible y configurable.
- Checkpoints que incluyen modelo, optimizador, configuración y paso actual.
- Entrenamiento local primero; model parallelism se añadirá como una capa
  posterior, sin cambiar el formato del modelo ni de los datos.

## Preparación

Se recomienda Python 3.11 y `uv`:

```bash
uv sync
```

## Primera ejecución

Entrena con el corpus de demostración:

```bash
uv run agerbot-train --config configs/tiny.json
```

Genera texto desde el último checkpoint:

```bash
uv run agerbot-generate \
  --checkpoint checkpoints/latest.pt \
  --prompt "Pregunta: ¿Qué es aprender?\nRespuesta:"
```

Inicia el runtime local para MISIL (solo escucha en este equipo):

```bash
uv run agerbot-serve \
  --checkpoint checkpoints/gastronomia-peruana-v2/best.pt
```

Comprueba que el modelo está listo:

```bash
curl http://127.0.0.1:4318/v1/health
```

El checkpoint debe tener un `manifest.json` válido en el mismo directorio. El
runtime verifica tamaño y SHA-256 antes de cargarlo y no registra conversaciones.

## Preparar una release de modelo

Los checkpoints no se guardan en Git. Se preparan como assets de GitHub Releases:

```bash
uv run python scripts/prepare_release.py \
  --checkpoint checkpoints/gastronomia-peruana-v2/best.pt \
  --evaluation reports/gastronomia-peruana-v2-evaluation.json \
  --version 0.2.0 \
  --published-at 2026-08-25T19:24:51Z
```

El comando verifica la carga restringida del checkpoint, tokenizador,
vocabulario y parámetros finitos; después crea en `dist/releases/0.2.0/` el
modelo, `agerbot-release.json`, evaluación y `checksums-sha256.txt`. No publica
nada. La publicación requiere una acción explícita con GitHub Actions o `gh`.

Estados del ciclo de modelo: `training`, `candidate`, `evaluated`, `stable`,
`rejected` y `superseded`. Solo una publicación explícita convierte un modelo
evaluado en `stable`; MISIL ignora candidatos.

Ejecuta las pruebas:

```bash
uv run python -m unittest discover -s tests
```

## Usar tus propios datos

Coloca uno o varios archivos `.txt` UTF-8 en `data/raw/` y cambia `data_path` en
`configs/tiny.json`. En la siguiente etapa añadiremos un preparador que combine,
limpie, deduplique y divida documentos antes del entrenamiento.

## Qué observar durante el entrenamiento

- `train_loss`: error sobre los lotes utilizados para aprender.
- `val_loss`: error sobre una parte del texto que el optimizador no utiliza.
- Las muestras generadas: permiten ver qué patrones empieza a aprender.
- La diferencia entre ambas pérdidas: si entrenamiento mejora y validación
  empeora, el modelo está memorizando demasiado.

## Camino hacia varias máquinas

El primer objetivo distribuido será pipeline model parallelism: unas capas vivirán
en el Mac y otras en Windows. Se transmitirán activaciones hacia adelante y
gradientes hacia atrás por una red privada. Esto no se activa todavía: antes deben
ser reproducibles el entrenamiento, los checkpoints y la generación local.
