# AI-Project-SEST-Alert-System

Sistema de alerta basado en ML para estimar riesgo en SEST y generar salidas para un mapa territorial (provincias/cantones).

## Estructura del repo

- `artifacts/`
  - `preprocess_bundle.pkl` (scaler + PCA + columnas de entrada)
  - `best_model_bundle.pkl` (modelo + threshold + pc_cols)
- `assets/geo/`
  - `provincias.geojson`
  - `cantones.geojson`
- `scripts/`
  - `preprocesamiento_sin_smote_varianza95_reglasclaras.py` (reproducir entrenamiento)
  - `generate_outputs.py` (inferir y crear outputs para mapas)  *(si aplica)*
- `notebook/`
  - `ML's_Code.ipynb` (entrenamiento + validación + mapas)
- `outputs/` (se genera al correr inferencia; no subir a git)
- `data/` (datasets; recomendado NO subir datos sensibles)

## Instalación

```bash
pip install -r requirements.txt
