# AIS Anomaly Detection (H3)

Proyecto Python para deteccion de anomalias AIS con `IsolationForest`,
contexto geografico H3 y graficos interactivos.

## Scripts principales (estado actual)

- `load_ais_data.py`: carga y preproceso AIS (fechas, limpieza, H3).
- `train_anomaly.py`: entrenamiento y exportacion de artefactos.
- `predict_realtime.py`: inferencia batch y por registro.
- `plot_anomalies.py`: grafico HTML de anomalias.

## Requisitos

- Python `>=3.11` (recomendado `3.14.x`).
- Dependencias en `requirements.txt`.

## Instalacion rapida (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si ya usas el entorno del repo, puedes ejecutar con `Scripts\python.exe`.

## Datos de entrada

- Entrenamiento completo: `data/ais-data.csv`
- Pruebas rapidas (inferencia/graficos): `data/ais-data-sample.csv`

Tambien se usa el mapa base desde:

- `shp/world.shp`
- `shp/world.shx`
- `shp/world.dbf`

## Flujo recomendado

### 1) Entrenar modelo

```powershell
python train_anomaly.py
```

Artefactos generados en `models/`:

- `isolation_forest_model.joblib`
- `scaler.joblib`
- `imputer.joblib`
- `h3_stats.joblib`
- `h3_parent_stats.joblib`
- `h3_config.json`
- `metadata.json`

Ademas se genera `data/anomalies_summary.csv`.

### 2) Inferencia

Batch por CLI:

```powershell
python predict_realtime.py
python predict_realtime.py data/ais-data-sample.csv
```

Como modulo:

```python
from predict_realtime import AISAnomalyDetector
from load_ais_data import preprocess

detector = AISAnomalyDetector()
df = preprocess("data/ais-data-sample.csv")
out = detector.predict(df)
print(out[["is_anomaly", "anomaly_score", "anomaly_reason"]].head())
```

### 3) Graficos

Ruta rapida (recomendada para pruebas):

```powershell
python plot_anomalies.py data/ais-data-sample.csv --suffix demo_sample
```

Ruta por defecto (si no pasas `csv_path`, usa `data/ais-data.csv`):

```powershell
python plot_anomalies.py --suffix h3_full
```

Forzar pipeline completo de prediccion (mas lento):

```powershell
python plot_anomalies.py data/ais-data-sample.csv --suffix full_mode --full
```

Salida principal:

- `plots/anomalies_scatter_<suffix>.html`

## Modelo actual

El modelo usa `IsolationForest` con:

- `n_estimators=100`
- `max_samples=1024`
- `contamination=0.01`
- `random_state=42`

Features actuales (23):

- Temporales: `hour_sin`, `hour_cos`, `day_of_week`, `month`
- Dinamicas: `sog`, `cog`, `heading`, `status`
- Estaticas: `vessel_type`, `length`, `width`, `draft`
- Contexto H3:
  - `hex_log_density`, `is_sparse_hex`, `is_new_hex`
  - `sog_delta_hex_med`, `sog_z_hex`
  - `cog_delta_sin_hex`, `cog_delta_cos_hex`
  - `heading_delta_sin_hex`, `heading_delta_cos_hex`
  - `vtype_mode_share_hex`, `is_unusual_vtype_hex`

Nota: `cargo` ya no se usa como feature del modelo.

## Salidas de inferencia

Columnas anadidas por inferencia:

- `is_anomaly` (`-1` anomalo, `1` normal)
- `anomaly_score` (mas negativo = mas anomalo)
- `anomaly_reason` (via SHAP, con fallback)

## Consejos practicos

- Para entrenar, usa `data/ais-data.csv`.
- Para pruebas de inferencia y graficos, usa `data/ais-data-sample.csv`.
- Si cambias el esquema de features, reentrena antes de inferir.
