# AIS Anomaly Detection

Pipeline para deteccion de anomalias AIS con Isolation Forest y visualizacion geografica.

## 1) Clonar el repositorio

```powershell
git clone <URL_DEL_REPOSITORIO>
cd ais-ai
```

## 2) Crear y activar entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Preparar datos

El repositorio puede incluir un ejemplo pequeño para pruebas rápidas:

- `data/ais-data-sample.csv`

Para ejecutar el flujo completo con tus datos, coloca el fichero AIS principal en:

- `data/ais-data.csv`

Importante:

- `ais-data.csv` debe tener el mismo formato de columnas que `data/ais-data-sample.csv`.
- El sample sirve como referencia del esquema esperado.

Opcionalmente, coloca el shapefile del mapa mundial en:

- `shp/world.shp`
- `shp/world.shx`
- `shp/world.dbf`

Esos tres son los únicos ficheros del shapefile necesarios para que el código genere el mapa base.

## 4) Entrenar y exportar modelo

```powershell
python train_anomaly.py
```

Dos modos de entrenamiento para `status`:

- `default` (recomendado): imputa `status` faltante como `15`.
- `--discard-missing-status`: descarta filas con `status` nulo.

Para conservar ambos modelos sin sobrescribir artefactos, usa carpetas distintas:

```powershell
python train_anomaly.py --models-dir models\status_imputed --anomalies-csv data\anomalies_status_imputed.csv
python train_anomaly.py --discard-missing-status --models-dir models\status_filtered --anomalies-csv data\anomalies_status_filtered.csv
```

Para una prueba rápida con el dataset de ejemplo puedes generar gráficos directamente:

```powershell
python plot_anomalies.py data/ais-data-sample.csv --suffix demo_sample
```

Artefactos exportados en `models/`:

- `isolation_forest_model.joblib`
- `scaler.joblib`
- `context_model.joblib`
- `metadata.json`

Features actuales del modelo:

- `latitude`
- `longitude`
- `vessel_type_mapped` (transformado)
- `status` (imputado a 15 o filtrado, según modo)
- `sog`
- `cog_sin`, `cog_cos`
- `heading_sin`, `heading_cos`
- `sog_ctx_logprob`
- `cog_ctx_logprob`
- `vessel_type_ctx_logprob`
- `status_ctx_logprob`

Transformacion de `vessel_type`:

- 1x -> 10
- 2x -> 20
- 6x -> 60
- 7x -> 70
- 8x -> 80
- 9x -> 90
- nulo / desconocido / otros -> 0

## 5) Generar graficos de anomalias

```powershell
python plot_anomalies.py --models-dir models\status_imputed --suffix con_shap_reason
```

Salida:

- `plots/anomalies_scatter_<suffix>.html`

El tooltip del grafico interactivo muestra `anomaly_score` con `fiabilidad (%)` relativa,
ademas de `vessel_type`, `vessel_type_mapped`, `status` y `sog`.

Inferencia por lote con el modelo que elijas:

```powershell
python predict_realtime.py data\ais-data-sample.csv --models-dir models\status_imputed
python predict_realtime.py data\ais-data-sample.csv --models-dir models\status_filtered
```

Si existe `shp/world.shp`, se dibuja el contorno del mapa mundial como capa base.

---

## Autor

Hecho por **Copilot** bajo la experta (y paciente) dirección de **Chuidiang**. 🤖✨

