# AIS Anomaly Detection

Pipeline para deteccion de anomalias AIS con Isolation Forest, explicaciones SHAP y visualizacion geografica.

## 0) Requisitos tecnicos

Para usar el proyecto desde cero, **si necesitas tener Python instalado**.

Version de Python recomendada y compatible con las librerias de `requirements.txt`:

- **Minima para todas las dependencias actuales**: `Python >= 3.11`
- **Probada en este repositorio**: `Python 3.14.3`
- Recomendacion practica: usar `Python 3.14.x` para reproducir el entorno de este repo.

Comprobacion rapida en **PowerShell (Windows)**:

```powershell
python --version
```

## 1) Clonar el repositorio

Comandos de **PowerShell (Windows)**:

```powershell
git clone <URL_DEL_REPOSITORIO>
cd ais-ai
```

## 2) Crear y activar entorno virtual

Comandos de **PowerShell (Windows)**:

```powershell
python -m venv .venv
.\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nota: si en tu copia local ya existe un entorno virtual en la raiz del repo,
tambien puedes activar `Scripts/Activate.ps1`.

Scripts de activacion incluidos en `Scripts/` para distintos shells:

- `activate` (sh/bash)
- `activate.fish`
- `activate.nu`
- `activate.bat` (cmd)
- `Activate.ps1` / `activate.ps1` (PowerShell)

## 3) Preparar datos

### Fuente de datos AIS

Los datos AIS de entrenamiento y pruebas se han descargado de:
https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2025/index.html

---

El repositorio incluye un ejemplo pequeño para pruebas rápidas y como ejemplo de formato esperado:

- `data/ais-data-sample.csv`

Para ejecutar el flujo completo con tus datos, coloca el fichero AIS para entrenamiento en:

- `data/ais-data.csv`

Importante:

- `ais-data.csv` debe tener el mismo formato de columnas que `data/ais-data-sample.csv`.

Los archivos del shapefile del mapa mundial están incluidos en:

- `shp/world.shp`
- `shp/world.shx`
- `shp/world.dbf`

Estos ficheros se utilizan para generar el mapa base en los gráficos de anomalías.

## 4) Entrenar y exportar modelo

Si has colocado tu dataset en `data/ais-data.csv`, ejecuta el entrenamiento.

Comando de **PowerShell (Windows)**:

```powershell
python train_anomaly.py
```
En `models/` se guardan los datos del modelo entrenado:

- `isolation_forest_model.joblib`
- `scaler.joblib`
- `imputer.joblib`
- `metadata.json`

Una vez entrenado el modelo y creados, por tanto, esos ficheros, puedes probarlo con el comando siguiente.

Comando de **PowerShell (Windows)**:

```powershell
python plot_anomalies.py data/ais-data-sample.csv --suffix demo_sample
```
donde 
- ais-data-sample.csv es el dataset con los datos en los que quieres detectar anomalías
- --suffix es el sufijo que se añadirá a los nombres de los archivos de salida para diferenciarlos.


## 5) Generar graficos de anomalias

Comando de **PowerShell (Windows)**:

```powershell
python plot_anomalies.py --suffix con_shap_reason
```

Salida:

- `plots/anomalies_scatter_<suffix>.html`

Ejemplo de salida:

![Ejemplo de salida de anomalías](./output-sample.png)

El tooltip del grafico interactivo muestra:

- `anomaly_reason` (`Motivo`)
- `anomaly_score`
- todas las features usadas por el modelo

El contorno de `shp/world.shp` se dibuja como capa base del mapa mundial.

## 6) Modo tecnico: experimentos para desarrolladores

Este apartado resume la interfaz real de los scripts para que puedas experimentar con parametros y datasets sin adivinar opciones.

### `plot_anomalies.py`

`plot_anomalies.py` acepta:

- `csv_path` (posicional, opcional): ruta del CSV AIS de entrada.
- `--suffix` (opcional): sufijo para el HTML de salida.

Si **no** pasas `csv_path`, usa el valor por defecto definido en `load_ais_data.py` (`CSV_FILE`), que apunta a `data/ais-data.csv`.

Con fichero explicito (PowerShell):

```powershell
python plot_anomalies.py data/ais-data-sample.csv --suffix demo_sample
```

Sin fichero explicito (usa `data/ais-data.csv` por defecto):

```powershell
python plot_anomalies.py --suffix experimento_default
```

Sin `--suffix`, el script usa timestamp (`YYYYMMDD_HHMMSS`) para versionar la salida automaticamente.

```powershell
python plot_anomalies.py
```

Salida esperada: `plots/anomalies_scatter_<suffix>.html` (o `plots/anomalies_scatter.html` si no hay sufijo).

### `train_anomaly.py`

`train_anomaly.py` **no expone argumentos CLI** ahora mismo. Se ejecuta asi:

```powershell
python train_anomaly.py
```

Para experimentar, ajusta constantes en `train_anomaly.py` y vuelve a lanzar:

- `CONTAMINATION`: porcentaje esperado de anomalias.
- `N_ESTIMATORS`: numero de arboles del Isolation Forest.
- `MAX_SAMPLES`: muestras por arbol.
- `RANDOM_STATE`: semilla para reproducibilidad.
- `HEADING_NO_DISP`: valor AIS para heading no disponible (`511`).

Tambien puedes experimentar con el preprocesado desde `load_ais_data.py`:

- `GRID_SIZE_DEG`: tamano de celda geografica.
- rangos validos (`LAT_MIN/LAT_MAX`, `LON_MIN/LON_MAX`, `SOG_MIN/SOG_MAX`, `COG_MIN/COG_MAX`).

Flujo tipico de experimento (PowerShell):

```powershell
python train_anomaly.py
python plot_anomalies.py --suffix exp_cont_001
```

Consejo practico: cambia un parametro cada vez (por ejemplo `CONTAMINATION` o `GRID_SIZE_DEG`) y compara el HTML resultante y `data/anomalies_summary.csv` entre ejecuciones.

---

## Autor

Hecho por **Copilot** bajo la experta (y paciente) dirección de **Chuidiang**. 🤖✨
