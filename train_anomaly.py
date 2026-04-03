"""
train_anomaly.py
----------------
Entrena un modelo Isolation Forest sobre datos AIS para detectar
comportamientos anómalos de embarcaciones.

Features:
  Espaciales (norm.) : lat_norm, lon_norm
  Cuadrícula (zona)  : grid_x, grid_y
  Temporales         : hour_sin, hour_cos (codificacion ciclica), day_of_week, month
  Dinámicas          : sog, cog, heading, status
  Estáticas del barco: vessel_type, length, width, draft, cargo

Salida:
  - Columna 'is_anomaly'    en el DataFrame  (-1 anomalía / 1 normal)
  - Columna 'anomaly_score' en el DataFrame  (más negativo → más anómalo)
  - models/isolation_forest_model.joblib  ← modelo IsolationForest
  - models/scaler.joblib                  ← StandardScaler ajustado
  - models/imputer.joblib                 ← SimpleImputer ajustado
  - models/metadata.json                  ← features, parámetros, versión
  - data/anomalies_summary.csv            ← resumen de anomalías detectadas
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from load_ais_data import preprocess

# ---------------------------------------------------------------------------
# Configuración  (todos los parámetros ajustables aquí)
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

# Rutas de exportación individuales  (compatibles con cualquier aplicación)
MODEL_PATH    = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.joblib")
IMPUTER_PATH  = os.path.join(MODELS_DIR, "imputer.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

ANOMALIES_CSV = os.path.join(DATA_DIR, "anomalies_summary.csv")

CONTAMINATION = 0.01   # fracción de puntos anómalos esperada (ajustable)
N_ESTIMATORS  = 100    # número de árboles
MAX_SAMPLES   = 1024   # muestras por árbol (↑ = más preciso, ↑ = más lento)
RANDOM_STATE  = 42

# Valor AIS que indica "heading no disponible"
HEADING_NO_DISP = 511

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Espaciales (normalizadas)
SPATIAL_FEATS   = ["lat_norm", "lon_norm"]
# Cuadrícula (zona geográfica)
GRID_FEATS      = ["grid_x", "grid_y"]
# Temporales
# Temporales: hour_sin/hour_cos en lugar de hour entero para capturar
# la continuidad 23 h → 0 h; day_of_week y month permanecen como enteros
TEMPORAL_FEATS  = ["hour_sin", "hour_cos", "day_of_week", "month"]
# Dinámicas: comportamiento en tiempo real
DYNAMIC_FEATS   = ["sog", "cog", "heading", "status"]
# Estáticas: características propias del barco
STATIC_FEATS    = ["vessel_type", "length", "width", "draft", "cargo"]

FEATURE_COLS = (
    SPATIAL_FEATS +
    GRID_FEATS    +
    TEMPORAL_FEATS+
    DYNAMIC_FEATS +
    STATIC_FEATS
)


# ---------------------------------------------------------------------------
# 1. Preparación de la matriz de features
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """
    Extrae y preprocesa las features para el modelo:
      1. Sustituye heading=511 (no disponible en AIS) por NaN.
      2. Imputa NaN con la mediana de cada columna.
      3. Escala con StandardScaler para que cada feature contribuya por igual.
    Devuelve: (X_scaled, imputer ajustado, scaler ajustado)
    """
    X = df[FEATURE_COLS].copy()

    # heading 511 → NaN
    X["heading"] = X["heading"].replace(HEADING_NO_DISP, np.nan)

    print(f"[INFO] Nulos por feature antes de imputar:\n{X.isnull().sum().to_string()}\n")

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_imp    = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, imputer, scaler


# ---------------------------------------------------------------------------
# 2. Entrenamiento del Isolation Forest
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray) -> IsolationForest:
    """
    Entrena el Isolation Forest.
    - contamination  : fracción de anomalías esperada.
    - max_samples    : puntos por árbol (equilibrio precisión / velocidad).
    - n_jobs=-1      : usa todos los núcleos disponibles.
    """
    print(
        f"[INFO] Entrenando Isolation Forest …\n"
        f"       filas={len(X):,}  features={X.shape[1]}"
        f"  n_estimators={N_ESTIMATORS}  max_samples={MAX_SAMPLES}"
        f"  contamination={CONTAMINATION}"
    )
    model = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        max_samples   = MAX_SAMPLES,
        contamination = CONTAMINATION,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    model.fit(X)
    print("[INFO] Entrenamiento completado.\n")
    return model


# ---------------------------------------------------------------------------
# 3. Predicción y etiquetado
# ---------------------------------------------------------------------------

def predict_and_label(
    df: pd.DataFrame,
    model: IsolationForest,
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    """
    Añade al DataFrame:
      'is_anomaly'    : -1 (anómalo) / 1 (normal)
      'anomaly_score' : puntuación continua; más negativo → más anómalo
    """
    df = df.copy()
    df["is_anomaly"]    = model.predict(X_scaled).astype(np.int8)
    df["anomaly_score"] = model.decision_function(X_scaled).astype(np.float32)

    n_total    = len(df)
    n_anomaly  = (df["is_anomaly"] == -1).sum()
    pct        = n_anomaly / n_total * 100

    print(f"[INFO] Total registros : {n_total:,}")
    print(f"[INFO] Anomalías (-1)  : {n_anomaly:,}  ({pct:.2f} %)")
    print(f"[INFO] Normales  ( 1)  : {n_total - n_anomaly:,}  ({100 - pct:.2f} %)\n")

    return df


# ---------------------------------------------------------------------------
# 4. Resumen de anomalías
# ---------------------------------------------------------------------------

REPORT_COLS = [
    "mmsi", "vessel_name", "base_date_time",
    "latitude", "longitude",
    "sog", "cog", "heading",
    "vessel_type", "status", "length", "width", "draft",
    "hour", "day_of_week",
    "grid_x", "grid_y",
    "anomaly_score",
]

def save_anomaly_report(df: pd.DataFrame, path: str) -> None:
    """Guarda las filas anómalas ordenadas por puntuación en un CSV."""
    anomalies = (
        df[df["is_anomaly"] == -1][REPORT_COLS]
        .sort_values("anomaly_score")          # más anómalos primero
    )
    anomalies.to_csv(path, index=False)
    print(f"[INFO] Resumen de anomalías guardado en '{path}'  ({len(anomalies):,} filas)\n")

    print("[INFO] Top 10 registros más anómalos:")
    print(anomalies.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    # 1. Cargar y preprocesar datos
    df = preprocess()
    print()

    # 2. Construir matriz de features
    X_scaled, imputer, scaler = prepare_features(df)

    # 3. Entrenar modelo
    model = train_model(X_scaled)

    # 4. Etiquetar anomalías
    df = predict_and_label(df, model, X_scaled)

    # 5. Exportar artefactos por separado
    save_artifacts(model, scaler, imputer)

    # 6. Guardar y mostrar resumen de anomalías
    save_anomaly_report(df, ANOMALIES_CSV)

    return df


# ---------------------------------------------------------------------------
# Exportación e importación de artefactos (joblib)
# ---------------------------------------------------------------------------

def save_artifacts(
    model:   IsolationForest,
    scaler:  StandardScaler,
    imputer: SimpleImputer,
    model_path:   str = MODEL_PATH,
    scaler_path:  str = SCALER_PATH,
    imputer_path: str = IMPUTER_PATH,
    metadata_path: str = METADATA_PATH,
) -> None:
    """
    Exporta cada componente del pipeline como fichero .joblib independiente
    y guarda los metadatos del entrenamiento en un JSON.

    Ficheros generados
    ------------------
    isolation_forest_model.joblib  → modelo IsolationForest serializado
    scaler.joblib                  → StandardScaler ajustado (media/std por feature)
    imputer.joblib                 → SimpleImputer con medianas de entrenamiento
    metadata.json                  → features, hiperparámetros, versiones y timestamp
    """
    joblib.dump(model,   model_path)
    joblib.dump(scaler,  scaler_path)
    joblib.dump(imputer, imputer_path)

    metadata = {
        "trained_at"    : datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "feature_cols"  : FEATURE_COLS,
        "heading_no_disp": HEADING_NO_DISP,
        "hyperparameters": {
            "n_estimators" : N_ESTIMATORS,
            "max_samples"  : MAX_SAMPLES,
            "contamination": CONTAMINATION,
            "random_state" : RANDOM_STATE,
        },
        "artifacts": {
            "model"  : os.path.basename(model_path),
            "scaler" : os.path.basename(scaler_path),
            "imputer": os.path.basename(imputer_path),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[INFO] Artefactos exportados en '{MODELS_DIR}':\n"
        f"       · {os.path.basename(model_path)}\n"
        f"       · {os.path.basename(scaler_path)}\n"
        f"       · {os.path.basename(imputer_path)}\n"
        f"       · {os.path.basename(metadata_path)}\n"
    )


def load_artifacts(
    model_path:    str = MODEL_PATH,
    scaler_path:   str = SCALER_PATH,
    imputer_path:  str = IMPUTER_PATH,
    metadata_path: str = METADATA_PATH,
) -> tuple[IsolationForest, StandardScaler, SimpleImputer, dict]:
    """
    Carga los artefactos exportados desde disco.
    Devuelve: (model, scaler, imputer, metadata)

    Uso típico en otra aplicación
    ------------------------------
    from train_anomaly import load_artifacts, HEADING_NO_DISP
    model, scaler, imputer, meta = load_artifacts()
    features = meta["feature_cols"]
    """
    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(
        f"[INFO] Artefactos cargados desde '{MODELS_DIR}' "
        f"(entrenado: {metadata.get('trained_at', 'desconocido')})"
    )
    return model, scaler, imputer, metadata


# ---------------------------------------------------------------------------
# Función de inferencia reutilizable (para nuevos datos)
# ---------------------------------------------------------------------------

def predict_new(
    df_new: pd.DataFrame,
    model_path:   str = MODEL_PATH,
    scaler_path:  str = SCALER_PATH,
    imputer_path: str = IMPUTER_PATH,
    metadata_path: str = METADATA_PATH,
) -> pd.DataFrame:
    """
    Aplica el pipeline exportado a un DataFrame nuevo sin necesidad de reentrenar.
    Devuelve el DataFrame con las columnas 'is_anomaly' y 'anomaly_score'.

    Parámetros
    ----------
    df_new : DataFrame que ya contiene las columnas de FEATURE_COLS
             (resultado de load_ais_data.preprocess o equivalente).
    """
    model, scaler, imputer, meta = load_artifacts(
        model_path, scaler_path, imputer_path, metadata_path
    )
    features = meta["feature_cols"]
    heading_no_disp = meta.get("heading_no_disp", 511)

    X = df_new[features].copy()
    X["heading"] = X["heading"].replace(heading_no_disp, np.nan)

    X_imp    = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    df_out = df_new.copy()
    df_out["is_anomaly"]    = model.predict(X_scaled).astype(np.int8)
    df_out["anomaly_score"] = model.decision_function(X_scaled).astype(np.float32)
    return df_out


if __name__ == "__main__":
    df_result = main()

