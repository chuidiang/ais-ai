"""
train_anomaly.py
----------------
Entrena un modelo Isolation Forest sobre datos AIS para detectar
comportamientos anómalos de embarcaciones.

Features:
  Temporales         : hour_sin, hour_cos (cíclicas), day_of_week, month
  Dinámicas          : sog, cog, heading, status
  Estáticas del barco: vessel_type, length, width, draft
  Contexto H3        : hex_log_density, is_sparse_hex, is_new_hex,
                       sog_delta_hex_med, sog_z_hex,
                       cog_delta_sin_hex, cog_delta_cos_hex,
                       heading_delta_sin_hex, heading_delta_cos_hex,
                       vtype_mode_share_hex, is_unusual_vtype_hex

Salida:
  - models/isolation_forest_model.joblib
  - models/scaler.joblib
  - models/imputer.joblib
  - models/h3_stats.joblib          ← estadísticas por celda H3 res=7
  - models/h3_parent_stats.joblib   ← estadísticas por celda H3 res=5
  - models/h3_config.json           ← resoluciones, umbrales y stats globales
  - models/metadata.json
  - data/anomalies_summary.csv
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

from load_ais_data import (
    preprocess, add_h3_cells, enrich_with_h3_context,
    H3_RESOLUTION, H3_PARENT_RESOLUTION, H3_MIN_OBS, H3_VTYPE_MIN_SHARE,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH          = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
SCALER_PATH         = os.path.join(MODELS_DIR, "scaler.joblib")
IMPUTER_PATH        = os.path.join(MODELS_DIR, "imputer.joblib")
H3_STATS_PATH       = os.path.join(MODELS_DIR, "h3_stats.joblib")
H3_PARENT_STATS_PATH= os.path.join(MODELS_DIR, "h3_parent_stats.joblib")
H3_CONFIG_PATH      = os.path.join(MODELS_DIR, "h3_config.json")
METADATA_PATH       = os.path.join(MODELS_DIR, "metadata.json")

ANOMALIES_CSV = os.path.join(DATA_DIR, "anomalies_summary.csv")

CONTAMINATION = 0.01
N_ESTIMATORS  = 100
MAX_SAMPLES   = 1024
RANDOM_STATE  = 42

HEADING_NO_DISP = 511

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

TEMPORAL_FEATS   = ["hour_sin", "hour_cos", "day_of_week", "month"]
DYNAMIC_FEATS    = ["sog", "cog", "heading", "status"]
STATIC_FEATS     = ["vessel_type", "length", "width", "draft"]
H3_CONTEXT_FEATS = [
    "hex_log_density",
    "is_sparse_hex",
    "is_new_hex",
    "sog_delta_hex_med",
    "sog_z_hex",
    "cog_delta_sin_hex",
    "cog_delta_cos_hex",
    "heading_delta_sin_hex",
    "heading_delta_cos_hex",
    "vtype_mode_share_hex",
    "is_unusual_vtype_hex",
]

FEATURE_COLS = TEMPORAL_FEATS + DYNAMIC_FEATS + STATIC_FEATS + H3_CONTEXT_FEATS


# ---------------------------------------------------------------------------
# 1. Estadísticas H3
# ---------------------------------------------------------------------------

def build_h3_stats(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Calcula estadísticas históricas por celda H3 (res=7 y res=5).

    Para cog y heading usa sin/cos para respetar la circularidad angular.

    Devuelve
    --------
    h3_stats        : stats por h3_res7
    h3_parent_stats : stats por h3_res5
    global_stats    : dict con valores globales (fallback de último nivel)
    """
    df_tmp = df.copy()
    heading_clean = df_tmp["heading"].replace(HEADING_NO_DISP, np.nan)

    cog_rad     = np.deg2rad(df_tmp["cog"].fillna(0.0))
    heading_rad = np.deg2rad(heading_clean.fillna(0.0))

    df_tmp["_cog_sin"]     = np.sin(cog_rad)
    df_tmp["_cog_cos"]     = np.cos(cog_rad)
    df_tmp["_heading_sin"] = np.sin(heading_rad)
    df_tmp["_heading_cos"] = np.cos(heading_rad)

    agg_spec = {
        "obs_count":        ("sog",          "count"),
        "sog_median":       ("sog",          "median"),
        "sog_std":          ("sog",          "std"),
        "cog_sin_mean":     ("_cog_sin",     "mean"),
        "cog_cos_mean":     ("_cog_cos",     "mean"),
        "heading_sin_mean": ("_heading_sin", "mean"),
        "heading_cos_mean": ("_heading_cos", "mean"),
    }

    def _vtype_mode_by_group(group_col: str) -> pd.DataFrame:
        valid = df_tmp.dropna(subset=["vessel_type"])
        if valid.empty:
            return pd.DataFrame(columns=[group_col, "mode_vessel_type", "mode_vessel_share"])

        counts = (
            valid.groupby([group_col, "vessel_type"]).size()
            .rename("vtype_count")
            .reset_index()
        )
        totals = (
            valid.groupby(group_col).size()
            .rename("vtype_total")
            .reset_index()
        )
        mode = counts.merge(totals, on=group_col, how="left")
        mode["mode_vessel_share"] = mode["vtype_count"] / mode["vtype_total"].clip(lower=1)
        mode = mode.sort_values([group_col, "vtype_count", "vessel_type"], ascending=[True, False, True])
        mode = mode.drop_duplicates(subset=[group_col], keep="first")
        mode = mode.rename(columns={"vessel_type": "mode_vessel_type"})
        return mode[[group_col, "mode_vessel_type", "mode_vessel_share"]]

    h3_stats        = df_tmp.groupby("h3_res7").agg(**agg_spec).reset_index()
    h3_parent_stats = df_tmp.groupby("h3_res5").agg(**agg_spec).reset_index()

    h3_stats = h3_stats.merge(_vtype_mode_by_group("h3_res7"), on="h3_res7", how="left")
    h3_parent_stats = h3_parent_stats.merge(_vtype_mode_by_group("h3_res5"), on="h3_res5", how="left")

    # Reemplazar NaN en sog_std (celdas con 1 sola observación)
    h3_stats["sog_std"]        = h3_stats["sog_std"].fillna(0.0)
    h3_parent_stats["sog_std"] = h3_parent_stats["sog_std"].fillna(0.0)
    h3_stats["mode_vessel_share"] = h3_stats["mode_vessel_share"].fillna(0.0)
    h3_parent_stats["mode_vessel_share"] = h3_parent_stats["mode_vessel_share"].fillna(0.0)

    valid_vtype = df_tmp["vessel_type"].dropna()
    if valid_vtype.empty:
        global_mode_vessel_type = None
        global_mode_vessel_share = 0.0
    else:
        vc = valid_vtype.value_counts(dropna=True)
        global_mode_vessel_type = float(vc.index[0])
        global_mode_vessel_share = float(vc.iloc[0] / vc.sum())

    global_stats = {
        "obs_count":        float(len(df_tmp)),
        "sog_median":       float(df_tmp["sog"].median()),
        "sog_std":          float(df_tmp["sog"].std()),
        "cog_sin_mean":     float(df_tmp["_cog_sin"].mean()),
        "cog_cos_mean":     float(df_tmp["_cog_cos"].mean()),
        "heading_sin_mean": float(df_tmp["_heading_sin"].mean()),
        "heading_cos_mean": float(df_tmp["_heading_cos"].mean()),
        "mode_vessel_type": global_mode_vessel_type,
        "mode_vessel_share": global_mode_vessel_share,
    }

    print(
        f"[INFO] H3 stats calculadas:\n"
        f"       res=7: {len(h3_stats):,} celdas  "
        f"| res=5: {len(h3_parent_stats):,} celdas\n"
        f"       global sog_median={global_stats['sog_median']:.2f}  "
        f"sog_std={global_stats['sog_std']:.2f}  "
        f"mode_vessel_type={global_stats['mode_vessel_type']}"
    )
    return h3_stats, h3_parent_stats, global_stats


# ---------------------------------------------------------------------------
# 2. Preparación de la matriz de features
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """
    Extrae y preprocesa las features para el modelo:
      1. Sustituye heading=511 por NaN.
      2. Imputa NaN con la mediana de cada columna.
      3. Escala con StandardScaler.
    Devuelve: (X_scaled, imputer, scaler)
    """
    X = df[FEATURE_COLS].copy()
    X["heading"] = X["heading"].replace(HEADING_NO_DISP, np.nan)

    print(f"[INFO] Nulos por feature antes de imputar:\n{X.isnull().sum().to_string()}\n")

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_imp    = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, imputer, scaler


# ---------------------------------------------------------------------------
# 3. Entrenamiento
# ---------------------------------------------------------------------------

def train_model(X: np.ndarray) -> IsolationForest:
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
# 4. Predicción y etiquetado
# ---------------------------------------------------------------------------

def predict_and_label(
    df: pd.DataFrame,
    model: IsolationForest,
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    df = df.copy()
    df["is_anomaly"]    = model.predict(X_scaled).astype(np.int8)
    df["anomaly_score"] = model.decision_function(X_scaled).astype(np.float32)

    n_total   = len(df)
    n_anomaly = (df["is_anomaly"] == -1).sum()
    pct       = n_anomaly / n_total * 100

    print(f"[INFO] Total registros : {n_total:,}")
    print(f"[INFO] Anomalías (-1)  : {n_anomaly:,}  ({pct:.2f} %)")
    print(f"[INFO] Normales  ( 1)  : {n_total - n_anomaly:,}  ({100-pct:.2f} %)\n")
    return df


# ---------------------------------------------------------------------------
# 5. Resumen de anomalías
# ---------------------------------------------------------------------------

REPORT_COLS = [
    "mmsi", "vessel_name", "base_date_time",
    "latitude", "longitude",
    "h3_res7",
    "sog", "cog", "heading",
    "vessel_type", "status", "length", "width", "draft",
    "hour", "day_of_week",
    "hex_log_density", "is_sparse_hex", "is_new_hex",
    "sog_delta_hex_med", "sog_z_hex",
    "anomaly_score",
]

def save_anomaly_report(df: pd.DataFrame, path: str) -> None:
    cols = [c for c in REPORT_COLS if c in df.columns]
    anomalies = (
        df[df["is_anomaly"] == -1][cols]
        .sort_values("anomaly_score")
    )
    anomalies.to_csv(path, index=False)
    print(f"[INFO] Resumen guardado en '{path}'  ({len(anomalies):,} filas)\n")
    print("[INFO] Top 10 registros más anómalos:")
    print(anomalies.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    # 1. Cargar y preprocesar
    df = preprocess()

    # 2. Asignar celdas H3
    df = add_h3_cells(df)

    # 3. Construir estadísticas H3 (desde datos de entrenamiento)
    h3_stats, h3_parent_stats, global_stats = build_h3_stats(df)

    # 4. Enriquecer con contexto H3
    df = enrich_with_h3_context(df, h3_stats, h3_parent_stats, global_stats)

    # 5. Preparar matriz de features
    X_scaled, imputer, scaler = prepare_features(df)

    # 6. Entrenar
    model = train_model(X_scaled)

    # 7. Etiquetar
    df = predict_and_label(df, model, X_scaled)

    # 8. Guardar artefactos
    save_artifacts(model, scaler, imputer, h3_stats, h3_parent_stats, global_stats)

    # 9. Resumen de anomalías
    save_anomaly_report(df, ANOMALIES_CSV)

    return df


# ---------------------------------------------------------------------------
# Exportación e importación de artefactos
# ---------------------------------------------------------------------------

def save_artifacts(
    model:           IsolationForest,
    scaler:          StandardScaler,
    imputer:         SimpleImputer,
    h3_stats:        pd.DataFrame,
    h3_parent_stats: pd.DataFrame,
    global_stats:    dict,
    model_path:            str = MODEL_PATH,
    scaler_path:           str = SCALER_PATH,
    imputer_path:          str = IMPUTER_PATH,
    h3_stats_path:         str = H3_STATS_PATH,
    h3_parent_stats_path:  str = H3_PARENT_STATS_PATH,
    h3_config_path:        str = H3_CONFIG_PATH,
    metadata_path:         str = METADATA_PATH,
) -> None:
    """
    Exporta todos los artefactos del pipeline.

    Ficheros generados
    ------------------
    isolation_forest_model.joblib
    scaler.joblib
    imputer.joblib
    h3_stats.joblib          ← estadísticas históricas por celda H3 res=7
    h3_parent_stats.joblib   ← estadísticas por celda H3 res=5 (fallback)
    h3_config.json           ← resoluciones H3, umbrales y stats globales
    metadata.json
    """
    joblib.dump(model,           model_path)
    joblib.dump(scaler,          scaler_path)
    joblib.dump(imputer,         imputer_path)
    joblib.dump(h3_stats,        h3_stats_path)
    joblib.dump(h3_parent_stats, h3_parent_stats_path)

    h3_config = {
        "h3_resolution":        H3_RESOLUTION,
        "parent_resolution":    H3_PARENT_RESOLUTION,
        "min_obs_hex":          H3_MIN_OBS,
        "vtype_min_share_hex":  H3_VTYPE_MIN_SHARE,
        "global_stats":         global_stats,
    }
    with open(h3_config_path, "w", encoding="utf-8") as f:
        json.dump(h3_config, f, indent=2)

    metadata = {
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "feature_cols":   FEATURE_COLS,
        "heading_no_disp": HEADING_NO_DISP,
        "hyperparameters": {
            "n_estimators" : N_ESTIMATORS,
            "max_samples"  : MAX_SAMPLES,
            "contamination": CONTAMINATION,
            "random_state" : RANDOM_STATE,
        },
        "artifacts": {
            "model":           os.path.basename(model_path),
            "scaler":          os.path.basename(scaler_path),
            "imputer":         os.path.basename(imputer_path),
            "h3_stats":        os.path.basename(h3_stats_path),
            "h3_parent_stats": os.path.basename(h3_parent_stats_path),
            "h3_config":       os.path.basename(h3_config_path),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[INFO] Artefactos exportados en '{MODELS_DIR}':\n"
        f"       · {os.path.basename(model_path)}\n"
        f"       · {os.path.basename(scaler_path)}\n"
        f"       · {os.path.basename(imputer_path)}\n"
        f"       · {os.path.basename(h3_stats_path)}\n"
        f"       · {os.path.basename(h3_parent_stats_path)}\n"
        f"       · {os.path.basename(h3_config_path)}\n"
        f"       · {os.path.basename(metadata_path)}\n"
    )


def load_artifacts(
    model_path:           str = MODEL_PATH,
    scaler_path:          str = SCALER_PATH,
    imputer_path:         str = IMPUTER_PATH,
    h3_stats_path:        str = H3_STATS_PATH,
    h3_parent_stats_path: str = H3_PARENT_STATS_PATH,
    h3_config_path:       str = H3_CONFIG_PATH,
    metadata_path:        str = METADATA_PATH,
) -> tuple[IsolationForest, StandardScaler, SimpleImputer,
           pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Carga todos los artefactos del pipeline.

    Devuelve
    --------
    (model, scaler, imputer, h3_stats, h3_parent_stats, h3_config, metadata)
    """
    model           = joblib.load(model_path)
    scaler          = joblib.load(scaler_path)
    imputer         = joblib.load(imputer_path)
    h3_stats        = joblib.load(h3_stats_path)
    h3_parent_stats = joblib.load(h3_parent_stats_path)

    with open(h3_config_path, "r", encoding="utf-8") as f:
        h3_config = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(
        f"[INFO] Artefactos cargados desde '{MODELS_DIR}' "
        f"(entrenado: {metadata.get('trained_at', 'desconocido')})\n"
        f"       H3 res={h3_config['h3_resolution']}  "
        f"parent_res={h3_config['parent_resolution']}  "
        f"min_obs={h3_config['min_obs_hex']}"
    )
    return model, scaler, imputer, h3_stats, h3_parent_stats, h3_config, metadata


# ---------------------------------------------------------------------------
# Inferencia sobre nuevos datos
# ---------------------------------------------------------------------------

def predict_new(
    df_new: pd.DataFrame,
    model_path:           str = MODEL_PATH,
    scaler_path:          str = SCALER_PATH,
    imputer_path:         str = IMPUTER_PATH,
    h3_stats_path:        str = H3_STATS_PATH,
    h3_parent_stats_path: str = H3_PARENT_STATS_PATH,
    h3_config_path:       str = H3_CONFIG_PATH,
    metadata_path:        str = METADATA_PATH,
) -> pd.DataFrame:
    """
    Aplica el pipeline exportado a un DataFrame nuevo.
    df_new debe contener al menos: latitude, longitude, sog, cog,
    heading, hour_sin, hour_cos, day_of_week, month, status,
    vessel_type, length, width, draft.
    """
    model, scaler, imputer, h3_stats, h3_parent_stats, h3_config, meta = load_artifacts(
        model_path, scaler_path, imputer_path,
        h3_stats_path, h3_parent_stats_path, h3_config_path, metadata_path
    )
    features        = meta["feature_cols"]
    heading_no_disp = meta.get("heading_no_disp", 511)
    global_stats    = h3_config["global_stats"]
    min_obs         = h3_config["min_obs_hex"]

    df_base = df_new.copy()

    # Asignar H3 si no están ya calculados
    if "h3_res7" not in df_base.columns:
        df_base = add_h3_cells(df_base,
                               resolution=h3_config["h3_resolution"],
                               parent_resolution=h3_config["parent_resolution"])

    # Enriquecer con contexto H3
    if "hex_log_density" not in df_base.columns:
        df_base = enrich_with_h3_context(
            df_base, h3_stats, h3_parent_stats, global_stats, min_obs=min_obs
        )

    X = df_base[features].copy()
    X["heading"] = X["heading"].replace(heading_no_disp, np.nan)

    X_imp    = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    df_out = df_base.copy()
    df_out["is_anomaly"]    = model.predict(X_scaled).astype(np.int8)
    df_out["anomaly_score"] = model.decision_function(X_scaled).astype(np.float32)
    return df_out


if __name__ == "__main__":
    df_result = main()

