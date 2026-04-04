"""
train_anomaly.py
----------------
Entrena un modelo Isolation Forest sobre datos AIS para detectar
comportamientos anómalos de embarcaciones.

Features:
  Temporales         : hour_sin, hour_cos (cíclicas), day_of_week, month
  Dinámicas          : sog, cog, heading, status
  Estáticas del barco: vessel_type, length, width, draft
  Contexto local     : H3 + vessel_type + hora/franja con fallback jerárquico
                       (densidad, residuals de sog/cog/heading/size,
                       rareza del tipo de barco)

Salida:
  - models/isolation_forest_model.joblib
  - models/scaler.joblib
  - models/imputer.joblib
  - models/h3_stats.joblib          ← bundle de estadísticas contextuales
  - models/h3_parent_stats.joblib   ← tabla padre resumida (compatibilidad)
  - models/h3_config.json           ← resoluciones, umbrales y configuración temporal
  - models/metadata.json
  - data/anomalies_summary.csv
"""

import argparse
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
    CSV_FILE,
    CONTEXT_HOUR_MODE,
    H3_CONTEXT_MIN_OBS,
    H3_MIN_OBS,
    H3_PARENT_RESOLUTION,
    H3_RESOLUTION,
    H3_VTYPE_MIN_SHARE,
    HOUR_BUCKET_SIZE,
    NUMERIC_CONTEXT_LEVELS,
    VTYPE_CONTEXT_LEVELS,
    add_h3_cells,
    enrich_with_h3_context,
    preprocess,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH           = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
SCALER_PATH          = os.path.join(MODELS_DIR, "scaler.joblib")
IMPUTER_PATH         = os.path.join(MODELS_DIR, "imputer.joblib")
H3_STATS_PATH        = os.path.join(MODELS_DIR, "h3_stats.joblib")
H3_PARENT_STATS_PATH = os.path.join(MODELS_DIR, "h3_parent_stats.joblib")
H3_CONFIG_PATH       = os.path.join(MODELS_DIR, "h3_config.json")
METADATA_PATH        = os.path.join(MODELS_DIR, "metadata.json")

ANOMALIES_CSV = os.path.join(DATA_DIR, "anomalies_summary.csv")

CONTAMINATION = 0.01
N_ESTIMATORS  = 100
MAX_SAMPLES   = 1024
RANDOM_STATE  = 42

HEADING_NO_DISP = 511

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

TEMPORAL_FEATS = ["hour_sin", "hour_cos", "day_of_week", "month"]
DYNAMIC_FEATS  = ["sog", "cog", "heading", "status"]
STATIC_FEATS   = ["vessel_type", "length", "width", "draft"]
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
    "length_delta_hex_med",
    "length_z_hex",
    "width_delta_hex_med",
    "width_z_hex",
    "draft_delta_hex_med",
    "draft_z_hex",
    "vtype_mode_share_hex",
    "is_unusual_vtype_hex",
]
FEATURE_COLS = TEMPORAL_FEATS + DYNAMIC_FEATS + STATIC_FEATS + H3_CONTEXT_FEATS

_NUMERIC_STD_COLS = ["sog_std", "length_std", "width_std", "draft_std"]
_NUMERIC_CONTEXT_STAT_COLS = [
    "obs_count",
    "sog_median", "sog_std",
    "cog_sin_mean", "cog_cos_mean",
    "heading_sin_mean", "heading_cos_mean",
    "length_median", "length_std",
    "width_median", "width_std",
    "draft_median", "draft_std",
]


# ---------------------------------------------------------------------------
# Utilidades de paths
# ---------------------------------------------------------------------------

def build_artifact_paths(models_dir: str = MODELS_DIR) -> dict[str, str]:
    os.makedirs(models_dir, exist_ok=True)
    return {
        "model": os.path.join(models_dir, "isolation_forest_model.joblib"),
        "scaler": os.path.join(models_dir, "scaler.joblib"),
        "imputer": os.path.join(models_dir, "imputer.joblib"),
        "h3_stats": os.path.join(models_dir, "h3_stats.joblib"),
        "h3_parent_stats": os.path.join(models_dir, "h3_parent_stats.joblib"),
        "h3_config": os.path.join(models_dir, "h3_config.json"),
        "metadata": os.path.join(models_dir, "metadata.json"),
    }


# ---------------------------------------------------------------------------
# 1. Estadísticas contextuales H3 + tipo + franja
# ---------------------------------------------------------------------------

def _prepare_context_frame(df: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.copy()
    heading_clean = df_tmp["heading"].replace(HEADING_NO_DISP, np.nan)

    cog_rad = np.deg2rad(df_tmp["cog"].fillna(0.0))
    heading_rad = np.deg2rad(heading_clean.fillna(0.0))

    df_tmp["_cog_sin"] = np.sin(cog_rad)
    df_tmp["_cog_cos"] = np.cos(cog_rad)
    df_tmp["_heading_sin"] = np.sin(heading_rad)
    df_tmp["_heading_cos"] = np.cos(heading_rad)
    return df_tmp


def _build_numeric_group_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + _NUMERIC_CONTEXT_STAT_COLS)

    grouped = df.groupby(group_cols)
    stats = grouped.agg(
        sog_median=("sog", "median"),
        sog_std=("sog", "std"),
        cog_sin_mean=("_cog_sin", "mean"),
        cog_cos_mean=("_cog_cos", "mean"),
        heading_sin_mean=("_heading_sin", "mean"),
        heading_cos_mean=("_heading_cos", "mean"),
        length_median=("length", "median"),
        length_std=("length", "std"),
        width_median=("width", "median"),
        width_std=("width", "std"),
        draft_median=("draft", "median"),
        draft_std=("draft", "std"),
    ).reset_index()
    counts = grouped.size().rename("obs_count").reset_index()
    out = counts.merge(stats, on=group_cols, how="left")
    for col in _NUMERIC_STD_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _build_vtype_mode_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    valid = df.dropna(subset=["vessel_type"])
    if valid.empty:
        return pd.DataFrame(columns=group_cols + ["obs_count", "mode_vessel_type", "mode_vessel_share"])

    counts = (
        valid.groupby(group_cols + ["vessel_type"]).size()
        .rename("vtype_count")
        .reset_index()
    )
    totals = (
        valid.groupby(group_cols).size()
        .rename("obs_count")
        .reset_index()
    )
    mode = counts.merge(totals, on=group_cols, how="left")
    mode["mode_vessel_share"] = mode["vtype_count"] / mode["obs_count"].clip(lower=1)
    mode = mode.sort_values(group_cols + ["vtype_count", "vessel_type"], ascending=[True] * len(group_cols) + [False, True])
    mode = mode.drop_duplicates(subset=group_cols, keep="first")
    mode = mode.rename(columns={"vessel_type": "mode_vessel_type"})
    return mode[group_cols + ["obs_count", "mode_vessel_type", "mode_vessel_share"]]


def build_h3_stats(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """
    Calcula estadísticas contextuales multi-nivel.

    Contexto numérico:
        (h3_res7, vessel_type, time_band) → ... → h3_res5 → global
    Contexto de tipo de barco:
        (h3_res7, time_band) → ... → h3_res5 → global
    """
    df_tmp = _prepare_context_frame(df)

    stats_bundle: dict[str, pd.DataFrame] = {}
    for level_name, group_cols in NUMERIC_CONTEXT_LEVELS.items():
        stats_bundle[level_name] = _build_numeric_group_stats(df_tmp, group_cols)
    for level_name, group_cols in VTYPE_CONTEXT_LEVELS.items():
        stats_bundle[level_name] = _build_vtype_mode_stats(df_tmp, group_cols)

    valid_vtype = df_tmp["vessel_type"].dropna()
    if valid_vtype.empty:
        global_mode_vessel_type = None
        global_mode_vessel_share = 0.0
    else:
        vc = valid_vtype.value_counts(dropna=True)
        global_mode_vessel_type = float(vc.index[0])
        global_mode_vessel_share = float(vc.iloc[0] / vc.sum())

    global_stats = {
        "obs_count": float(len(df_tmp)),
        "sog_median": float(df_tmp["sog"].median()),
        "sog_std": float(df_tmp["sog"].std() or 0.0),
        "cog_sin_mean": float(df_tmp["_cog_sin"].mean()),
        "cog_cos_mean": float(df_tmp["_cog_cos"].mean()),
        "heading_sin_mean": float(df_tmp["_heading_sin"].mean()),
        "heading_cos_mean": float(df_tmp["_heading_cos"].mean()),
        "length_median": float(df_tmp["length"].median()),
        "length_std": float(df_tmp["length"].std() or 0.0),
        "width_median": float(df_tmp["width"].median()),
        "width_std": float(df_tmp["width"].std() or 0.0),
        "draft_median": float(df_tmp["draft"].median()),
        "draft_std": float(df_tmp["draft"].std() or 0.0),
        "mode_vessel_type": global_mode_vessel_type,
        "mode_vessel_share": global_mode_vessel_share,
    }

    print(
        "[INFO] Estadisticas contextuales calculadas:\n"
        f"       ctx_exact={len(stats_bundle['ctx_exact']):,}  "
        f"ctx_h3_vtype={len(stats_bundle['ctx_h3_vtype']):,}  "
        f"ctx_h3_time={len(stats_bundle['ctx_h3_time']):,}\n"
        f"       ctx_h3={len(stats_bundle['ctx_h3']):,}  "
        f"parent_exact={len(stats_bundle['parent_exact']):,}  "
        f"parent_h3={len(stats_bundle['parent_h3']):,}\n"
        f"       global sog_median={global_stats['sog_median']:.2f}  "
        f"length_median={global_stats['length_median']:.2f}  "
        f"mode_vessel_type={global_stats['mode_vessel_type']}"
    )
    return stats_bundle, stats_bundle.get("parent_h3", pd.DataFrame()), global_stats


# ---------------------------------------------------------------------------
# 2. Preparación de la matriz de features
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
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
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, imputer, scaler


# ---------------------------------------------------------------------------
# 3. Entrenamiento
# ---------------------------------------------------------------------------

def train_model(
    X: np.ndarray,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    max_samples: int = MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> IsolationForest:
    print(
        f"[INFO] Entrenando Isolation Forest …\n"
        f"       filas={len(X):,}  features={X.shape[1]}"
        f"  n_estimators={n_estimators}  max_samples={max_samples}"
        f"  contamination={contamination}"
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
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
    df["is_anomaly"] = model.predict(X_scaled).astype(np.int8)
    df["anomaly_score"] = model.decision_function(X_scaled).astype(np.float32)

    n_total = len(df)
    n_anomaly = (df["is_anomaly"] == -1).sum()
    pct = n_anomaly / n_total * 100

    print(f"[INFO] Total registros : {n_total:,}")
    print(f"[INFO] Anomalías (-1)  : {n_anomaly:,}  ({pct:.2f} %)")
    print(f"[INFO] Normales  ( 1)  : {n_total - n_anomaly:,}  ({100 - pct:.2f} %)\n")
    return df


# ---------------------------------------------------------------------------
# 5. Resumen de anomalías
# ---------------------------------------------------------------------------

REPORT_COLS = [
    "mmsi", "vessel_name", "base_date_time",
    "latitude", "longitude",
    "h3_res7", "h3_res5",
    "time_band", "time_band_label", "context_level", "vtype_context_level",
    "sog", "cog", "heading",
    "vessel_type", "status", "length", "width", "draft",
    "hour", "day_of_week",
    "hex_log_density", "is_sparse_hex", "is_new_hex",
    "sog_delta_hex_med", "sog_z_hex",
    "length_delta_hex_med", "length_z_hex",
    "width_delta_hex_med", "width_z_hex",
    "draft_delta_hex_med", "draft_z_hex",
    "vtype_mode_share_hex", "is_unusual_vtype_hex",
    "anomaly_score",
]


def save_anomaly_report(df: pd.DataFrame, path: str) -> None:
    cols = [c for c in REPORT_COLS if c in df.columns]
    anomalies = df[df["is_anomaly"] == -1][cols].sort_values("anomaly_score")
    anomalies.to_csv(path, index=False)
    print(f"[INFO] Resumen guardado en '{path}'  ({len(anomalies):,} filas)\n")
    if not anomalies.empty:
        print("[INFO] Top 10 registros más anómalos:")
        print(anomalies.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main(
    csv_path: str = CSV_FILE,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    max_samples: int = MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
    context_hour_mode: str = CONTEXT_HOUR_MODE,
    hour_bucket_size: int = HOUR_BUCKET_SIZE,
    min_obs_context: int = H3_CONTEXT_MIN_OBS,
    min_obs_hex: int = H3_MIN_OBS,
    vtype_min_share: float = H3_VTYPE_MIN_SHARE,
    models_dir: str = MODELS_DIR,
    anomalies_csv: str = ANOMALIES_CSV,
) -> pd.DataFrame:
    # 1. Cargar y preprocesar
    df = preprocess(
        csv_path,
        context_hour_mode=context_hour_mode,
        hour_bucket_size=hour_bucket_size,
    )

    # 2. Asignar celdas H3
    df = add_h3_cells(df)

    # 3. Construir estadísticas contextuales
    h3_stats, h3_parent_stats, global_stats = build_h3_stats(df)

    # 4. Enriquecer con contexto H3 + tipo + hora
    df = enrich_with_h3_context(
        df,
        h3_stats,
        h3_parent_stats,
        global_stats,
        min_obs=min_obs_hex,
        min_obs_context=min_obs_context,
        vtype_min_share=vtype_min_share,
        hour_mode=context_hour_mode,
        hour_bucket_size=hour_bucket_size,
    )

    # 5. Preparar matriz de features
    X_scaled, imputer, scaler = prepare_features(df)

    # 6. Entrenar
    model = train_model(
        X_scaled,
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
    )

    # 7. Etiquetar
    df = predict_and_label(df, model, X_scaled)

    # 8. Guardar artefactos
    save_artifacts(
        model,
        scaler,
        imputer,
        h3_stats,
        h3_parent_stats,
        global_stats,
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        context_hour_mode=context_hour_mode,
        hour_bucket_size=hour_bucket_size,
        min_obs_context=min_obs_context,
        min_obs_hex=min_obs_hex,
        vtype_min_share=vtype_min_share,
        models_dir=models_dir,
    )

    # 9. Resumen de anomalías
    save_anomaly_report(df, anomalies_csv)
    return df


# ---------------------------------------------------------------------------
# Exportación e importación de artefactos
# ---------------------------------------------------------------------------

def save_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    imputer: SimpleImputer,
    h3_stats: dict[str, pd.DataFrame],
    h3_parent_stats: pd.DataFrame,
    global_stats: dict,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    max_samples: int = MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
    context_hour_mode: str = CONTEXT_HOUR_MODE,
    hour_bucket_size: int = HOUR_BUCKET_SIZE,
    min_obs_context: int = H3_CONTEXT_MIN_OBS,
    min_obs_hex: int = H3_MIN_OBS,
    vtype_min_share: float = H3_VTYPE_MIN_SHARE,
    models_dir: str = MODELS_DIR,
) -> None:
    """Exporta todos los artefactos del pipeline."""
    paths = build_artifact_paths(models_dir)

    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(imputer, paths["imputer"])
    joblib.dump(h3_stats, paths["h3_stats"])
    joblib.dump(h3_parent_stats, paths["h3_parent_stats"])

    h3_config = {
        "h3_resolution": H3_RESOLUTION,
        "parent_resolution": H3_PARENT_RESOLUTION,
        "context_hour_mode": context_hour_mode,
        "hour_bucket_size": hour_bucket_size,
        "min_obs_hex": min_obs_hex,
        "min_obs_context": min_obs_context,
        "vtype_min_share_hex": vtype_min_share,
        "numeric_context_levels": NUMERIC_CONTEXT_LEVELS,
        "vtype_context_levels": VTYPE_CONTEXT_LEVELS,
        "global_stats": global_stats,
    }
    with open(paths["h3_config"], "w", encoding="utf-8") as f:
        json.dump(h3_config, f, indent=2)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "feature_cols": FEATURE_COLS,
        "heading_no_disp": HEADING_NO_DISP,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
            "random_state": random_state,
        },
        "context": {
            "hour_mode": context_hour_mode,
            "hour_bucket_size": hour_bucket_size,
            "min_obs_context": min_obs_context,
            "min_obs_hex": min_obs_hex,
            "vtype_min_share": vtype_min_share,
        },
        "artifacts": {
            "model": os.path.basename(paths["model"]),
            "scaler": os.path.basename(paths["scaler"]),
            "imputer": os.path.basename(paths["imputer"]),
            "h3_stats": os.path.basename(paths["h3_stats"]),
            "h3_parent_stats": os.path.basename(paths["h3_parent_stats"]),
            "h3_config": os.path.basename(paths["h3_config"]),
        },
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[INFO] Artefactos exportados en '{models_dir}':\n"
        f"       · {os.path.basename(paths['model'])}\n"
        f"       · {os.path.basename(paths['scaler'])}\n"
        f"       · {os.path.basename(paths['imputer'])}\n"
        f"       · {os.path.basename(paths['h3_stats'])}\n"
        f"       · {os.path.basename(paths['h3_parent_stats'])}\n"
        f"       · {os.path.basename(paths['h3_config'])}\n"
        f"       · {os.path.basename(paths['metadata'])}\n"
    )


def load_artifacts(
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    imputer_path: str = IMPUTER_PATH,
    h3_stats_path: str = H3_STATS_PATH,
    h3_parent_stats_path: str = H3_PARENT_STATS_PATH,
    h3_config_path: str = H3_CONFIG_PATH,
    metadata_path: str = METADATA_PATH,
) -> tuple[IsolationForest, StandardScaler, SimpleImputer, dict, pd.DataFrame, dict, dict]:
    """Carga todos los artefactos del pipeline."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    h3_stats = joblib.load(h3_stats_path)
    h3_parent_stats = joblib.load(h3_parent_stats_path)

    with open(h3_config_path, "r", encoding="utf-8") as f:
        h3_config = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(h3_stats, dict):
        h3_stats = {"ctx_h3": h3_stats, "parent_h3": h3_parent_stats}

    print(
        f"[INFO] Artefactos cargados desde '{os.path.dirname(model_path)}' "
        f"(entrenado: {metadata.get('trained_at', 'desconocido')})\n"
        f"       H3 res={h3_config['h3_resolution']}  "
        f"parent_res={h3_config['parent_resolution']}  "
        f"hour_mode={h3_config.get('context_hour_mode', 'bucket')}  "
        f"min_obs_context={h3_config.get('min_obs_context', H3_CONTEXT_MIN_OBS)}"
    )
    return model, scaler, imputer, h3_stats, h3_parent_stats, h3_config, metadata


# ---------------------------------------------------------------------------
# Inferencia sobre nuevos datos
# ---------------------------------------------------------------------------

def predict_new(
    df_new: pd.DataFrame,
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    imputer_path: str = IMPUTER_PATH,
    h3_stats_path: str = H3_STATS_PATH,
    h3_parent_stats_path: str = H3_PARENT_STATS_PATH,
    h3_config_path: str = H3_CONFIG_PATH,
    metadata_path: str = METADATA_PATH,
) -> pd.DataFrame:
    """Aplica el pipeline exportado a un DataFrame nuevo."""
    model, scaler, imputer, h3_stats, h3_parent_stats, h3_config, meta = load_artifacts(
        model_path,
        scaler_path,
        imputer_path,
        h3_stats_path,
        h3_parent_stats_path,
        h3_config_path,
        metadata_path,
    )
    features = meta["feature_cols"]
    heading_no_disp = meta.get("heading_no_disp", 511)
    global_stats = h3_config["global_stats"]

    df_base = df_new.copy()
    if "h3_res7" not in df_base.columns:
        df_base = add_h3_cells(
            df_base,
            resolution=h3_config["h3_resolution"],
            parent_resolution=h3_config["parent_resolution"],
        )

    if "hex_log_density" not in df_base.columns:
        df_base = enrich_with_h3_context(
            df_base,
            h3_stats,
            h3_parent_stats,
            global_stats,
            min_obs=h3_config.get("min_obs_hex", H3_MIN_OBS),
            min_obs_context=h3_config.get("min_obs_context", H3_CONTEXT_MIN_OBS),
            vtype_min_share=h3_config.get("vtype_min_share_hex", H3_VTYPE_MIN_SHARE),
            hour_mode=h3_config.get("context_hour_mode", CONTEXT_HOUR_MODE),
            hour_bucket_size=h3_config.get("hour_bucket_size", HOUR_BUCKET_SIZE),
        )

    X = df_base[features].copy()
    X["heading"] = X["heading"].replace(heading_no_disp, np.nan)

    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)

    df_out = df_base.copy()
    df_out["is_anomaly"] = model.predict(X_scaled).astype(np.int8)
    df_out["anomaly_score"] = model.decision_function(X_scaled).astype(np.float32)
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo AIS con contexto H3+tipo+hora.")
    parser.add_argument("csv_path", nargs="?", default=CSV_FILE)
    parser.add_argument("--contamination", type=float, default=CONTAMINATION)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--context-hour-mode", choices=["bucket", "exact"], default=CONTEXT_HOUR_MODE)
    parser.add_argument("--hour-bucket-size", type=int, default=HOUR_BUCKET_SIZE)
    parser.add_argument("--min-obs-context", type=int, default=H3_CONTEXT_MIN_OBS)
    parser.add_argument("--min-obs-hex", type=int, default=H3_MIN_OBS)
    parser.add_argument("--vtype-min-share", type=float, default=H3_VTYPE_MIN_SHARE)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--anomalies-csv", default=ANOMALIES_CSV)
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        random_state=args.random_state,
        context_hour_mode=args.context_hour_mode,
        hour_bucket_size=args.hour_bucket_size,
        min_obs_context=args.min_obs_context,
        min_obs_hex=args.min_obs_hex,
        vtype_min_share=args.vtype_min_share,
        models_dir=args.models_dir,
        anomalies_csv=args.anomalies_csv,
    )
