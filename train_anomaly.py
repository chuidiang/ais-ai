"""
train_anomaly.py
Entrena modelo Isolation Forest con lat/lon escalados.
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
from sklearn.preprocessing import StandardScaler

from load_ais_data import CSV_FILE, preprocess

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
ANOMALIES_CSV = os.path.join(DATA_DIR, "anomalies_summary.csv")

CONTAMINATION = 0.01
N_ESTIMATORS = 100
MAX_SAMPLES = 1024
RANDOM_STATE = 42
FEATURE_COLS = ["latitude", "longitude", "vessel_type_mapped"]


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Escala lat/lon y vessel_type_mapped."""
    X = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[INFO] Features escaladas: {X.shape[1]} cols, {X.shape[0]:,} rows")
    return X_scaled, scaler


def train_model(
    X: np.ndarray,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    max_samples: int = MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> IsolationForest:
    print(f"[INFO] Entrenando IsolationForest: n={len(X):,} features={X.shape[1]} contam={contamination}")
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    print("[INFO] Entrenamiento completado\n")
    return model


def predict_and_label(
    df: pd.DataFrame,
    model: IsolationForest,
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    df = df.copy()
    df["is_anomaly"] = model.predict(X_scaled).astype(np.int8)
    df["anomaly_score"] = model.decision_function(X_scaled).astype(np.float32)

    n_total = len(df)
    n_anom = (df["is_anomaly"] == -1).sum()
    pct = n_anom / n_total * 100

    print(f"[INFO] Total: {n_total:,}  Anomalías: {n_anom:,} ({pct:.2f}%)\n")
    return df


def save_anomaly_report(df: pd.DataFrame, path: str) -> None:
    anomalies = df[df["is_anomaly"] == -1][
        ["mmsi", "vessel_name", "base_date_time", "latitude", "longitude", "vessel_type_mapped", "anomaly_score"]
    ].sort_values("anomaly_score")
    anomalies.to_csv(path, index=False)
    print(f"[INFO] Resumen guardado: {path} ({len(anomalies):,} anomalías)")
    if not anomalies.empty:
        print("[INFO] Top 5 más anómalos:")
        print(anomalies.head(5).to_string(index=False))


def save_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    max_samples: int = MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
    models_dir: str = MODELS_DIR,
) -> None:
    """Guarda modelo, scaler y metadata."""
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "isolation_forest_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    metadata_path = os.path.join(models_dir, "metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "feature_cols": FEATURE_COLS,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "contamination": contamination,
            "random_state": random_state,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Artefactos guardados en {models_dir}/")


def load_artifacts(
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    metadata_path: str = METADATA_PATH,
) -> tuple[IsolationForest, StandardScaler, dict]:
    """Carga modelo, scaler y metadata."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[INFO] Artefactos cargados (entrenado: {metadata.get('trained_at', 'N/A')})\n")
    return model, scaler, metadata


def main(
    csv_path: str = CSV_FILE,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    max_samples: int = MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
    models_dir: str = MODELS_DIR,
    anomalies_csv: str = ANOMALIES_CSV,
) -> pd.DataFrame:
    df = preprocess(csv_path)
    X_scaled, scaler = prepare_features(df)
    model = train_model(X_scaled, contamination, n_estimators, max_samples, random_state)
    df = predict_and_label(df, model, X_scaled)
    save_artifacts(model, scaler, contamination, n_estimators, max_samples, random_state, models_dir)
    save_anomaly_report(df, anomalies_csv)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena modelo AIS (lat/lon).")
    parser.add_argument("csv_path", nargs="?", default=CSV_FILE)
    parser.add_argument("--contamination", type=float, default=CONTAMINATION)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
