"""
predict_realtime.py
Inferencia: carga modelo/scaler y predice anomalías basadas en lat/lon.
"""

import os
import sys

import numpy as np
import pandas as pd

from load_ais_data import CSV_FILE, preprocess
from train_anomaly import load_artifacts, MODELS_DIR


class AISAnomalyDetector:
    """Detector de anomalías AIS por posición espacial."""

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        model_path = os.path.join(models_dir, "isolation_forest_model.joblib")
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        metadata_path = os.path.join(models_dir, "metadata.json")

        self.model, self.scaler, self.meta = load_artifacts(model_path, scaler_path, metadata_path)
        self.feature_cols = self.meta["feature_cols"]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predice anomalías."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")

        df_out = df.copy()
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.transform(X)

        df_out["is_anomaly"] = self.model.predict(X_scaled).astype(np.int8)
        df_out["anomaly_score"] = self.model.decision_function(X_scaled).astype(np.float32)
        return df_out

    def predict_record(self, record: dict) -> dict:
        """Predice un registro individual."""
        row = {
            "latitude": record.get("latitude", np.nan),
            "longitude": record.get("longitude", np.nan),
        }
        df = pd.DataFrame([row])
        result = self.predict(df)

        return {
            "is_anomaly": int(result["is_anomaly"].iloc[0]),
            "anomaly_score": float(result["anomaly_score"].iloc[0]),
        }


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE

    detector = AISAnomalyDetector()

    print(f"[INFO] Cargando y preprocesando '{csv_path}' …\n")
    df = preprocess(csv_path)

    print("[INFO] Ejecutando inferencia …")
    result = detector.predict(df)

    n_total = len(result)
    n_anom = (result["is_anomaly"] == -1).sum()
    pct = n_anom / n_total * 100

    print(f"\n[INFO] Total: {n_total:,}  Anomalías: {n_anom:,} ({pct:.2f}%)")
    print("[INFO] Top 5 más anómalos:")
    top = result[result["is_anomaly"] == -1][["mmsi", "vessel_name", "latitude", "longitude", "anomaly_score"]].sort_values("anomaly_score").head(5)
    print(top.to_string(index=False))
