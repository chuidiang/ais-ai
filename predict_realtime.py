"""
predict_realtime.py
Inferencia: carga modelo/scaler y predice anomalías basadas en lat/lon.
"""

import os
import sys

import numpy as np
import pandas as pd
try:
    import shap
except Exception:
    shap = None

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
        self._shap_explainer = None

    def _get_shap_explainer(self):
        if shap is None:
            return None
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    def _add_anomaly_reasons(self, df_out: pd.DataFrame, X: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
        """Añade motivo de anomalía (feature dominante) para filas anómalas."""
        df_out = df_out.copy()
        df_out["anomaly_reason"] = ""

        anomaly_mask = df_out["is_anomaly"] == -1
        if not anomaly_mask.any():
            return df_out

        idx = np.where(anomaly_mask.values)[0]

        # 1) SHAP sobre las filas anómalas
        try:
            explainer = self._get_shap_explainer()
            if explainer is not None:
                shap_values = explainer.shap_values(X_scaled[idx])
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                top_feat_idx = np.abs(shap_values).argmax(axis=1)
                reasons = []
                for row_pos, feat_i in enumerate(top_feat_idx):
                    feat = self.feature_cols[feat_i]
                    val = X.iloc[idx[row_pos], feat_i]
                    impact = float(shap_values[row_pos, feat_i])
                    reasons.append(f"{feat} fuera de patrón (valor={val:.5f}, impacto_shap={impact:.4f})")

                df_out.iloc[idx, df_out.columns.get_loc("anomaly_reason")] = reasons
                return df_out
        except Exception as exc:
            print(f"[WARN] SHAP no disponible ({exc}). Se usa fallback.")

        # 2) Fallback: mayor desviación estandarizada en espacio escalado
        z_abs = np.abs(X_scaled[idx])
        top_feat_idx = z_abs.argmax(axis=1)
        reasons = []
        for row_pos, feat_i in enumerate(top_feat_idx):
            feat = self.feature_cols[feat_i]
            val = X.iloc[idx[row_pos], feat_i]
            z_val = float(X_scaled[idx[row_pos], feat_i])
            reasons.append(f"{feat} fuera de patrón (valor={val:.5f}, z={z_val:.3f})")

        df_out.iloc[idx, df_out.columns.get_loc("anomaly_reason")] = reasons
        return df_out

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
        df_out = self._add_anomaly_reasons(df_out, X, X_scaled)
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
            "anomaly_reason": str(result["anomaly_reason"].iloc[0]),
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
