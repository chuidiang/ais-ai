"""
predict_realtime.py
Inferencia: carga modelo/scaler y predice anomalías basadas en lat/lon.
"""

import argparse
import os

import numpy as np
import pandas as pd
import shap

from load_ais_data import CSV_FILE, preprocess, map_vessel_type
from train_anomaly import load_artifacts, MODELS_DIR


class AISAnomalyDetector:
    """Detector de anomalías AIS por posición espacial."""

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        model_path = os.path.join(models_dir, "isolation_forest_model.joblib")
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        metadata_path = os.path.join(models_dir, "metadata.json")

        self.model, self.scaler, self.meta = load_artifacts(model_path, scaler_path, metadata_path)
        self.feature_cols = self.meta["feature_cols"]
        self.discard_missing_status = bool(self.meta.get("preprocessing", {}).get("discard_missing_status", False))
        self._shap_explainer = None

    def _get_shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    def _add_anomaly_reasons(self, df_out: pd.DataFrame, X: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
        """Calcula SHAP solo en anómalos y añade feature dominante."""
        df_out = df_out.copy()
        df_out["anomaly_reason"] = ""

        anomaly_mask = df_out["is_anomaly"] == -1
        if not anomaly_mask.any():
            return df_out

        anom_idx = np.where(anomaly_mask.values)[0]
        X_anom_scaled = X_scaled[anom_idx]

        try:
            explainer = self._get_shap_explainer()
            shap_values = explainer.shap_values(X_anom_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            top_feat_idx = np.abs(shap_values).argmax(axis=1)
            reasons = []
            for i, feat_idx in enumerate(top_feat_idx):
                feat = self.feature_cols[feat_idx]
                value = X.iloc[anom_idx[i], feat_idx]
                impact = float(shap_values[i, feat_idx])
                reasons.append(f"{feat} es la feature dominante (SHAP={impact:.4f}, valor={value})")
            df_out.iloc[anom_idx, df_out.columns.get_loc("anomaly_reason")] = reasons
        except Exception as exc:
            # Fallback robusto si SHAP falla por versión/modelo
            z_abs = np.abs(X_anom_scaled)
            top_feat_idx = z_abs.argmax(axis=1)
            reasons = []
            for i, feat_idx in enumerate(top_feat_idx):
                feat = self.feature_cols[feat_idx]
                value = X.iloc[anom_idx[i], feat_idx]
                reasons.append(f"{feat} es la feature dominante (fallback, valor={value})")
            df_out.iloc[anom_idx, df_out.columns.get_loc("anomaly_reason")] = reasons
            print(f"[WARN] SHAP no disponible ({exc}); usando fallback.")

        return df_out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predice anomalías."""
        df = df.copy()
        if "vessel_type_mapped" not in df.columns:
            if "vessel_type" in df.columns:
                vt = pd.to_numeric(df["vessel_type"], errors="coerce")
                vt_int = vt.fillna(0).astype("int32")
                vt_dec = (vt_int // 10) * 10
                df["vessel_type_mapped"] = vt_int
                df.loc[vt.isna(), "vessel_type_mapped"] = 0
                allowed = {10, 20, 60, 70, 80, 90}
                df.loc[vt_dec.isin(allowed), "vessel_type_mapped"] = vt_dec[vt_dec.isin(allowed)]
                df["vessel_type_mapped"] = df["vessel_type_mapped"].astype("int32")
            else:
                df["vessel_type_mapped"] = 0

        # Status: imputa 15 si falta
        if "status" not in df.columns:
            df["status"] = 15
        else:
            df["status"] = df["status"].fillna(15).astype("int32")

        # SOG (Speed Over Ground)
        if "sog" not in df.columns:
            df["sog"] = 0.0
        else:
            df["sog"] = pd.to_numeric(df["sog"], errors="coerce").fillna(0).astype("float32")

        # COG sin/cos
        if "cog_sin" not in df.columns or "cog_cos" not in df.columns:
            if "cog" in df.columns:
                cog_rad = pd.to_numeric(df["cog"], errors="coerce") * (3.14159265 / 180.0)
                df["cog_sin"] = np.sin(cog_rad).fillna(0).astype("float32")
                df["cog_cos"] = np.cos(cog_rad).fillna(0).astype("float32")
            else:
                df["cog_sin"] = 0.0
                df["cog_cos"] = 0.0

        # Heading sin/cos
        if "heading_sin" not in df.columns or "heading_cos" not in df.columns:
            if "heading" in df.columns:
                heading_rad = pd.to_numeric(df["heading"], errors="coerce") * (3.14159265 / 180.0)
                df["heading_sin"] = np.sin(heading_rad).fillna(0).astype("float32")
                df["heading_cos"] = np.cos(heading_rad).fillna(0).astype("float32")
            else:
                df["heading_sin"] = 0.0
                df["heading_cos"] = 0.0

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
        # Convertir ángulos a radianes para sin/cos
        def to_trig(angle_deg):
            if pd.isna(angle_deg) or angle_deg is None:
                return 0.0, 0.0
            rad = float(angle_deg) * (3.14159265 / 180.0)
            return float(np.sin(rad)), float(np.cos(rad))

        cog_sin, cog_cos = to_trig(record.get("cog"))
        heading_sin, heading_cos = to_trig(record.get("heading"))

        row = {
            "latitude": record.get("latitude", np.nan),
            "longitude": record.get("longitude", np.nan),
            "vessel_type_mapped": map_vessel_type(record.get("vessel_type", np.nan)),
            "status": int(record.get("status", 15)) if record.get("status") is not None else 15,
            "sog": float(record.get("sog", 0.0)) if record.get("sog") is not None else 0.0,
            "cog_sin": cog_sin,
            "cog_cos": cog_cos,
            "heading_sin": heading_sin,
            "heading_cos": heading_cos,
        }
        df = pd.DataFrame([row])
        result = self.predict(df)

        return {
            "is_anomaly": int(result["is_anomaly"].iloc[0]),
            "anomaly_score": float(result["anomaly_score"].iloc[0]),
            "anomaly_reason": str(result["anomaly_reason"].iloc[0]),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia AIS con modelo entrenado.")
    parser.add_argument("csv_path", nargs="?", default=CSV_FILE)
    parser.add_argument("--models-dir", default=MODELS_DIR,
                        help="Directorio desde el que cargar modelo/scaler/metadata")
    args = parser.parse_args()

    detector = AISAnomalyDetector(models_dir=args.models_dir)

    print(f"[INFO] Cargando y preprocesando '{args.csv_path}' …\n")
    print(f"[INFO] Modo status del modelo: discard_missing_status={detector.discard_missing_status}")
    df = preprocess(args.csv_path, discard_missing_status=detector.discard_missing_status)

    print("[INFO] Ejecutando inferencia …")
    result = detector.predict(df)

    n_total = len(result)
    n_anom = (result["is_anomaly"] == -1).sum()
    pct = n_anom / n_total * 100

    print(f"\n[INFO] Total: {n_total:,}  Anomalías: {n_anom:,} ({pct:.2f}%)")
    print("[INFO] Top 5 más anómalos:")
    top = result[result["is_anomaly"] == -1][["mmsi", "vessel_name", "latitude", "longitude", "anomaly_score", "anomaly_reason"]].sort_values("anomaly_score").head(5)
    print(top.to_string(index=False))
