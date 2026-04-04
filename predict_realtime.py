"""
predict_realtime.py
Inferencia: carga modelo/scaler y predice anomalías basadas en lat/lon.
"""

import argparse
import os

import numpy as np
import pandas as pd
import shap
import joblib

from load_ais_data import CSV_FILE, preprocess, map_vessel_type
from train_anomaly import load_artifacts, MODELS_DIR
from spatial_context import CTX_FEATURE_COLS, add_spatial_context_features


class AISAnomalyDetector:
    """Detector de anomalías AIS por posición espacial."""

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        model_path = os.path.join(models_dir, "isolation_forest_model.joblib")
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        metadata_path = os.path.join(models_dir, "metadata.json")
        context_path = os.path.join(models_dir, "context_model.joblib")

        self.model, self.scaler, self.meta = load_artifacts(model_path, scaler_path, metadata_path)
        self.feature_cols = self.meta["feature_cols"]
        self.discard_missing_status = bool(self.meta.get("preprocessing", {}).get("discard_missing_status", False))
        self.context_model = joblib.load(context_path) if os.path.exists(context_path) else None
        self.score_calibration = self.meta.get("score_calibration")
        self.contamination = float(self.meta.get("hyperparameters", {}).get("contamination", 0.01))
        self._shap_explainer = None

    def _score_to_cdf(self, scores: np.ndarray) -> np.ndarray:
        """Devuelve CDF(score) estimada sobre entrenamiento (o fallback por lote)."""
        scores = np.asarray(scores, dtype=float)
        calib = self.score_calibration
        if isinstance(calib, dict) and calib.get("score_grid") and calib.get("cdf_grid"):
            score_grid = np.asarray(calib["score_grid"], dtype=float)
            cdf_grid = np.asarray(calib["cdf_grid"], dtype=float)
            order = np.argsort(score_grid)
            score_grid = score_grid[order]
            cdf_grid = cdf_grid[order]
            return np.interp(scores, score_grid, cdf_grid, left=0.0, right=1.0).astype("float32")

        # Fallback para modelos antiguos sin calibración en metadata.
        if scores.size == 0:
            return np.array([], dtype="float32")
        return pd.Series(scores).rank(method="average", pct=True).to_numpy(dtype="float32")

    def _anomaly_tail_reliability_pct(self, scores: np.ndarray, is_anomaly: np.ndarray) -> np.ndarray:
        """Fiabilidad 0-100 solo para anómalos, reescalada en la cola anómala por percentil."""
        scores = np.asarray(scores, dtype=float)
        is_anomaly = np.asarray(is_anomaly, dtype=bool)
        rel = np.full(scores.shape, np.nan, dtype="float32")
        if scores.size == 0:
            return rel

        q_cutoff = float(np.clip(self.contamination, 1e-6, 0.5))
        cdf = self._score_to_cdf(scores).astype(float)
        # En la cola izquierda (más anómalo = menor CDF), 0% en el umbral y 100% en el extremo.
        tail_rel = ((q_cutoff - cdf) / q_cutoff) * 100.0
        tail_rel = np.clip(tail_rel, 0.0, 100.0).astype("float32")
        rel[is_anomaly] = tail_rel[is_anomaly]
        return rel

    def _get_shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    def _build_operator_reason(self, feature: str, value) -> str:
        """Convierte la feature dominante en un texto entendible para operación."""
        messages = {
            "latitude": "Posicion geografica poco habitual en este sector.",
            "longitude": "Posicion geografica poco habitual en este sector.",
            "vessel_type_mapped": "Tipo de barco poco habitual para la zona.",
            "status": "Estado de navegacion poco habitual para la zona.",
            "sog": "Velocidad poco habitual para la zona.",
            "cog_sin": "Rumbo de desplazamiento poco habitual para la zona.",
            "cog_cos": "Rumbo de desplazamiento poco habitual para la zona.",
            "heading_sin": "Orientacion de proa poco habitual para la zona.",
            "heading_cos": "Orientacion de proa poco habitual para la zona.",
            "sog_ctx_logprob": "Velocidad no usual en el sector geografico para este tipo de trafico.",
            "cog_ctx_logprob": "Rumbo no usual en el sector geografico para este tipo de trafico.",
            "vessel_type_ctx_logprob": "Tipo de barco no usual en el sector geografico.",
            "status_ctx_logprob": "Estado de navegacion no usual en el sector geografico.",
        }

        base = messages.get(feature, "Comportamiento no usual para el patron local de trafico.")
        show_value_for = {
            "latitude",
            "longitude",
            "vessel_type_mapped",
            "status",
            "sog",
        }
        if feature not in show_value_for:
            return base

        if pd.isna(value):
            return base

        try:
            if isinstance(value, (int, float, np.integer, np.floating)):
                val_txt = f"{float(value):.2f}"
            else:
                val_txt = str(value)
        except Exception:
            val_txt = str(value)
        return f"{base} Dato observado: {val_txt}."

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
                reasons.append(self._build_operator_reason(feat, value))
            df_out.iloc[anom_idx, df_out.columns.get_loc("anomaly_reason")] = reasons
        except Exception as exc:
            # Fallback robusto si SHAP falla por versión/modelo
            z_abs = np.abs(X_anom_scaled)
            top_feat_idx = z_abs.argmax(axis=1)
            reasons = []
            for i, feat_idx in enumerate(top_feat_idx):
                feat = self.feature_cols[feat_idx]
                value = X.iloc[anom_idx[i], feat_idx]
                reasons.append(self._build_operator_reason(feat, value))
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

        # Features de contexto espacial si el modelo las requiere.
        needs_context = any(col in self.feature_cols for col in CTX_FEATURE_COLS)
        if needs_context:
            if self.context_model is None:
                raise ValueError("El modelo requiere context_model.joblib y no fue encontrado")
            df = add_spatial_context_features(df, self.context_model)

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")

        df_out = df.copy()
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.transform(X)

        df_out["is_anomaly"] = self.model.predict(X_scaled).astype(np.int8)
        df_out["anomaly_score"] = self.model.decision_function(X_scaled).astype(np.float32)
        df_out["anomaly_reliability_pct"] = self._anomaly_tail_reliability_pct(
            df_out["anomaly_score"].to_numpy(),
            (df_out["is_anomaly"] == -1).to_numpy(),
        )
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
            "anomaly_reliability_pct": (
                float(result["anomaly_reliability_pct"].iloc[0])
                if pd.notna(result["anomaly_reliability_pct"].iloc[0])
                else None
            ),
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
