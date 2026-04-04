"""
predict_realtime.py
-------------------
Script de inferencia en tiempo real para detección de anomalías AIS.
Carga los artefactos exportados por train_anomaly.py y predice si nuevos
registros AIS son anómalos sin necesidad de reentrenar.

Uso
---
  # Desde la línea de comandos:
  python predict_realtime.py                    # usa data/ais-data.csv
  python predict_realtime.py data/ais-data.csv  # pasa otro fichero

  # Importado como módulo:
  from predict_realtime import AISAnomalyDetector
  detector = AISAnomalyDetector()
  result   = detector.predict(df_preprocesado)

  # Un solo registro (dict con campos AIS en bruto):
  detector.predict_record({
      "latitude": 18.46, "longitude": -66.10,
      "sog": 0.0, "cog": 176.7, "heading": None,
      "hour": 0, "day_of_week": 2, "month": 1,
      "status": 0.0, "vessel_type": 70.0,
      "length": 70.0, "width": 18.0, "draft": 4.0,
  })
"""

import os
import sys

import numpy as np
import pandas as pd
import shap

from load_ais_data import (
    CSV_FILE,
    add_h3_cells,
    add_time_context,
    enrich_with_h3_context,
    preprocess,
)
from train_anomaly import HEADING_NO_DISP, MODELS_DIR, load_artifacts

# ---------------------------------------------------------------------------
# Mapa de features → razón legible
# ---------------------------------------------------------------------------

FEATURE_REASON_MAP = {
    # Temporales
    "hour_sin": "Horario de navegacion inusual",
    "hour_cos": "Horario de navegacion inusual",
    "day_of_week": "Dia de la semana inusual",
    "month": "Estacionalidad inusual",
    # Dinámicas
    "sog": "Velocidad inusual",
    "cog": "Rumbo inusual",
    "heading": "Orientacion inusual",
    "status": "Estado de navegacion inusual",
    # Estáticas
    "vessel_type": "Tipo de embarcacion fuera de patron",
    "length": "Eslora fuera de patron",
    "width": "Manga fuera de patron",
    "draft": "Calado fuera de patron",
    # Contexto H3 + tipo + hora
    "hex_log_density": "Contexto geografico-horario con densidad inusual de trafico",
    "is_sparse_hex": "Contexto geografico-horario poco frecuente",
    "is_new_hex": "Contexto geografico-horario desconocido",
    "sog_delta_hex_med": "Velocidad inusual para esta zona, tipo y franja horaria",
    "sog_z_hex": "Velocidad muy desviada respecto a su contexto local",
    "cog_delta_sin_hex": "Rumbo inusual para esta zona, tipo y franja horaria",
    "cog_delta_cos_hex": "Rumbo inusual para esta zona, tipo y franja horaria",
    "heading_delta_sin_hex": "Orientacion inusual para esta zona, tipo y franja horaria",
    "heading_delta_cos_hex": "Orientacion inusual para esta zona, tipo y franja horaria",
    "length_delta_hex_med": "Eslora fuera de patron en su contexto local",
    "length_z_hex": "Eslora muy desviada respecto a su contexto local",
    "width_delta_hex_med": "Manga fuera de patron en su contexto local",
    "width_z_hex": "Manga muy desviada respecto a su contexto local",
    "draft_delta_hex_med": "Calado fuera de patron en su contexto local",
    "draft_z_hex": "Calado muy desviado respecto a su contexto local",
    "vtype_mode_share_hex": "Tipo de embarcacion poco habitual para esta zona y franja horaria",
    "is_unusual_vtype_hex": "Tipo de embarcacion no habitual en esta zona y franja horaria",
}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class AISAnomalyDetector:
    """
    Encapsula el pipeline de detección de anomalías AIS.

    Carga una sola vez todos los artefactos desde disco y expone:
      - predict(df)         → DataFrame batch ya preprocesado
      - predict_record(dict)→ dict con un solo registro AIS en bruto
    """

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        model_path = os.path.join(models_dir, "isolation_forest_model.joblib")
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        imputer_path = os.path.join(models_dir, "imputer.joblib")
        h3_stats_path = os.path.join(models_dir, "h3_stats.joblib")
        h3_parent_stats_path = os.path.join(models_dir, "h3_parent_stats.joblib")
        h3_config_path = os.path.join(models_dir, "h3_config.json")
        metadata_path = os.path.join(models_dir, "metadata.json")

        (
            self.model,
            self.scaler,
            self.imputer,
            self.h3_stats,
            self.h3_parent_stats,
            self.h3_config,
            self.meta,
        ) = load_artifacts(
            model_path=model_path,
            scaler_path=scaler_path,
            imputer_path=imputer_path,
            h3_stats_path=h3_stats_path,
            h3_parent_stats_path=h3_parent_stats_path,
            h3_config_path=h3_config_path,
            metadata_path=metadata_path,
        )

        self.feature_cols = self.meta["feature_cols"]
        self.heading_no_disp = self.meta.get("heading_no_disp", 511)
        self.global_stats = self.h3_config["global_stats"]
        self.min_obs_hex = self.h3_config["min_obs_hex"]
        self.min_obs_context = self.h3_config.get("min_obs_context", self.min_obs_hex)
        self.hour_mode = self.h3_config.get("context_hour_mode", "bucket")
        self.hour_bucket_size = self.h3_config.get("hour_bucket_size", 6)
        self.vtype_min_share = self.h3_config.get("vtype_min_share_hex", 0.55)
        self._shap_explainer = None

        ctx_exact_count = len(self.h3_stats.get("ctx_exact", [])) if isinstance(self.h3_stats, dict) else len(self.h3_stats)
        print(
            f"[INFO] AISAnomalyDetector listo.\n"
            f"       Features ({len(self.feature_cols)}): {self.feature_cols}\n"
            f"       Contaminación: {self.meta['hyperparameters']['contamination']}\n"
            f"       H3 res={self.h3_config['h3_resolution']}  contextos_exactos={ctx_exact_count:,}\n"
            f"       Contexto temporal: mode={self.hour_mode}  bucket={self.hour_bucket_size}\n"
        )

    # ------------------------------------------------------------------
    def _ensure_h3_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade celdas H3 y features de contexto si aún no existen."""
        df = add_time_context(
            df,
            hour_mode=self.hour_mode,
            hour_bucket_size=self.hour_bucket_size,
        )
        if "h3_res7" not in df.columns:
            df = add_h3_cells(
                df,
                resolution=self.h3_config["h3_resolution"],
                parent_resolution=self.h3_config["parent_resolution"],
            )
        if "hex_log_density" not in df.columns:
            df = enrich_with_h3_context(
                df,
                self.h3_stats,
                self.h3_parent_stats,
                self.global_stats,
                min_obs=self.min_obs_hex,
                min_obs_context=self.min_obs_context,
                vtype_min_share=self.vtype_min_share,
                hour_mode=self.hour_mode,
                hour_bucket_size=self.hour_bucket_size,
            )
        return df

    # ------------------------------------------------------------------
    def _prepare_feature_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Construye la matriz de features transformada para inferencia."""
        X = df[self.feature_cols].copy()
        X["heading"] = X["heading"].replace(self.heading_no_disp, np.nan)
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return X, X_scaled

    # ------------------------------------------------------------------
    def _get_shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    # ------------------------------------------------------------------
    def _map_feature_to_reason(self, feature_name: str) -> str:
        return FEATURE_REASON_MAP.get(feature_name, f"Patron inusual en {feature_name}")

    # ------------------------------------------------------------------
    def _add_anomaly_reasons(self, df_out: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
        df_out = df_out.copy()
        df_out["anomaly_reason"] = "Normal"

        anomaly_mask = df_out["is_anomaly"] == -1
        if not anomaly_mask.any():
            return df_out

        anomaly_idx = np.where(anomaly_mask.values)[0]
        X_anomaly = X_scaled[anomaly_idx]

        try:
            explainer = self._get_shap_explainer()
            shap_values = explainer.shap_values(X_anomaly)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            top_feat_idx = np.abs(shap_values).argmax(axis=1)
            top_features = [self.feature_cols[i] for i in top_feat_idx]
            reasons = [self._map_feature_to_reason(f) for f in top_features]

        except Exception as exc:
            print(f"[WARN] SHAP no disponible ({exc}). Fallback por desviacion estandar.")
            z_abs = np.abs(X_anomaly)
            top_feat_idx = z_abs.argmax(axis=1)
            top_features = [self.feature_cols[i] for i in top_feat_idx]
            reasons = [self._map_feature_to_reason(f) for f in top_features]

        df_out.iloc[anomaly_idx, df_out.columns.get_loc("anomaly_reason")] = reasons
        return df_out

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice anomalías sobre un DataFrame preprocesado.
        Si el DataFrame no tiene las features contextuales calculadas,
        las calcula automáticamente a partir de latitude/longitude y hour.
        """
        df = self._ensure_h3_context(df)

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas requeridas ausentes: {missing}")

        _, X_scaled = self._prepare_feature_matrix(df)

        df_out = df.copy()
        df_out["is_anomaly"] = self.model.predict(X_scaled).astype(np.int8)
        df_out["anomaly_score"] = self.model.decision_function(X_scaled).astype(np.float32)
        df_out = self._add_anomaly_reasons(df_out, X_scaled)
        return df_out

    # ------------------------------------------------------------------
    def predict_record(self, record: dict) -> dict:
        """
        Predice sobre un único registro AIS expresado como diccionario.

        Devuelve columnas clave de trazabilidad del contexto, además
        del resultado de anomalía.
        """
        row = {k: record.get(k, np.nan) for k in [
            "latitude", "longitude",
            "sog", "cog", "heading", "status",
            "vessel_type", "length", "width", "draft",
            "hour", "day_of_week", "month",
            "hour_sin", "hour_cos",
            "time_band", "time_band_label",
        ]}
        if pd.isna(row.get("hour_sin")) and not pd.isna(row.get("hour")):
            angle = 2 * np.pi * row["hour"] / 24
            row["hour_sin"] = float(np.sin(angle))
            row["hour_cos"] = float(np.cos(angle))

        df_single = pd.DataFrame([row])
        result = self.predict(df_single)

        return {
            "is_anomaly": int(result["is_anomaly"].iloc[0]),
            "anomaly_score": float(result["anomaly_score"].iloc[0]),
            "anomaly_reason": str(result["anomaly_reason"].iloc[0]),
            "h3_res7": str(result["h3_res7"].iloc[0]),
            "time_band": float(result["time_band"].iloc[0]) if "time_band" in result.columns else None,
            "time_band_label": str(result["time_band_label"].iloc[0]) if "time_band_label" in result.columns else "N/A",
            "context_level": str(result["context_level"].iloc[0]) if "context_level" in result.columns else "N/A",
            "sog_delta_hex_med": float(result["sog_delta_hex_med"].iloc[0]),
            "sog_z_hex": float(result["sog_z_hex"].iloc[0]),
            "length_z_hex": float(result["length_z_hex"].iloc[0]) if "length_z_hex" in result.columns else None,
            "width_z_hex": float(result["width_z_hex"].iloc[0]) if "width_z_hex" in result.columns else None,
            "draft_z_hex": float(result["draft_z_hex"].iloc[0]) if "draft_z_hex" in result.columns else None,
            "is_new_hex": int(result["is_new_hex"].iloc[0]),
            "is_sparse_hex": int(result["is_sparse_hex"].iloc[0]),
        }


# ---------------------------------------------------------------------------
# Demo / ejecución directa
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    n_total = len(df)
    n_anomaly = (df["is_anomaly"] == -1).sum()
    pct = n_anomaly / n_total * 100

    print(f"\n{'=' * 60}")
    print(f"  Total registros : {n_total:,}")
    print(f"  Anomalías (-1)  : {n_anomaly:,}  ({pct:.2f} %)")
    print(f"  Normales  ( 1)  : {n_total - n_anomaly:,}  ({100 - pct:.2f} %)")
    print(f"{'=' * 60}\n")

    cols_show = [
        "mmsi", "vessel_name", "latitude", "longitude", "h3_res7",
        "time_band_label", "context_level",
        "sog", "sog_delta_hex_med", "sog_z_hex",
        "length", "length_z_hex", "width", "width_z_hex", "draft", "draft_z_hex",
        "is_new_hex", "is_sparse_hex",
        "is_anomaly", "anomaly_score", "anomaly_reason",
    ]
    available = [c for c in cols_show if c in df.columns]
    print("[INFO] Top 10 más anómalos:")
    print(
        df[df["is_anomaly"] == -1][available]
        .sort_values("anomaly_score")
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE

    detector = AISAnomalyDetector()

    print(f"[INFO] Cargando y preprocesando '{csv_path}' …\n")
    df_raw = preprocess(
        csv_path,
        context_hour_mode=detector.hour_mode,
        hour_bucket_size=detector.hour_bucket_size,
    )

    print("[INFO] Ejecutando inferencia …")
    df_result = detector.predict(df_raw)

    _print_summary(df_result)

    sample = df_raw.iloc[0]
    single_result = detector.predict_record({
        "latitude": float(sample["latitude"]),
        "longitude": float(sample["longitude"]),
        "sog": float(sample["sog"]) if not pd.isna(sample["sog"]) else None,
        "cog": float(sample["cog"]) if not pd.isna(sample["cog"]) else None,
        "heading": float(sample["heading"]) if not pd.isna(sample["heading"]) else None,
        "hour": int(sample["hour"]),
        "day_of_week": int(sample["day_of_week"]),
        "month": int(sample["month"]),
        "status": float(sample["status"]) if not pd.isna(sample["status"]) else None,
        "vessel_type": float(sample["vessel_type"]) if not pd.isna(sample["vessel_type"]) else None,
        "length": float(sample["length"]) if not pd.isna(sample["length"]) else None,
        "width": float(sample["width"]) if not pd.isna(sample["width"]) else None,
        "draft": float(sample["draft"]) if not pd.isna(sample["draft"]) else None,
    })
    print(f"\n[INFO] Predicción registro individual:\n{single_result}")
