"""
predict_realtime.py
-------------------
Script de inferencia en tiempo real para detección de anomalías AIS.
Carga los artefactos exportados por train_anomaly.py (modelo, scaler e imputer)
y predice si nuevos registros AIS son anómalos sin necesidad de reentrenar.

Uso
---
  # Desde la línea de comandos:
  python predict_realtime.py                    # usa data/ais-data.csv
  python predict_realtime.py data/ais-data.csv # pasa otro fichero

  # Importado como módulo en otra aplicación:
  from predict_realtime import AISAnomalyDetector
  detector = AISAnomalyDetector()
  result   = detector.predict(df_nuevos_datos)
"""

import os
import sys

import pandas as pd
import shap

from load_ais_data import preprocess, CSV_FILE
from train_anomaly import load_artifacts, HEADING_NO_DISP, MODELS_DIR
import numpy as np

# ---------------------------------------------------------------------------
# Clase reutilizable para inferencia
# ---------------------------------------------------------------------------

FEATURE_REASON_MAP = {
    "sog": "Velocidad inusual",
    "cog": "Rumbo inusual",
    "heading": "Orientacion inusual",
    "lat_norm": "Ubicacion geografica fuera de patron",
    "lon_norm": "Ubicacion geografica fuera de patron",
    "grid_x": "Zona geografica fuera de patron",
    "grid_y": "Zona geografica fuera de patron",
    "hour_sin": "Horario de navegacion inusual",
    "hour_cos": "Horario de navegacion inusual",
    "day_of_week": "Dia de la semana inusual",
    "month": "Estacionalidad inusual",
    "status": "Estado de navegacion inusual",
    "vessel_type": "Tipo de embarcacion fuera de patron",
    "length": "Dimension de eslora fuera de patron",
    "width": "Dimension de manga fuera de patron",
    "draft": "Calado fuera de patron",
    "cargo": "Tipo de carga fuera de patron",
}

class AISAnomalyDetector:
    """
    Encapsula el pipeline de detección de anomalías AIS ya entrenado.

    Carga una sola vez modelo, scaler e imputer desde disco y expone
    el método ``predict()`` para uso en tiempo real.

    Parámetros
    ----------
    models_dir : carpeta donde están los artefactos .joblib y metadata.json
    """

    def __init__(self, models_dir: str = MODELS_DIR) -> None:
        model_path    = os.path.join(models_dir, "isolation_forest_model.joblib")
        scaler_path   = os.path.join(models_dir, "scaler.joblib")
        imputer_path  = os.path.join(models_dir, "imputer.joblib")
        metadata_path = os.path.join(models_dir, "metadata.json")

        self.model, self.scaler, self.imputer, self.meta = load_artifacts(
            model_path, scaler_path, imputer_path, metadata_path
        )
        self.feature_cols    = self.meta["feature_cols"]
        self.heading_no_disp = self.meta.get("heading_no_disp", 511)
        self._shap_explainer = None

        print(
            f"[INFO] AISAnomalyDetector listo.\n"
            f"       Features ({len(self.feature_cols)}): {self.feature_cols}\n"
            f"       Contaminación: {self.meta['hyperparameters']['contamination']}\n"
        )

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
        """Inicializa el explainer SHAP una sola vez."""
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)
        return self._shap_explainer

    # ------------------------------------------------------------------
    def _map_feature_to_reason(self, feature_name: str) -> str:
        """Traduce el nombre de feature dominante a texto legible."""
        return FEATURE_REASON_MAP.get(feature_name, f"Patron inusual en {feature_name}")

    # ------------------------------------------------------------------
    def _add_anomaly_reasons(self, df_out: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
        """
        Agrega 'anomaly_reason' usando SHAP en filas anomalias.
        Para filas normales se marca como 'Normal'.
        """
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
            df_out.iloc[anomaly_idx, df_out.columns.get_loc("anomaly_reason")] = reasons

        except Exception as exc:
            print(f"[WARN] SHAP no disponible ({exc}). Se aplicara fallback por desviacion estandar.")
            z_abs = np.abs(X_anomaly)
            top_feat_idx = z_abs.argmax(axis=1)
            top_features = [self.feature_cols[i] for i in top_feat_idx]
            reasons = [self._map_feature_to_reason(f) for f in top_features]
            df_out.iloc[anomaly_idx, df_out.columns.get_loc("anomaly_reason")] = reasons

        return df_out

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice anomalías sobre un DataFrame ya preprocesado
        (salida de load_ais_data.preprocess o equivalente).

        Columnas añadidas
        -----------------
        is_anomaly    : int8  → -1 anómalo / 1 normal
        anomaly_score : float → más negativo = más anómalo

        Devuelve una copia del DataFrame con esas dos columnas.
        """
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"El DataFrame no contiene las columnas requeridas: {missing}"
            )

        _, X_scaled = self._prepare_feature_matrix(df)

        df_out = df.copy()
        df_out["is_anomaly"]    = self.model.predict(X_scaled).astype(np.int8)
        df_out["anomaly_score"] = self.model.decision_function(X_scaled).astype(np.float32)
        df_out = self._add_anomaly_reasons(df_out, X_scaled)
        return df_out

    # ------------------------------------------------------------------
    def predict_record(self, record: dict) -> dict:
        """
        Predice sobre un único registro AIS expresado como diccionario.
        Útil para integraciones con colas de mensajes (Kafka, MQTT, etc.).

        Ejemplo
        -------
        detector.predict_record({
            "lat_norm": 0.36, "lon_norm": 0.33,
            "grid_x": 227, "grid_y": 216,
            "hour": 0, "day_of_week": 2, "month": 1,
            "sog": 0.0, "cog": 176.7, "heading": None,
            "status": 0.0, "vessel_type": 70.0,
            "length": 70.0, "width": 18.0, "draft": 4.0, "cargo": 70.0,
        })
        → {"is_anomaly": 1, "anomaly_score": 0.042}
        """
        row = {col: record.get(col, np.nan) for col in self.feature_cols}
        df_single = pd.DataFrame([row])
        result = self.predict(df_single)
        return {
            "is_anomaly"   : int(result["is_anomaly"].iloc[0]),
            "anomaly_score": float(result["anomaly_score"].iloc[0]),
            "anomaly_reason": str(result["anomaly_reason"].iloc[0]),
        }


# ---------------------------------------------------------------------------
# Demo / ejecución directa
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    n_total   = len(df)
    n_anomaly = (df["is_anomaly"] == -1).sum()
    pct       = n_anomaly / n_total * 100

    print(f"\n{'='*60}")
    print(f"  Total registros : {n_total:,}")
    print(f"  Anomalías (-1)  : {n_anomaly:,}  ({pct:.2f} %)")
    print(f"  Normales  ( 1)  : {n_total - n_anomaly:,}  ({100-pct:.2f} %)")
    print(f"{'='*60}\n")

    cols_show = [
        "mmsi", "vessel_name", "latitude", "longitude",
        "sog", "hour", "vessel_type", "is_anomaly", "anomaly_score", "anomaly_reason",
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
    # Permite pasar una ruta CSV como argumento opcional
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE

    print(f"[INFO] Cargando y preprocesando '{csv_path}' …\n")
    df_raw = preprocess(csv_path)

    # Instanciar el detector (carga artefactos una sola vez)
    detector = AISAnomalyDetector()

    # Inferencia
    print("[INFO] Ejecutando inferencia …")
    df_result = detector.predict(df_raw)

    _print_summary(df_result)

    # Ejemplo de predicción sobre un único registro en tiempo real
    sample_record = df_raw.iloc[0][detector.feature_cols].to_dict()
    single_result = detector.predict_record(sample_record)
    print(f"[INFO] Predicción sobre un registro individual: {single_result}")

