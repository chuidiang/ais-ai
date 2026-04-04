"""
load_ais_data.py
Carga y preprocesa datos AIS desde CSV.
Solo mantiene lat/lon para detección de anomalías espaciales.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_FILE = os.path.join(DATA_DIR, "ais-data.csv")

LAT_MIN, LAT_MAX = -90.0,  90.0
LON_MIN, LON_MAX = -180.0, 180.0


def load_csv(path: str) -> pd.DataFrame:
    """Lee CSV con columnas necesarias."""
    print(f"[INFO] Cargando '{path}' …")
    df = pd.read_csv(path, dtype={"mmsi": str, "vessel_name": str}, low_memory=False)
    print(f"[INFO] Filas cargadas: {len(df):,}")
    return df


def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte base_date_time a datetime."""
    df["base_date_time"] = pd.to_datetime(df["base_date_time"], errors="coerce")
    invalid = df["base_date_time"].isna().sum()
    if invalid:
        print(f"[WARN] {invalid:,} fechas inválidas eliminadas")
    df = df.dropna(subset=["base_date_time"]).copy()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia valores nulos y erróneos."""
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX) &
        df["longitude"].between(LON_MIN, LON_MAX)
    ]
    after = len(df)
    print(f"[INFO] Limpieza: {before:,} -> {after:,} filas")
    return df.reset_index(drop=True)


def preprocess(path: str = CSV_FILE) -> pd.DataFrame:
    """Pipeline de preprocesado."""
    df = load_csv(path)
    df = process_datetime(df)
    df = clean_data(df)
    print(f"[INFO] Filas finales: {len(df):,}\n")
    return df


if __name__ == "__main__":
    df = preprocess()
