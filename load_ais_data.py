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


def map_vessel_type(value) -> int:
    """
    Mapea vessel_type con reglas parciales:
      1x->10, 2x->20, 6x->60, 7x->70, 8x->80, 9x->90
    Nulo/desconocido -> 0.
    Cualquier otro valor se conserva.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0
    if pd.isna(num):
        return 0

    num_i = int(num)
    dec = num_i // 10 * 10
    if dec in {10, 20, 60, 70, 80, 90}:
        return dec
    return num_i


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

    # Feature categórica simplificada para el modelo (vectorizado)
    if "vessel_type" in df.columns:
        vt = pd.to_numeric(df["vessel_type"], errors="coerce")
        vt_int = vt.fillna(0).astype("int32")
        vt_dec = (vt_int // 10) * 10
        df["vessel_type_mapped"] = vt_int
        # Nulo/desconocido => 0
        df.loc[vt.isna(), "vessel_type_mapped"] = 0
        allowed = {10, 20, 60, 70, 80, 90}
        df.loc[vt_dec.isin(allowed), "vessel_type_mapped"] = vt_dec[vt_dec.isin(allowed)]
        df["vessel_type_mapped"] = df["vessel_type_mapped"].astype("int32")
    else:
        df["vessel_type_mapped"] = 0
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
