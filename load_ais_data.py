"""
load_ais_data.py
----------------
Carga y preprocesa datos AIS desde un fichero CSV.

Pasos:
  1. Carga todas las columnas del CSV.
  2. Convierte 'base_date_time' a objeto datetime.
  3. Extrae 'hour', 'day_of_week' y 'month' como columnas numéricas.
     Aplica codificación cíclica seno/coseno a 'hour' (period=24).
  4. Limpia valores nulos y erróneos.
  5. Segmenta el área geográfica en una cuadrícula (grid_x, grid_y).
  6. Normaliza 'latitude' y 'longitude' → 'lat_norm' y 'lon_norm'.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_FILE = os.path.join(DATA_DIR, "ais-data.csv")

# Tamaño de celda de la cuadrícula (en grados)
GRID_SIZE_DEG = 0.5

# Rangos válidos para campos numéricos AIS
LAT_MIN, LAT_MAX       = -90.0,  90.0
LON_MIN, LON_MAX       = -180.0, 180.0
SOG_MIN, SOG_MAX       =   0.0, 102.3   # nudos (102.3 = "≥102.2" en AIS)
COG_MIN, COG_MAX       =   0.0, 360.0
HEADING_VALID          = {*range(0, 361), 511}  # 511 = no disponible


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Lee el CSV con todas sus columnas."""
    print(f"[INFO] Cargando '{path}' …")
    df = pd.read_csv(
        path,
        dtype={
            "mmsi":        str,
            "vessel_name": str,
            "imo":         str,
            "call_sign":   str,
            "transceiver": str,
        },
        low_memory=False,
    )
    print(f"[INFO] Filas cargadas: {len(df):,}  |  Columnas: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 2. Conversión de fechas y extracción de características temporales
# ---------------------------------------------------------------------------

def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte 'base_date_time' y extrae características temporales.

    Columnas generadas
    ------------------
    hour        : entero 0-23  (para display / filtros)
    hour_sin    : sin(2π · hour / 24)  ── codificación cíclica
    hour_cos    : cos(2π · hour / 24)  ──   "         "
    day_of_week : entero 0 (lun) … 6 (dom)
    month       : entero 1 … 12

    La codificación cíclica hace que 23 h y 0 h queden contiguos
    en el espacio de features, eliminando el salto artificial
    que produciría usar el entero directamente.
    """
    df["base_date_time"] = pd.to_datetime(df["base_date_time"], errors="coerce")
    invalid_dates = df["base_date_time"].isna().sum()
    if invalid_dates:
        print(f"[WARN] {invalid_dates:,} filas con 'base_date_time' no parseable -> se eliminarán.")
    df = df.dropna(subset=["base_date_time"]).copy()

    hour_raw = df["base_date_time"].dt.hour

    # Columna original (útil para informes y filtros)
    df["hour"]        = hour_raw.astype(np.int8)
    df["day_of_week"] = df["base_date_time"].dt.dayofweek.astype(np.int8)  # 0=lunes … 6=domingo
    df["month"]       = df["base_date_time"].dt.month.astype(np.int8)

    # Codificación cíclica: period = 24 h
    angle = 2 * np.pi * hour_raw / 24
    df["hour_sin"] = np.sin(angle).astype(np.float32)
    df["hour_cos"] = np.cos(angle).astype(np.float32)

    print(
        f"[INFO] Codificacion ciclica aplicada a 'hour' -> "
        f"'hour_sin', 'hour_cos'  (period=24)"
    )
    return df


# ---------------------------------------------------------------------------
# 3. Limpieza de valores nulos y erróneos
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina o corrige filas con valores fuera de rango o nulos críticos."""
    before = len(df)

    # Coordenadas obligatorias y dentro de rango
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX) &
        df["longitude"].between(LON_MIN, LON_MAX)
    ]

    # SOG: debe ser numérico y no negativo
    df["sog"] = pd.to_numeric(df["sog"], errors="coerce")
    df = df[df["sog"].isna() | df["sog"].between(SOG_MIN, SOG_MAX)]

    # COG: entre 0 y 360
    df["cog"] = pd.to_numeric(df["cog"], errors="coerce")
    df = df[df["cog"].isna() | df["cog"].between(COG_MIN, COG_MAX)]

    # Heading: 0-360 o 511 (no disponible)
    df["heading"] = pd.to_numeric(df["heading"], errors="coerce")
    df = df[df["heading"].isna() | df["heading"].isin(HEADING_VALID)]

    # MMSI: exactamente 9 dígitos numéricos
    df = df[df["mmsi"].str.match(r"^\d{9}$", na=False)]

    # Columnas numéricas opcionales: convertir sin eliminar filas
    for col in ["vessel_type", "status", "length", "width", "draft", "cargo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    after = len(df)
    print(f"[INFO] Limpieza: {before:,} -> {after:,} filas  ({before - after:,} eliminadas)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Cuadrícula geográfica (grid)
# ---------------------------------------------------------------------------

def add_grid(df: pd.DataFrame, cell_deg: float = GRID_SIZE_DEG) -> pd.DataFrame:
    """
    Asigna cada punto a una celda de cuadrícula.
    'grid_x' → índice de columna (longitud)
    'grid_y' → índice de fila    (latitud)
    """
    df["grid_x"] = np.floor((df["longitude"] - LON_MIN) / cell_deg).astype(np.int16)
    df["grid_y"] = np.floor((df["latitude"]  - LAT_MIN) / cell_deg).astype(np.int16)
    n_cells = df[["grid_x", "grid_y"]].drop_duplicates().shape[0]
    print(f"[INFO] Cuadrícula {cell_deg}°: {n_cells:,} celdas ocupadas")
    return df


# ---------------------------------------------------------------------------
# 5. Normalización de coordenadas
# ---------------------------------------------------------------------------

def normalize_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza 'latitude' y 'longitude' al rango [0, 1] mediante MinMaxScaler.
    Añade columnas 'lat_norm' y 'lon_norm'.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    coords = df[["latitude", "longitude"]].values
    coords_norm = scaler.fit_transform(coords)
    df["lat_norm"] = coords_norm[:, 0].astype(np.float32)
    df["lon_norm"] = coords_norm[:, 1].astype(np.float32)
    print("[INFO] Coordenadas normalizadas -> 'lat_norm', 'lon_norm'")
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def preprocess(path: str = CSV_FILE) -> pd.DataFrame:
    df = load_csv(path)
    df = process_datetime(df)
    df = clean_data(df)
    df = add_grid(df)
    df = normalize_coords(df)

    print("\n[INFO] Resumen final:")
    print(df.dtypes.to_string())
    print(f"\n[INFO] Filas finales: {len(df):,}")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    df = preprocess()

