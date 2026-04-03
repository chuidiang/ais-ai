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
  5. Asigna celdas H3 (resolución 7 y 5 como padre).
  6. Enriquece con contexto H3: densidad, residuals de sog/cog/heading.
"""

import os
import pandas as pd
import numpy as np
import h3 as _h3

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_FILE = os.path.join(DATA_DIR, "ais-data.csv")

# Resoluciones H3
H3_RESOLUTION        = 7   # celda local (~5 km de lado)
H3_PARENT_RESOLUTION = 5   # macrozona de fallback (~86 km de lado)
H3_MIN_OBS           = 500 # mínimo de observaciones para considerar una celda "bien cubierta"
H3_VTYPE_MIN_SHARE   = 0.55 # share mínimo del tipo dominante para marcar rareza de tipo

# Rangos válidos para campos numéricos AIS
LAT_MIN, LAT_MAX   = -90.0,  90.0
LON_MIN, LON_MAX   = -180.0, 180.0
SOG_MIN, SOG_MAX   =   0.0, 102.3
COG_MIN, COG_MAX   =   0.0, 360.0
HEADING_VALID      = {*range(0, 361), 511}
_HEADING_NO_DISP   = 511


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
    hour        : entero 0-23
    hour_sin    : sin(2π · hour / 24)  ── codificación cíclica
    hour_cos    : cos(2π · hour / 24)
    day_of_week : 0 (lun) … 6 (dom)
    month       : 1 … 12
    """
    df["base_date_time"] = pd.to_datetime(df["base_date_time"], errors="coerce")
    invalid_dates = df["base_date_time"].isna().sum()
    if invalid_dates:
        print(f"[WARN] {invalid_dates:,} filas con 'base_date_time' no parseable -> se eliminarán.")
    df = df.dropna(subset=["base_date_time"]).copy()

    hour_raw = df["base_date_time"].dt.hour
    df["hour"]        = hour_raw.astype(np.int8)
    df["day_of_week"] = df["base_date_time"].dt.dayofweek.astype(np.int8)
    df["month"]       = df["base_date_time"].dt.month.astype(np.int8)

    angle = 2 * np.pi * hour_raw / 24
    df["hour_sin"] = np.sin(angle).astype(np.float32)
    df["hour_cos"] = np.cos(angle).astype(np.float32)

    print("[INFO] Codificacion ciclica aplicada a 'hour' -> 'hour_sin', 'hour_cos'  (period=24)")
    return df


# ---------------------------------------------------------------------------
# 3. Limpieza de valores nulos y erróneos
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina o corrige filas con valores fuera de rango o nulos críticos."""
    before = len(df)

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX) &
        df["longitude"].between(LON_MIN, LON_MAX)
    ]

    df["sog"] = pd.to_numeric(df["sog"], errors="coerce")
    df = df[df["sog"].isna() | df["sog"].between(SOG_MIN, SOG_MAX)]

    df["cog"] = pd.to_numeric(df["cog"], errors="coerce")
    df = df[df["cog"].isna() | df["cog"].between(COG_MIN, COG_MAX)]

    df["heading"] = pd.to_numeric(df["heading"], errors="coerce")
    df = df[df["heading"].isna() | df["heading"].isin(HEADING_VALID)]

    df = df[df["mmsi"].str.match(r"^\d{9}$", na=False)]

    for col in ["vessel_type", "status", "length", "width", "draft", "cargo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    after = len(df)
    print(f"[INFO] Limpieza: {before:,} -> {after:,} filas  ({before - after:,} eliminadas)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Segmentación geográfica con H3
# ---------------------------------------------------------------------------

def add_h3_cells(
    df: pd.DataFrame,
    resolution: int = H3_RESOLUTION,
    parent_resolution: int = H3_PARENT_RESOLUTION,
) -> pd.DataFrame:
    """
    Asigna a cada punto su celda H3 (resolución fina y padre).

    Columnas generadas
    ------------------
    h3_res7 : celda H3 resolución 7  (~5 km de lado)
    h3_res5 : celda padre resolución 5 (~86 km) para fallback de estadísticas
    """
    lats = df["latitude"].values
    lons = df["longitude"].values

    cells  = [_h3.latlng_to_cell(float(lat), float(lon), resolution)
              for lat, lon in zip(lats, lons)]
    parent = [_h3.cell_to_parent(c, parent_resolution) for c in cells]

    df = df.copy()
    df["h3_res7"] = cells
    df["h3_res5"] = parent

    n_cells  = len(set(cells))
    n_parent = len(set(parent))
    print(f"[INFO] H3 asignado: {n_cells:,} celdas res={resolution}  |  {n_parent:,} celdas res={parent_resolution}")
    return df


# ---------------------------------------------------------------------------
# 5. Enriquecimiento con contexto H3
# ---------------------------------------------------------------------------

def enrich_with_h3_context(
    df: pd.DataFrame,
    h3_stats: pd.DataFrame,
    h3_parent_stats: pd.DataFrame,
    global_stats: dict,
    min_obs: int = H3_MIN_OBS,
    vtype_min_share: float = H3_VTYPE_MIN_SHARE,
) -> pd.DataFrame:
    """
    Para cada fila calcula features contextuales basadas en las estadísticas
    históricas de su celda H3.  Usa fallback jerárquico:
        h3_res7 (bien cubierta)  →  h3_res5 (padre)  →  global

    Parámetros
    ----------
    h3_stats        : DataFrame con columnas [h3_res7, obs_count, sog_median,
                      sog_std, cog_sin_mean, cog_cos_mean,
                      heading_sin_mean, heading_cos_mean]
    h3_parent_stats : igual pero indexado por h3_res5
    global_stats    : dict con las mismas claves, valores escalares
    min_obs         : umbral mínimo para considerar celda "bien cubierta"

    Columnas generadas
    ------------------
    hex_log_density       : log1p(obs_count resuelto)
    is_sparse_hex         : 1 si celda fina escasa (se usó padre)
    is_new_hex            : 1 si celda completamente desconocida (se usó global)
    sog_delta_hex_med     : sog - mediana_sog_hex  (residual absoluto)
    sog_z_hex             : sog_delta / std_sog_hex (residual estandarizado)
    cog_delta_sin_hex     : sin(cog) - media_sin_cog_hex  (residual circular)
    cog_delta_cos_hex     : cos(cog) - media_cos_cog_hex
    heading_delta_sin_hex : sin(heading) - media_sin_heading_hex
    heading_delta_cos_hex : cos(heading) - media_cos_heading_hex
    vtype_mode_share_hex   : share del tipo dominante en la zona resuelta
    is_unusual_vtype_hex   : 1 si vessel_type no coincide con el tipo dominante
                             en una zona bien cubierta y con patrón dominante claro
    """
    df = df.copy()

    # Sin/cos de cog y heading (circulares) para los residuals
    # to_numeric garantiza float aunque el df venga de un dict con None
    cog_num     = pd.to_numeric(df["cog"],     errors="coerce").fillna(0.0)
    heading_num = pd.to_numeric(df["heading"], errors="coerce").replace(_HEADING_NO_DISP, np.nan).fillna(0.0)
    cog_rad     = np.deg2rad(cog_num)
    heading_rad = np.deg2rad(heading_num)
    _cog_sin     = np.sin(cog_rad).values
    _cog_cos     = np.cos(cog_rad).values
    _heading_sin = np.sin(heading_rad).values
    _heading_cos = np.cos(heading_rad).values

    # Merge con estadísticas de celda fina (h3_res7)
    stats7 = h3_stats.rename(columns={
        c: f"_s7_{c}" for c in h3_stats.columns if c != "h3_res7"
    })
    df = df.merge(stats7, on="h3_res7", how="left")

    # Merge con estadísticas de celda padre (h3_res5)
    stats5 = h3_parent_stats.rename(columns={
        c: f"_s5_{c}" for c in h3_parent_stats.columns if c != "h3_res5"
    })
    df = df.merge(stats5, on="h3_res5", how="left")

    # Flags de cobertura
    obs7      = df["_s7_obs_count"].values
    obs5      = df["_s5_obs_count"].values
    no_local  = np.isnan(obs7.astype(float)) | (obs7 < min_obs)
    no_parent = np.isnan(obs5.astype(float))

    df["is_new_hex"]    = (no_local &  no_parent).astype(np.int8)
    df["is_sparse_hex"] = (no_local & ~no_parent).astype(np.int8)

    # Resolver estadísticas con fallback: local → padre → global
    _STAT_COLS = [
        "obs_count", "sog_median", "sog_std",
        "cog_sin_mean", "cog_cos_mean",
        "heading_sin_mean", "heading_cos_mean",
        "mode_vessel_type", "mode_vessel_share",
    ]

    for col in _STAT_COLS:
        s7 = f"_s7_{col}"
        s5 = f"_s5_{col}"
        if s7 not in df.columns:
            df[s7] = np.nan
        if s5 not in df.columns:
            df[s5] = np.nan

    resolved = {}
    for col in _STAT_COLS:
        v7 = df[f"_s7_{col}"].values.astype(float)
        v5 = df[f"_s5_{col}"].values.astype(float)
        g_raw = global_stats.get(col, np.nan)
        g = np.nan if g_raw is None else float(g_raw)
        resolved[col] = np.where(~no_local, v7,
                        np.where(~no_parent, v5, g))

    # Features H3 derivadas
    sog_vals  = df["sog"].values.astype(float)
    sm_r      = resolved["sog_median"]
    ss_r      = np.maximum(resolved["sog_std"], 0.1)  # evitar división por cero
    # Si sog es NaN, delta = 0 (sin desviación respecto a la mediana local)
    sog_for_delta = np.where(np.isnan(sog_vals), sm_r, sog_vals)
    sog_delta     = sog_for_delta - sm_r

    df["hex_log_density"]       = np.log1p(resolved["obs_count"]).astype(np.float32)
    df["sog_delta_hex_med"]     = sog_delta.astype(np.float32)
    df["sog_z_hex"]             = (sog_delta / ss_r).astype(np.float32)
    df["cog_delta_sin_hex"]     = (_cog_sin  - resolved["cog_sin_mean"]).astype(np.float32)
    df["cog_delta_cos_hex"]     = (_cog_cos  - resolved["cog_cos_mean"]).astype(np.float32)
    df["heading_delta_sin_hex"] = (_heading_sin - resolved["heading_sin_mean"]).astype(np.float32)
    df["heading_delta_cos_hex"] = (_heading_cos - resolved["heading_cos_mean"]).astype(np.float32)

    vessel_type_num = pd.to_numeric(df["vessel_type"], errors="coerce").values
    mode_type       = resolved["mode_vessel_type"]
    mode_share      = np.clip(resolved["mode_vessel_share"], 0.0, 1.0)
    has_type_info   = (~np.isnan(vessel_type_num)) & (~np.isnan(mode_type))
    clear_pattern   = (resolved["obs_count"] >= min_obs) & (mode_share >= vtype_min_share)
    is_unusual      = has_type_info & clear_pattern & (vessel_type_num != mode_type)

    df["vtype_mode_share_hex"] = mode_share.astype(np.float32)
    df["is_unusual_vtype_hex"] = is_unusual.astype(np.int8)

    # Limpiar columnas temporales del merge
    tmp = [c for c in df.columns if c.startswith("_s7_") or c.startswith("_s5_")]
    df = df.drop(columns=tmp)

    n_new    = int(df["is_new_hex"].sum())
    n_sparse = int(df["is_sparse_hex"].sum())
    print(f"[INFO] H3 context: new={n_new:,}  sparse={n_sparse:,}  "
          f"local={len(df) - n_new - n_sparse:,}")
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def preprocess(path: str = CSV_FILE) -> pd.DataFrame:
    """
    Pipeline base de preprocesado AIS: carga, fechas y limpieza.

    La asignación H3 y el enriquecimiento con contexto H3 se realizan
    en pasos separados (add_h3_cells / enrich_with_h3_context) para
    poder reutilizar las estadísticas en entrenamiento e inferencia.
    """
    df = load_csv(path)
    df = process_datetime(df)
    df = clean_data(df)

    print("\n[INFO] Resumen final:")
    print(df.dtypes.to_string())
    print(f"\n[INFO] Filas finales: {len(df):,}")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    df = preprocess()

