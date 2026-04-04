"""
load_ais_data.py
----------------
Carga y preprocesa datos AIS desde un fichero CSV.

Pasos:
  1. Carga todas las columnas del CSV.
  2. Convierte 'base_date_time' a objeto datetime.
  3. Extrae 'hour', 'day_of_week' y 'month' como columnas numéricas.
     Aplica codificación cíclica seno/coseno a 'hour' (period=24).
  4. Deriva una franja horaria configurable para el contexto del modelo.
  5. Limpia valores nulos y erróneos.
  6. Asigna celdas H3 (resolución 7 y 5 como padre).
  7. Enriquece con contexto H3 + vessel_type + hora/franja:
     densidad, residuals de sog/cog/heading y de tamaño.
"""

import os

import h3 as _h3
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_FILE = os.path.join(DATA_DIR, "ais-data.csv")

# Resoluciones H3
H3_RESOLUTION        = 7    # celda local (~5 km de lado)
H3_PARENT_RESOLUTION = 5    # macrozona de fallback (~86 km de lado)
H3_MIN_OBS           = 500  # mínimo para usar contexto H3 puro
H3_CONTEXT_MIN_OBS   = 100  # mínimo para usar contexto H3+tipo+franja
H3_VTYPE_MIN_SHARE   = 0.55 # share mínimo para marcar rareza de tipo

# Contexto temporal del modelo
CONTEXT_HOUR_MODE = "bucket"  # "bucket" o "exact"
HOUR_BUCKET_SIZE  = 6          # horas por franja cuando se usa modo bucket

NUMERIC_CONTEXT_LEVELS = {
    "ctx_exact":       ["h3_res7", "vessel_type", "time_band"],
    "ctx_h3_vtype":    ["h3_res7", "vessel_type"],
    "ctx_h3_time":     ["h3_res7", "time_band"],
    "ctx_h3":          ["h3_res7"],
    "parent_exact":    ["h3_res5", "vessel_type", "time_band"],
    "parent_h3_vtype": ["h3_res5", "vessel_type"],
    "parent_h3_time":  ["h3_res5", "time_band"],
    "parent_h3":       ["h3_res5"],
}
NUMERIC_CONTEXT_FALLBACK_ORDER = list(NUMERIC_CONTEXT_LEVELS.keys())

VTYPE_CONTEXT_LEVELS = {
    "vtype_h3_time":     ["h3_res7", "time_band"],
    "vtype_h3":          ["h3_res7"],
    "vtype_parent_time": ["h3_res5", "time_band"],
    "vtype_parent":      ["h3_res5"],
}
VTYPE_CONTEXT_FALLBACK_ORDER = list(VTYPE_CONTEXT_LEVELS.keys())

NUMERIC_CONTEXT_STAT_COLS = [
    "obs_count",
    "sog_median", "sog_std",
    "cog_sin_mean", "cog_cos_mean",
    "heading_sin_mean", "heading_cos_mean",
    "length_median", "length_std",
    "width_median", "width_std",
    "draft_median", "draft_std",
]
VTYPE_CONTEXT_STAT_COLS = ["obs_count", "mode_vessel_type", "mode_vessel_share"]

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


def add_time_context(
    df: pd.DataFrame,
    hour_mode: str = CONTEXT_HOUR_MODE,
    hour_bucket_size: int = HOUR_BUCKET_SIZE,
) -> pd.DataFrame:
    """
    Añade una representación temporal estable para el contexto del modelo.

    Columnas generadas
    ------------------
    time_band       : entero usado en agrupaciones contextuales
    time_band_label : texto legible para trazabilidad y tooltip
    """
    if "hour" not in df.columns:
        if "base_date_time" not in df.columns:
            raise ValueError("Se requiere 'hour' o 'base_date_time' para construir el contexto temporal.")
        df = process_datetime(df.copy())
    else:
        df = df.copy()

    hour_mode = (hour_mode or CONTEXT_HOUR_MODE).strip().lower()
    if hour_mode not in {"bucket", "exact"}:
        raise ValueError(f"hour_mode no soportado: {hour_mode!r}. Usa 'bucket' o 'exact'.")

    hour_vals = pd.to_numeric(df["hour"], errors="coerce")
    valid = hour_vals.notna()
    labels = pd.Series("N/A", index=df.index, dtype="object")

    if hour_mode == "exact":
        band = hour_vals.astype("float32")
        labels.loc[valid] = hour_vals.loc[valid].astype(int).map(lambda h: f"{h:02d}:00")
    else:
        bucket = int(hour_bucket_size)
        if bucket <= 0 or bucket > 24:
            raise ValueError("hour_bucket_size debe estar entre 1 y 24.")
        band = np.floor(hour_vals / bucket).astype("float32")

        def _bucket_label(hour_value: float) -> str:
            hour_int = int(hour_value)
            start = (hour_int // bucket) * bucket
            end = min(start + bucket - 1, 23)
            return f"{start:02d}-{end:02d}"

        labels.loc[valid] = hour_vals.loc[valid].map(_bucket_label)

    df["time_band"] = band
    df["time_band_label"] = labels
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
# 5. Enriquecimiento con contexto H3 + vessel_type + hora/franja
# ---------------------------------------------------------------------------

def _lookup_stats_by_level(
    df: pd.DataFrame,
    stats_df: pd.DataFrame | None,
    group_cols: list[str],
    stat_cols: list[str],
) -> dict[str, np.ndarray]:
    if stats_df is None or stats_df.empty:
        return {col: np.full(len(df), np.nan, dtype=float) for col in stat_cols}

    stats_idx = stats_df.set_index(group_cols)
    lookup_idx = pd.MultiIndex.from_frame(df[group_cols])
    matched = stats_idx.reindex(lookup_idx)

    out = {}
    for col in stat_cols:
        out[col] = pd.to_numeric(matched[col], errors="coerce").to_numpy(dtype=float)
    return out


def _resolve_numeric_context(
    df: pd.DataFrame,
    stats_bundle: dict,
    global_stats: dict,
    min_obs_context: int,
    min_obs_hex: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_rows = len(df)
    resolved = {col: np.full(n_rows, np.nan, dtype=float) for col in NUMERIC_CONTEXT_STAT_COLS}
    level_used = np.full(n_rows, "global", dtype=object)
    remaining = np.ones(n_rows, dtype=bool)

    for level_name in NUMERIC_CONTEXT_FALLBACK_ORDER:
        if not remaining.any():
            break

        group_cols = NUMERIC_CONTEXT_LEVELS[level_name]
        sub_df = df.loc[remaining, group_cols]
        level_stats = _lookup_stats_by_level(
            sub_df,
            stats_bundle.get(level_name),
            group_cols,
            NUMERIC_CONTEXT_STAT_COLS,
        )

        obs = level_stats["obs_count"]
        min_obs_level = min_obs_hex if level_name in {"ctx_h3", "parent_h3"} else min_obs_context
        qualifies = ~np.isnan(obs) & (obs >= min_obs_level)
        if not qualifies.any():
            continue

        sub_idx = np.flatnonzero(remaining)
        chosen_idx = sub_idx[qualifies]
        for col in NUMERIC_CONTEXT_STAT_COLS:
            resolved[col][chosen_idx] = level_stats[col][qualifies]
        level_used[chosen_idx] = level_name
        remaining[chosen_idx] = False

    if remaining.any():
        for col in NUMERIC_CONTEXT_STAT_COLS:
            g_val = global_stats.get(col, np.nan)
            resolved[col][remaining] = np.nan if g_val is None else float(g_val)

    return resolved, level_used


def _resolve_vtype_context(
    df: pd.DataFrame,
    stats_bundle: dict,
    global_stats: dict,
    min_obs_context: int,
    min_obs_hex: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_rows = len(df)
    resolved = {col: np.full(n_rows, np.nan, dtype=float) for col in VTYPE_CONTEXT_STAT_COLS}
    level_used = np.full(n_rows, "global", dtype=object)
    remaining = np.ones(n_rows, dtype=bool)

    for level_name in VTYPE_CONTEXT_FALLBACK_ORDER:
        if not remaining.any():
            break

        group_cols = VTYPE_CONTEXT_LEVELS[level_name]
        sub_df = df.loc[remaining, group_cols]
        level_stats = _lookup_stats_by_level(
            sub_df,
            stats_bundle.get(level_name),
            group_cols,
            VTYPE_CONTEXT_STAT_COLS,
        )

        obs = level_stats["obs_count"]
        min_obs_level = min_obs_hex if level_name in {"vtype_h3", "vtype_parent"} else min_obs_context
        qualifies = ~np.isnan(obs) & (obs >= min_obs_level)
        if not qualifies.any():
            continue

        sub_idx = np.flatnonzero(remaining)
        chosen_idx = sub_idx[qualifies]
        for col in VTYPE_CONTEXT_STAT_COLS:
            resolved[col][chosen_idx] = level_stats[col][qualifies]
        level_used[chosen_idx] = level_name
        remaining[chosen_idx] = False

    if remaining.any():
        for col in VTYPE_CONTEXT_STAT_COLS:
            g_val = global_stats.get(col, np.nan)
            resolved[col][remaining] = np.nan if g_val is None else float(g_val)

    return resolved, level_used


def enrich_with_h3_context(
    df: pd.DataFrame,
    h3_stats,
    h3_parent_stats: pd.DataFrame | None,
    global_stats: dict,
    min_obs: int = H3_MIN_OBS,
    min_obs_context: int = H3_CONTEXT_MIN_OBS,
    vtype_min_share: float = H3_VTYPE_MIN_SHARE,
    hour_mode: str = CONTEXT_HOUR_MODE,
    hour_bucket_size: int = HOUR_BUCKET_SIZE,
) -> pd.DataFrame:
    """
    Calcula features contextuales con fallback jerárquico usando dos cadenas:
      1. Contexto numérico: H3 + vessel_type + franja → ... → global
      2. Rareza de tipo    : H3 + franja              → ... → global
    """
    df = add_time_context(df.copy(), hour_mode=hour_mode, hour_bucket_size=hour_bucket_size)

    stats_bundle = h3_stats if isinstance(h3_stats, dict) else {
        "ctx_h3": h3_stats,
        "parent_h3": h3_parent_stats,
    }

    cog_num = pd.to_numeric(df["cog"], errors="coerce").fillna(0.0)
    heading_num = (
        pd.to_numeric(df["heading"], errors="coerce")
        .replace(_HEADING_NO_DISP, np.nan)
        .fillna(0.0)
    )
    cog_rad = np.deg2rad(cog_num)
    heading_rad = np.deg2rad(heading_num)
    _cog_sin = np.sin(cog_rad).values
    _cog_cos = np.cos(cog_rad).values
    _heading_sin = np.sin(heading_rad).values
    _heading_cos = np.cos(heading_rad).values

    resolved_num, level_used = _resolve_numeric_context(
        df,
        stats_bundle,
        global_stats,
        min_obs_context=min_obs_context,
        min_obs_hex=min_obs,
    )
    resolved_vtype, vtype_level_used = _resolve_vtype_context(
        df,
        stats_bundle,
        global_stats,
        min_obs_context=min_obs_context,
        min_obs_hex=min_obs,
    )

    sog_vals = pd.to_numeric(df["sog"], errors="coerce").values.astype(float)
    sog_med = resolved_num["sog_median"]
    sog_std = np.maximum(resolved_num["sog_std"], 0.1)
    sog_for_delta = np.where(np.isnan(sog_vals), sog_med, sog_vals)
    sog_delta = sog_for_delta - sog_med

    def _delta_and_z(series_name: str, median_key: str, std_key: str) -> tuple[np.ndarray, np.ndarray]:
        raw_vals = pd.to_numeric(df[series_name], errors="coerce").values.astype(float)
        med = resolved_num[median_key]
        std = np.maximum(resolved_num[std_key], 0.1)
        vals = np.where(np.isnan(raw_vals), med, raw_vals)
        delta = vals - med
        return delta.astype(np.float32), (delta / std).astype(np.float32)

    length_delta, length_z = _delta_and_z("length", "length_median", "length_std")
    width_delta, width_z = _delta_and_z("width", "width_median", "width_std")
    draft_delta, draft_z = _delta_and_z("draft", "draft_median", "draft_std")

    sparse_levels = set(NUMERIC_CONTEXT_FALLBACK_ORDER) - {"ctx_exact"}

    df["context_level"] = level_used.astype(str)
    df["vtype_context_level"] = vtype_level_used.astype(str)
    df["context_obs_count"] = resolved_num["obs_count"].astype(np.float32)
    df["vtype_context_obs_count"] = resolved_vtype["obs_count"].astype(np.float32)

    df["hex_log_density"] = np.log1p(resolved_num["obs_count"]).astype(np.float32)
    df["is_new_hex"] = (level_used == "global").astype(np.int8)
    df["is_sparse_hex"] = np.isin(level_used, list(sparse_levels)).astype(np.int8)
    df["sog_delta_hex_med"] = sog_delta.astype(np.float32)
    df["sog_z_hex"] = (sog_delta / sog_std).astype(np.float32)
    df["cog_delta_sin_hex"] = (_cog_sin - resolved_num["cog_sin_mean"]).astype(np.float32)
    df["cog_delta_cos_hex"] = (_cog_cos - resolved_num["cog_cos_mean"]).astype(np.float32)
    df["heading_delta_sin_hex"] = (_heading_sin - resolved_num["heading_sin_mean"]).astype(np.float32)
    df["heading_delta_cos_hex"] = (_heading_cos - resolved_num["heading_cos_mean"]).astype(np.float32)
    df["length_delta_hex_med"] = length_delta
    df["length_z_hex"] = length_z
    df["width_delta_hex_med"] = width_delta
    df["width_z_hex"] = width_z
    df["draft_delta_hex_med"] = draft_delta
    df["draft_z_hex"] = draft_z

    vessel_type_num = pd.to_numeric(df["vessel_type"], errors="coerce").values.astype(float)
    mode_type = resolved_vtype["mode_vessel_type"]
    mode_share = np.clip(resolved_vtype["mode_vessel_share"], 0.0, 1.0)
    has_type_info = (~np.isnan(vessel_type_num)) & (~np.isnan(mode_type))
    clear_pattern = (resolved_vtype["obs_count"] >= min_obs_context) & (mode_share >= vtype_min_share)
    is_unusual = has_type_info & clear_pattern & (vessel_type_num != mode_type)

    df["vtype_mode_share_hex"] = mode_share.astype(np.float32)
    df["is_unusual_vtype_hex"] = is_unusual.astype(np.int8)

    n_new = int(df["is_new_hex"].sum())
    n_sparse = int(df["is_sparse_hex"].sum())
    n_exact = int((df["context_level"] == "ctx_exact").sum())
    print(f"[INFO] Contexto H3+tipo+hora: new={n_new:,}  sparse={n_sparse:,}  exact={n_exact:,}")
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def preprocess(
    path: str = CSV_FILE,
    context_hour_mode: str = CONTEXT_HOUR_MODE,
    hour_bucket_size: int = HOUR_BUCKET_SIZE,
) -> pd.DataFrame:
    """
    Pipeline base de preprocesado AIS: carga, fechas, franja temporal y limpieza.

    La asignación H3 y el enriquecimiento con contexto H3 se realizan
    en pasos separados (add_h3_cells / enrich_with_h3_context) para
    poder reutilizar las estadísticas en entrenamiento e inferencia.
    """
    df = load_csv(path)
    df = process_datetime(df)
    df = add_time_context(df, hour_mode=context_hour_mode, hour_bucket_size=hour_bucket_size)
    df = clean_data(df)

    print("\n[INFO] Resumen final:")
    print(df.dtypes.to_string())
    print(f"\n[INFO] Filas finales: {len(df):,}")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    df = preprocess()

