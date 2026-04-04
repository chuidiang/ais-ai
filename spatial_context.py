"""
spatial_context.py
Features de contexto espacial para AIS usando probabilidades locales por celda.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRID_SIZE_DEG = 0.25
ALPHA = 1.0
SOG_MAX = 40.0
SOG_BINS = 20
COG_BINS = 16

CTX_FEATURE_COLS = [
    "sog_ctx_logprob",
    "cog_ctx_logprob",
    "vessel_type_ctx_logprob",
    "status_ctx_logprob",
]


def build_spatial_cell(df: pd.DataFrame, grid_size_deg: float = GRID_SIZE_DEG) -> pd.Series:
    """Crea ID de celda regular a partir de lat/lon."""
    lat_bin = np.floor((df["latitude"].astype(float) + 90.0) / grid_size_deg).astype("int32")
    lon_bin = np.floor((df["longitude"].astype(float) + 180.0) / grid_size_deg).astype("int32")
    return lat_bin.astype(str) + "_" + lon_bin.astype(str)


def _fit_local_prob_model(
    cell: pd.Series,
    value: pd.Series,
    n_values: int,
    alpha: float,
) -> dict:
    """Ajusta conteos por (celda, valor) y globales para calcular probabilidades suavizadas."""
    value = value.astype(str)
    key = cell.astype(str) + "|" + value

    key_counts = key.value_counts().to_dict()
    cell_totals = cell.astype(str).value_counts().to_dict()
    global_counts = value.value_counts().to_dict()

    return {
        "key_counts": key_counts,
        "cell_totals": cell_totals,
        "global_counts": global_counts,
        "n_values": int(max(n_values, 1)),
        "alpha": float(alpha),
    }


def _transform_local_logprob(
    cell: pd.Series,
    value: pd.Series,
    model: dict,
) -> np.ndarray:
    """Transforma valores en log-probabilidad local con fallback global."""
    cell_s = cell.astype(str)
    value_s = value.astype(str)

    key = cell_s + "|" + value_s

    key_counts = key.map(model["key_counts"]).fillna(0.0).astype(float)
    cell_totals = cell_s.map(model["cell_totals"]).fillna(0.0).astype(float)

    global_counts = value_s.map(model["global_counts"]).fillna(0.0).astype(float)

    n_values = float(model["n_values"])
    alpha = float(model["alpha"])
    global_total = float(sum(model["global_counts"].values()))

    local_prob = (key_counts + alpha) / (cell_totals + alpha * n_values)
    global_prob = (global_counts + alpha) / (global_total + alpha * n_values)

    prob = np.where(cell_totals > 0.0, local_prob, global_prob)
    return np.log(np.clip(prob, 1e-12, 1.0)).astype("float32")


def fit_spatial_context_model(
    df: pd.DataFrame,
    grid_size_deg: float = GRID_SIZE_DEG,
    alpha: float = ALPHA,
) -> dict:
    """Entrena modelos locales de probabilidad para SOG/COG/vessel_type/status."""
    cell = build_spatial_cell(df, grid_size_deg=grid_size_deg)

    sog_edges = np.linspace(0.0, SOG_MAX, SOG_BINS + 1)
    sog_bin = np.digitize(np.clip(df["sog"].astype(float), 0.0, SOG_MAX - 1e-9), sog_edges[1:-1], right=False)

    cog_deg = pd.to_numeric(df.get("cog", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0) % 360.0
    cog_edges = np.linspace(0.0, 360.0, COG_BINS + 1)
    cog_bin = np.digitize(cog_deg.astype(float), cog_edges[1:-1], right=False)

    vessel_type = df["vessel_type_mapped"].fillna(0).astype("int32")
    status = df["status"].fillna(15).astype("int32")

    model = {
        "grid_size_deg": float(grid_size_deg),
        "alpha": float(alpha),
        "sog_edges": sog_edges.tolist(),
        "cog_edges": cog_edges.tolist(),
        "sog_model": _fit_local_prob_model(cell, pd.Series(sog_bin, index=df.index), n_values=SOG_BINS, alpha=alpha),
        "cog_model": _fit_local_prob_model(cell, pd.Series(cog_bin, index=df.index), n_values=COG_BINS, alpha=alpha),
        "vessel_type_model": _fit_local_prob_model(
            cell, vessel_type, n_values=vessel_type.nunique(dropna=False), alpha=alpha
        ),
        "status_model": _fit_local_prob_model(
            cell, status, n_values=status.nunique(dropna=False), alpha=alpha
        ),
    }
    return model


def add_spatial_context_features(df: pd.DataFrame, context_model: dict) -> pd.DataFrame:
    """Agrega features de log-probabilidad local al dataframe."""
    out = df.copy()
    cell = build_spatial_cell(out, grid_size_deg=float(context_model["grid_size_deg"]))

    sog_edges = np.array(context_model["sog_edges"], dtype=float)
    sog_bin = np.digitize(np.clip(out["sog"].astype(float), 0.0, sog_edges[-1] - 1e-9), sog_edges[1:-1], right=False)

    # Si COG no existe, se asume 0 para mantener inferencia robusta.
    cog_deg = pd.to_numeric(out.get("cog", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0) % 360.0
    cog_edges = np.array(context_model["cog_edges"], dtype=float)
    cog_bin = np.digitize(cog_deg.astype(float), cog_edges[1:-1], right=False)

    vessel_type = out["vessel_type_mapped"].fillna(0).astype("int32")
    status = out["status"].fillna(15).astype("int32")

    out["sog_ctx_logprob"] = _transform_local_logprob(cell, pd.Series(sog_bin, index=out.index), context_model["sog_model"])
    out["cog_ctx_logprob"] = _transform_local_logprob(cell, pd.Series(cog_bin, index=out.index), context_model["cog_model"])
    out["vessel_type_ctx_logprob"] = _transform_local_logprob(cell, vessel_type, context_model["vessel_type_model"])
    out["status_ctx_logprob"] = _transform_local_logprob(cell, status, context_model["status_model"])

    return out

