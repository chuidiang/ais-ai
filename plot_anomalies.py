"""
plot_anomalies.py
-----------------
Genera un scatter plot interactivo (Plotly)
con la posición geográfica de los barcos, coloreados por 'is_anomaly'.

  Azul  (#4C9BE8) → comportamiento normal  (is_anomaly =  1)
  Rojo  (#E84C4C) → comportamiento anómalo (is_anomaly = -1)

Salidas
-------
  plots/anomalies_scatter_<sufijo>.html   ← interactivo, abrir en navegador
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shapefile

try:
    import geopandas as gpd
except Exception:
    gpd = None

from predict_realtime import AISAnomalyDetector
from load_ais_data import preprocess, CSV_FILE

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR          = os.path.dirname(__file__)
PLOTS_DIR         = os.path.join(BASE_DIR, "plots")
SHP_DIR           = os.path.join(BASE_DIR, "shp")
WORLD_SHP         = os.path.join(SHP_DIR, "world.shp")
ANOMALIES_SUMMARY = os.path.join(BASE_DIR, "data", "anomalies_summary.csv")
os.makedirs(PLOTS_DIR, exist_ok=True)

HTML_OUT = os.path.join(PLOTS_DIR, "anomalies_scatter.html")

# Para no saturar el gráfico con millones de puntos normales
NORMAL_SAMPLE = 80_000
RANDOM_SEED   = 42

COLOR_NORMAL  = "#4C9BE8"
COLOR_ANOMALY = "#E84C4C"
COLOR_MAP     = "#a7b2be"


def build_output_path(name_suffix: str | None = None) -> str:
    if not name_suffix:
        return HTML_OUT
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name_suffix)
    return os.path.join(PLOTS_DIR, f"anomalies_scatter_{safe}.html")


# ---------------------------------------------------------------------------
# Mapa base
# ---------------------------------------------------------------------------

def load_world_lines(shp_path: str = WORLD_SHP):
    if not os.path.exists(shp_path):
        print(f"[WARN] No se encontro shapefile en '{shp_path}'. Se omite mapa de fondo.")
        return []

    if gpd is not None:
        try:
            gdf   = gpd.read_file(shp_path)
            lines = []
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                boundary = geom.boundary
                geoms    = getattr(boundary, "geoms", [boundary])
                for line in geoms:
                    coords = list(line.coords)
                    if len(coords) < 2:
                        continue
                    lines.append(([c[0] for c in coords], [c[1] for c in coords]))
            print(f"[INFO] Mapa cargado (geopandas): {len(lines):,} segmentos")
            return lines
        except Exception as exc:
            print(f"[WARN] geopandas error: {exc}")

    try:
        reader = shapefile.Reader(shp_path)
        lines  = []
        for shape in reader.shapes():
            pts   = shape.points
            parts = list(shape.parts) + [len(pts)]
            for i in range(len(parts) - 1):
                seg = pts[parts[i]:parts[i + 1]]
                if len(seg) >= 2:
                    lines.append(([p[0] for p in seg], [p[1] for p in seg]))
        print(f"[INFO] Mapa cargado (pyshp): {len(lines):,} segmentos")
        return lines
    except Exception as exc:
        print(f"[WARN] No se pudo leer '{shp_path}': {exc}")
        return []


# ---------------------------------------------------------------------------
# 1. Carga de datos y predicción
# ---------------------------------------------------------------------------

def load_fast(csv_path: str = CSV_FILE,
              anomalies_csv: str = ANOMALIES_SUMMARY) -> pd.DataFrame:
    """
    Ruta rápida (recomendada):
      - Anomalías  → leídas directamente de anomalies_summary.csv (ya calculadas).
      - Normales   → muestra aleatoria del CSV principal, sin H3 completo.
    Evita correr predict() sobre millones de filas.
    """
    if not os.path.exists(anomalies_csv):
        print(f"[WARN] '{anomalies_csv}' no encontrado. Usando pipeline completo.")
        return load_and_predict(csv_path)

    # -- Anomalías ya calculadas --
    anom = pd.read_csv(anomalies_csv, low_memory=False,
                       dtype={"mmsi": str, "vessel_name": str})
    anom["is_anomaly"]    = np.int8(-1)
    anom["anomaly_reason"] = anom.apply(_reason_from_row, axis=1)
    print(f"[INFO] Anomalias cargadas desde summary: {len(anom):,}")

    # -- Muestra de normales del CSV principal --
    print("[INFO] Muestreando normales del CSV principal ...")
    df_raw = pd.read_csv(
        csv_path, low_memory=False,
        dtype={"mmsi": str, "vessel_name": str, "imo": str,
               "call_sign": str, "transceiver": str},
        usecols=["mmsi", "vessel_name", "base_date_time",
                 "latitude", "longitude", "sog", "cog", "heading",
                 "vessel_type", "status", "length", "width", "draft", "cargo"],
    )
    # Excluir filas que ya son anomalías (por mmsi+timestamp si disponible)
    df_raw = df_raw.sample(n=min(NORMAL_SAMPLE * 5, len(df_raw)),
                           random_state=RANDOM_SEED)

    df_raw["is_anomaly"]    = np.int8(1)
    df_raw["anomaly_reason"] = "Normal"
    df_raw["anomaly_score"]  = np.float32(0.0)
    # Campos H3 no disponibles para normales (tooltip los omite si faltan)
    normals = df_raw.sample(n=min(NORMAL_SAMPLE, len(df_raw)),
                            random_state=RANDOM_SEED)
    print(f"[INFO] Normales muestreados: {len(normals):,}")

    return pd.concat([anom, normals], ignore_index=True)


def _reason_from_row(row) -> str:
    """Infiere el motivo de anomalía a partir de las features H3."""
    try:
        sog_z = float(row.get("sog_z_hex", 0) or 0)
        if abs(sog_z) > 5:
            return "Velocidad muy desviada respecto a la zona"
        is_new = int(row.get("is_new_hex", 0) or 0)
        if is_new:
            return "Embarcacion en zona geografica poco frecuentada"
    except Exception:
        pass
    return "Comportamiento anomalo"


def load_and_predict(csv_path: str = CSV_FILE):
    """Pipeline completo (lento, ~5 min para 7 M filas)."""
    detector = AISAnomalyDetector()
    print("[INFO] Preprocesando datos AIS ...")
    df = preprocess(csv_path)           # carga + fechas + limpieza
    print("[INFO] Ejecutando deteccion de anomalias ...")
    df = detector.predict(df)           # H3 se calcula internamente
    return df


# ---------------------------------------------------------------------------
# 2. Muestra para visualización
# ---------------------------------------------------------------------------

def build_plot_df(df):
    anomalies = df[df["is_anomaly"] == -1].copy()
    normals_pool = df[df["is_anomaly"] == 1]
    normals = normals_pool.sample(
        n=min(NORMAL_SAMPLE, len(normals_pool)),
        random_state=RANDOM_SEED,
    ).copy()
    print(f"[INFO] Puntos a graficar: {len(anomalies):,} anomalos + {len(normals):,} normales")
    return anomalies, normals


# ---------------------------------------------------------------------------
# 3. Gráfico interactivo
# ---------------------------------------------------------------------------

# Campos que se muestran en el tooltip (en orden)
FEATURE_FIELDS = [
    ("h3_res7",              "h3_cell"),
    ("h3_res5",              "h3_parent"),
    ("time_band",            "time_band"),
    ("time_band_label",      "time_band_label"),
    ("context_level",        "context_level"),
    ("vtype_context_level",  "vtype_context_level"),
    ("hour_sin",             "hour_sin"),
    ("hour_cos",             "hour_cos"),
    ("day_of_week",          "day_of_week"),
    ("month",                "month"),
    ("sog",                  "sog_kn"),
    ("cog",                  "cog_deg"),
    ("heading",              "heading_deg"),
    ("status",               "status"),
    ("vessel_type",          "vessel_type"),
    ("length",               "length_m"),
    ("width",                "width_m"),
    ("draft",                "draft_m"),
    ("cargo",                "cargo"),
    ("hex_log_density",      "ctx_log_density"),
    ("context_obs_count",    "context_obs_count"),
    ("vtype_context_obs_count", "vtype_context_obs_count"),
    ("is_sparse_hex",        "is_sparse_hex"),
    ("is_new_hex",           "is_new_hex"),
    ("sog_delta_hex_med",    "sog_delta_hex_med"),
    ("sog_z_hex",            "sog_z_hex"),
    ("length_delta_hex_med", "length_delta_hex_med"),
    ("length_z_hex",         "length_z_hex"),
    ("width_delta_hex_med",  "width_delta_hex_med"),
    ("width_z_hex",          "width_z_hex"),
    ("draft_delta_hex_med",  "draft_delta_hex_med"),
    ("draft_z_hex",          "draft_z_hex"),
    ("vtype_mode_share_hex", "vtype_mode_share_hex"),
    ("is_unusual_vtype_hex", "is_unusual_vtype_hex"),
]

INT_COLS    = {"day_of_week", "month", "status", "vessel_type", "cargo",
               "is_sparse_hex", "is_new_hex", "is_unusual_vtype_hex"}
FLOAT2_COLS = {"sog", "cog", "length", "width", "draft",
               "sog_delta_hex_med", "sog_z_hex", "hex_log_density",
               "context_obs_count", "vtype_context_obs_count",
               "length_delta_hex_med", "length_z_hex",
               "width_delta_hex_med", "width_z_hex",
               "draft_delta_hex_med", "draft_z_hex",
               "vtype_mode_share_hex", "time_band"}


def plot_plotly(anomalies, normals, out_path: str = HTML_OUT) -> None:
    map_lines = load_world_lines()

    def _fmt(series, decimals: int = 3):
        if series.dtype.kind in "biufc":
            return series.round(decimals).astype(str).where(series.notna(), "N/A")
        return series.fillna("N/A").astype(str)

    def _fmt_heading(series):
        return series.apply(lambda v: "N/A" if pd.isna(v) or v == 511 else f"{int(v)}")

    def make_trace(subset, name, color, opacity, size):
        hover = (
            "<b>" + subset["vessel_name"].fillna("Desconocido") + "</b><br>" +
            "MMSI: "   + subset["mmsi"].astype(str) + "<br>" +
            "lat: "    + _fmt(subset["latitude"],  4) + "<br>" +
            "lon: "    + _fmt(subset["longitude"], 4) + "<br>" +
            "Motivo: " + subset["anomaly_reason"].fillna("Normal") + "<br>" +
            "score: "  + _fmt(subset["anomaly_score"], 4)
        )

        for col, label in FEATURE_FIELDS:
            if col not in subset.columns:
                continue
            if col == "heading":
                value = _fmt_heading(subset[col])
            elif col in INT_COLS:
                value = _fmt(subset[col], 0)
            elif col in FLOAT2_COLS:
                value = _fmt(subset[col], 2)
            else:
                value = _fmt(subset[col], 4)
            hover = hover + "<br>" + f"{label}: " + value

        return go.Scattergl(
            x=subset["longitude"], y=subset["latitude"],
            mode="markers", name=name,
            text=hover, hovertemplate="%{text}<extra></extra>",
            marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0)),
        )

    fig = go.Figure()

    for xs, ys in map_lines:
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="lines",
            line=dict(color=COLOR_MAP, width=0.8),
            hoverinfo="skip", showlegend=False,
        ))

    fig.add_trace(make_trace(normals,   "Normal (1)",   COLOR_NORMAL,  0.25, 3))
    fig.add_trace(make_trace(anomalies, "Anomalo (-1)", COLOR_ANOMALY, 0.80, 5))

    fig.update_layout(
        title=dict(text="Deteccion de Anomalias AIS — Scatter Geografico (H3 + tipo + hora)", font_size=20),
        xaxis=dict(title="Longitud", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Latitud",  showgrid=True, gridcolor="#e0e0e0",
                   scaleanchor="x", scaleratio=1),
        legend=dict(title="Clasificacion", itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="#cccccc", borderwidth=1),
        plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff",
        hovermode="closest", width=1400, height=800,
        margin=dict(l=60, r=30, t=70, b=60),
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[INFO] Grafico guardado -> {out_path}")


# ---------------------------------------------------------------------------
# 4. Pipeline principal
# ---------------------------------------------------------------------------

def main(csv_path: str = CSV_FILE, name_suffix: str | None = None,
         fast: bool = True) -> None:
    if fast and csv_path == CSV_FILE:
        df = load_fast(csv_path)
    else:
        df = load_and_predict(csv_path)
    anomalies, normals = build_plot_df(df)
    html_out = build_output_path(name_suffix)
    print("\n[INFO] Generando grafico ...")
    plot_plotly(anomalies, normals, html_out)
    print(f"\n[OK] Listo.  HTML: {html_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera scatter de anomalias AIS.")
    parser.add_argument("csv_path", nargs="?", default=CSV_FILE)
    parser.add_argument("--suffix", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--full", action="store_true",
                        help="Usar pipeline completo (lento). Por defecto usa ruta rapida.")
    args = parser.parse_args()
    main(csv_path=args.csv_path, name_suffix=args.suffix, fast=not args.full)
