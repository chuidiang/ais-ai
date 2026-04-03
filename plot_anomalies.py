"""
plot_anomalies.py
-----------------
Genera un scatter plot interactivo (Plotly)
con la posición geográfica de los barcos, coloreados por 'is_anomaly'.

  Azul  (#4C9BE8) → comportamiento normal  (is_anomaly =  1)
  Rojo  (#E84C4C) → comportamiento anómalo (is_anomaly = -1)

Salidas
-------
  plots/anomalies_scatter.html   ← interactivo, abrir en navegador
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import shapefile

try:
    import geopandas as gpd
except Exception:  # geopandas es opcional
    gpd = None

from predict_realtime import AISAnomalyDetector
from load_ais_data import preprocess, CSV_FILE
from load_ais_data import GRID_SIZE_DEG, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.dirname(__file__)
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
SHP_DIR    = os.path.join(BASE_DIR, "shp")
WORLD_SHP  = os.path.join(SHP_DIR, "world.shp")
os.makedirs(PLOTS_DIR, exist_ok=True)

HTML_OUT = os.path.join(PLOTS_DIR, "anomalies_scatter.html")

# Para no saturar el grafico con millones de puntos normales
NORMAL_SAMPLE = 80_000
RANDOM_SEED = 42


def build_output_path(name_suffix: str | None = None) -> str:
    """Construye la ruta HTML de salida opcionalmente con sufijo."""
    if not name_suffix:
        return HTML_OUT

    safe_suffix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name_suffix)
    return os.path.join(PLOTS_DIR, f"anomalies_scatter_{safe_suffix}.html")

COLOR_NORMAL  = "#4C9BE8"   # azul
COLOR_ANOMALY = "#E84C4C"   # rojo
COLOR_MAP     = "#a7b2be"   # gris para contorno de mapa
COLOR_GRID    = "#FF3333"   # rojo para cuadricula de segmentacion


def load_world_lines(shp_path: str = WORLD_SHP):
    """
    Carga geometria de world.shp y la convierte a listas de segmentos
    (x, y) para trazar contornos del mapa.
    """


def load_grid_lines(df_all, grid_size_deg: float = GRID_SIZE_DEG,
                     lat_min: float = LAT_MIN, lon_min: float = LON_MIN):
    """
    Genera líneas de cuadrícula basadas en los grid_x y grid_y únicos
    que realmente contienen datos. Visualiza la segmentación exacta del algoritmo.
    """
    lines = []

    # Obtener grid_x y grid_y únicos
    unique_grid_x = sorted(df_all["grid_x"].unique())
    unique_grid_y = sorted(df_all["grid_y"].unique())

    if not unique_grid_x or not unique_grid_y:
        return lines

    min_gx = unique_grid_x[0]
    max_gx = unique_grid_x[-1]
    min_gy = unique_grid_y[0]
    max_gy = unique_grid_y[-1]

    # Convertir límites de grid a lon/lat
    lon_min_data = lon_min + min_gx * grid_size_deg
    lon_max_data = lon_min + (max_gx + 1) * grid_size_deg
    lat_min_data = lat_min + min_gy * grid_size_deg
    lat_max_data = lat_min + (max_gy + 1) * grid_size_deg

    # Líneas verticales (en cada límite de grid_x)
    for gx in unique_grid_x:
        lon = lon_min + gx * grid_size_deg
        lines.append({"x": [lon, lon], "y": [lat_min_data, lat_max_data]})

    # Línea vertical del borde derecho
    lon = lon_min + (max_gx + 1) * grid_size_deg
    lines.append({"x": [lon, lon], "y": [lat_min_data, lat_max_data]})

    # Líneas horizontales (en cada límite de grid_y)
    for gy in unique_grid_y:
        lat = lat_min + gy * grid_size_deg
        lines.append({"x": [lon_min_data, lon_max_data], "y": [lat, lat]})

    # Línea horizontal del borde superior
    lat = lat_min + (max_gy + 1) * grid_size_deg
    lines.append({"x": [lon_min_data, lon_max_data], "y": [lat, lat]})

    return lines


def load_world_lines(shp_path: str = WORLD_SHP):
    """
    Carga geometria de world.shp y la convierte a listas de segmentos
    (x, y) para trazar contornos del mapa.
    """
    if not os.path.exists(shp_path):
        print(f"[WARN] No se encontro shapefile base en '{shp_path}'. Se omite mapa de fondo.")
        return []

    # 1) Intento preferente con geopandas (mas robusto para SHP variados)
    if gpd is not None:
        try:
            gdf = gpd.read_file(shp_path)
            lines = []
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue

                boundary = geom.boundary
                geoms = getattr(boundary, "geoms", [boundary])
                for line in geoms:
                    coords = list(line.coords)
                    if len(coords) < 2:
                        continue
                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]
                    lines.append((xs, ys))

            print(f"[INFO] Mapa base cargado con geopandas: {len(lines):,} segmentos desde '{shp_path}'")
            return lines
        except Exception as exc:
            print(f"[WARN] geopandas no pudo leer '{shp_path}': {exc}")

    # 2) Fallback con pyshp
    try:
        reader = shapefile.Reader(shp_path)
        lines = []
        for shape in reader.shapes():
            pts = shape.points
            if not pts:
                continue

            parts = list(shape.parts) + [len(pts)]
            for i in range(len(parts) - 1):
                start, end = parts[i], parts[i + 1]
                part_pts = pts[start:end]
                if len(part_pts) < 2:
                    continue
                xs = [p[0] for p in part_pts]
                ys = [p[1] for p in part_pts]
                lines.append((xs, ys))

        print(f"[INFO] Mapa base cargado con pyshp: {len(lines):,} segmentos desde '{shp_path}'")
        return lines
    except Exception as exc:
        print(f"[WARN] No se pudo leer shapefile '{shp_path}' con geopandas/pyshp: {exc}")
        return []


# ---------------------------------------------------------------------------
# 1. Carga de datos y predicción
# ---------------------------------------------------------------------------

def load_and_predict(csv_path: str = CSV_FILE):
    print("[INFO] Preprocesando datos AIS ...")
    df = preprocess(csv_path)

    detector = AISAnomalyDetector()
    print("[INFO] Ejecutando deteccion de anomalias ...")
    df = detector.predict(df)
    return df


# ---------------------------------------------------------------------------
# 2. Muestra representativa para visualización
# ---------------------------------------------------------------------------

def build_plot_df(df):
    """
    Separa anomalías (todas) y normales (muestra aleatoria).
    Devuelve un DataFrame reducido listo para graficar.
    """
    anomalies = df[df["is_anomaly"] == -1].copy()
    normals   = df[df["is_anomaly"] ==  1].sample(
        n=min(NORMAL_SAMPLE, (df["is_anomaly"] == 1).sum()),
        random_state=RANDOM_SEED,
    ).copy()

    print(
        f"[INFO] Puntos a graficar: "
        f"{len(anomalies):,} anomalos + {len(normals):,} normales"
    )
    return anomalies, normals


# ---------------------------------------------------------------------------
# 3. Gráfico interactivo con Plotly
# ---------------------------------------------------------------------------

def plot_plotly(anomalies, normals, out_path: str = HTML_OUT) -> None:
    """Scatter interactivo con tooltip enriquecido. Guarda como HTML."""
    map_lines = load_world_lines()

    def _fmt(series, decimals: int = 3):
        """Formatea series numéricas/texto con nulos legibles para tooltip."""
        if series.dtype.kind in "biufc":
            out = series.round(decimals).astype(str)
            return out.where(series.notna(), "N/A")
        return series.fillna("N/A").astype(str)

    def _fmt_heading(series):
        """AIS heading=511 significa no disponible."""
        return series.apply(lambda v: "N/A" if pd.isna(v) or v == 511 else f"{int(v)}")

    feature_fields = [
        ("lat_norm", "lat_norm"),
        ("lon_norm", "lon_norm"),
        ("grid_x", "grid_x"),
        ("grid_y", "grid_y"),
        ("hour_sin", "hour_sin"),
        ("hour_cos", "hour_cos"),
        ("day_of_week", "day_of_week"),
        ("month", "month"),
        ("sog", "sog_kn"),
        ("cog", "cog_deg"),
        ("heading", "heading_deg"),
        ("status", "status"),
        ("vessel_type", "vessel_type"),
        ("length", "length_m"),
        ("width", "width_m"),
        ("draft", "draft_m"),
        ("cargo", "cargo"),
    ]

    def make_trace(subset, name, color, opacity, size):
        heading_text = _fmt_heading(subset["heading"]) if "heading" in subset else pd.Series(["N/A"] * len(subset))

        # Campos base visibles primero
        hover = (
            "<b>" + subset["vessel_name"].fillna("Desconocido") + "</b><br>" +
            "MMSI: "        + subset["mmsi"].astype(str) + "<br>" +
            "lat: "         + _fmt(subset["latitude"], 4) + "<br>" +
            "lon: "         + _fmt(subset["longitude"], 4) + "<br>" +
            "Motivo: "      + subset["anomaly_reason"].fillna("Normal") + "<br>" +
            "score: "       + _fmt(subset["anomaly_score"], 4)
        )

        # Añadir todas las features del modelo con su valor
        for col, label in feature_fields:
            if col not in subset.columns:
                continue
            if col == "heading":
                value = heading_text
            elif col in {"grid_x", "grid_y", "day_of_week", "month", "status", "vessel_type", "cargo"}:
                value = _fmt(subset[col], 0)
            elif col in {"sog", "cog", "length", "width", "draft"}:
                value = _fmt(subset[col], 2)
            elif col in {"lat_norm", "lon_norm", "hour_sin", "hour_cos"}:
                value = _fmt(subset[col], 4)
            else:
                value = _fmt(subset[col], 3)
            hover = hover + "<br>" + f"{label}: " + value

        return go.Scattergl(
            x          = subset["longitude"],
            y          = subset["latitude"],
            mode       = "markers",
            name       = name,
            text       = hover,
            hovertemplate = "%{text}<extra></extra>",
            marker     = dict(
                color   = color,
                size    = size,
                opacity = opacity,
                line    = dict(width=0),
            ),
        )

    fig = go.Figure()


    # Dibujar mapa base primero para dejar los barcos por encima
    if map_lines:
        for xs, ys in map_lines:
            fig.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=COLOR_MAP, width=0.8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.add_trace(make_trace(normals,   "Normal (1)",   COLOR_NORMAL,  0.25, 3))
    fig.add_trace(make_trace(anomalies, "Anomalo (-1)", COLOR_ANOMALY, 0.80, 5))

    fig.update_layout(
        title = dict(
            text      = "Deteccion de Anomalias AIS — Scatter Geografico",
            font_size = 20,
        ),
        xaxis = dict(title="Longitud", showgrid=True, gridcolor="#e0e0e0"),
        yaxis = dict(title="Latitud",  showgrid=True, gridcolor="#e0e0e0",
                     scaleanchor="x", scaleratio=1),
        legend = dict(
            title     = "Clasificacion",
            itemsizing= "constant",
            bgcolor   = "rgba(255,255,255,0.8)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        plot_bgcolor  = "#f8f9fa",
        paper_bgcolor = "#ffffff",
        hovermode     = "closest",
        width  = 1400,
        height = 800,
        margin = dict(l=60, r=30, t=70, b=60),
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[INFO] Grafico interactivo guardado -> {out_path}")

# ---------------------------------------------------------------------------
# 4. Pipeline principal
# ---------------------------------------------------------------------------

def main(csv_path: str = CSV_FILE, name_suffix: str | None = None) -> None:
    df = load_and_predict(csv_path)
    anomalies, normals = build_plot_df(df)
    html_out = build_output_path(name_suffix)

    print("\n[INFO] Generando graficos ...")
    plot_plotly(anomalies, normals, html_out)

    print("\n[OK] Listo.")
    print(f"     HTML : {html_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera scatter plots de anomalias AIS.")
    parser.add_argument("csv_path", nargs="?", default=CSV_FILE, help="Ruta CSV AIS de entrada.")
    parser.add_argument(
        "--suffix",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Sufijo para versionar el nombre de los ficheros de salida.",
    )
    args = parser.parse_args()
    main(csv_path=args.csv_path, name_suffix=args.suffix)
