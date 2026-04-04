"""
plot_anomalies.py
Genera scatter plot interactivo con anomalías AIS detectadas.
Azul: normal, Rojo: anomalía.
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import shapefile

try:
    import geopandas as gpd
except Exception:
    gpd = None

from predict_realtime import AISAnomalyDetector
from load_ais_data import preprocess, CSV_FILE

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SHP_DIR = os.path.join(BASE_DIR, "shp")
WORLD_SHP = os.path.join(SHP_DIR, "world.shp")
os.makedirs(PLOTS_DIR, exist_ok=True)

HTML_OUT = os.path.join(PLOTS_DIR, "anomalies_scatter.html")
NORMAL_SAMPLE = 80_000
RANDOM_SEED = 42

COLOR_NORMAL = "#4C9BE8"
COLOR_ANOMALY = "#E84C4C"
COLOR_MAP = "#a7b2be"


def build_output_path(name_suffix: str | None = None) -> str:
    if not name_suffix:
        return HTML_OUT
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name_suffix)
    return os.path.join(PLOTS_DIR, f"anomalies_scatter_{safe}.html")


def load_world_lines(shp_path: str = WORLD_SHP):
    """Carga geometría del shapefile mundial."""
    if not os.path.exists(shp_path):
        print(f"[WARN] Shapefile no encontrado. Se omite mapa.")
        return []

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
                    lines.append(([c[0] for c in coords], [c[1] for c in coords]))
            print(f"[INFO] Mapa cargado: {len(lines):,} segmentos")
            return lines
        except Exception as exc:
            print(f"[WARN] geopandas error: {exc}")

    try:
        reader = shapefile.Reader(shp_path)
        lines = []
        for shape in reader.shapes():
            pts = shape.points
            parts = list(shape.parts) + [len(pts)]
            for i in range(len(parts) - 1):
                seg = pts[parts[i] : parts[i + 1]]
                if len(seg) >= 2:
                    lines.append(([p[0] for p in seg], [p[1] for p in seg]))
        print(f"[INFO] Mapa cargado: {len(lines):,} segmentos")
        return lines
    except Exception as exc:
        print(f"[WARN] Error loading shapefile: {exc}")
        return []


def load_and_predict(csv_path: str = CSV_FILE):
    print("[INFO] Preprocesando …")
    df = preprocess(csv_path)
    detector = AISAnomalyDetector()
    print("[INFO] Prediciendo …")
    df = detector.predict(df)
    return df


def build_plot_df(df):
    anomalies = df[df["is_anomaly"] == -1].copy()
    normals = df[df["is_anomaly"] == 1].sample(
        n=min(NORMAL_SAMPLE, (df["is_anomaly"] == 1).sum()),
        random_state=RANDOM_SEED,
    ).copy()
    print(f"[INFO] Puntos: {len(anomalies):,} anomalías + {len(normals):,} normales")
    return anomalies, normals


def plot_plotly(anomalies, normals, out_path: str = HTML_OUT) -> None:
    map_lines = load_world_lines()

    def _fmt(series, decimals: int = 3):
        if series.dtype.kind in "biufc":
            return series.round(decimals).astype(str).where(series.notna(), "N/A")
        return series.fillna("N/A").astype(str)

    def make_trace(subset, name, color, opacity, size):
        hover = (
            "<b>" + subset["vessel_name"].fillna("Unknown") + "</b><br>" +
            "MMSI: " + subset["mmsi"].astype(str) + "<br>" +
            "lat: " + _fmt(subset["latitude"], 4) + "<br>" +
            "lon: " + _fmt(subset["longitude"], 4) + "<br>" +
            "score: " + _fmt(subset["anomaly_score"], 4)
        )

        return go.Scattergl(
            x=subset["longitude"],
            y=subset["latitude"],
            mode="markers",
            name=name,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0)),
        )

    fig = go.Figure()

    for xs, ys in map_lines:
        fig.add_trace(
            go.Scattergl(
                x=xs, y=ys, mode="lines",
                line=dict(color=COLOR_MAP, width=0.8),
                hoverinfo="skip", showlegend=False,
            )
        )

    fig.add_trace(make_trace(normals, "Normal", COLOR_NORMAL, 0.25, 3))
    fig.add_trace(make_trace(anomalies, "Anomaly", COLOR_ANOMALY, 0.80, 5))

    fig.update_layout(
        title=dict(text="AIS Anomalies — Spatial Distribution", font_size=18),
        xaxis=dict(title="Longitude", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Latitude", showgrid=True, gridcolor="#e0e0e0",
                   scaleanchor="x", scaleratio=1),
        legend=dict(itemsizing="constant", bgcolor="rgba(255,255,255,0.8)"),
        plot_bgcolor="#f8f9fa", paper_bgcolor="#ffffff",
        hovermode="closest", width=1400, height=800,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[INFO] Gráfico guardado: {out_path}")


def main(csv_path: str = CSV_FILE, name_suffix: str | None = None) -> None:
    df = load_and_predict(csv_path)
    anomalies, normals = build_plot_df(df)
    html_out = build_output_path(name_suffix)
    print("[INFO] Generando gráfico …")
    plot_plotly(anomalies, normals, html_out)
    print(f"[OK] Done: {html_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AIS anomalies.")
    parser.add_argument("csv_path", nargs="?", default=CSV_FILE)
    parser.add_argument("--suffix", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    main(csv_path=args.csv_path, name_suffix=args.suffix)
