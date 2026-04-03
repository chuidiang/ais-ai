"""
Script para crear un mapa con el shapefile del mundo y los centros de KMeans marcados
"""
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from shapely.geometry import Point

# Cargar modelos y escalers
geo_kmeans = joblib.load('models/geo_kmeans.joblib')
coords_scaler = joblib.load('models/coords_scaler.joblib')

# Cargar datos para obtener los rangos de normalización
from load_ais_data import preprocess
print("[INFO] Preprocesando datos...")
df = preprocess(geo_zone_mode="kmeans", fit_geo_kmeans=False, fit_coords_scaler=False,
                geo_kmeans_model=geo_kmeans, coords_scaler=coords_scaler)

# Obtener los centroides del KMeans (en espacio normalizado [0,1])
centroids_norm = geo_kmeans.cluster_centers_  # Shape: (n_clusters, 2)

# Desnormalizar los centroides a coordenadas lat/lon reales
centroids_real = coords_scaler.inverse_transform(centroids_norm)
centroids_df = pd.DataFrame(
    centroids_real,
    columns=['latitude', 'longitude']
)
centroids_df['cluster_id'] = range(len(centroids_df))

print(f"[INFO] Centros de KMeans desnormalizados:")
print(centroids_df.head(10))

# Cargar shapefile del mundo
print("\n[INFO] Cargando shapefile del mundo...")
world = gpd.read_file('shp/world.shp')

# Crear figura con plotly
print("[INFO] Creando mapa...")
fig = go.Figure()

# Agregar el mapa del mundo (shapefile)
for idx, row in world.iterrows():
    if row.geometry.geom_type == 'Polygon':
        x, y = row.geometry.exterior.xy
        fig.add_trace(go.Scattergeo(
            lon=list(x),
            lat=list(y),
            mode='lines',
            line=dict(width=0.5, color='lightgray'),
            hoverinfo='skip',
            showlegend=False
        ))
    elif row.geometry.geom_type == 'MultiPolygon':
        for geom in row.geometry.geoms:
            x, y = geom.exterior.xy
            fig.add_trace(go.Scattergeo(
                lon=list(x),
                lat=list(y),
                mode='lines',
                line=dict(width=0.5, color='lightgray'),
                hoverinfo='skip',
                showlegend=False
            ))

# Agregar centros de KMeans
fig.add_trace(go.Scattergeo(
    lon=centroids_df['longitude'],
    lat=centroids_df['latitude'],
    mode='markers',
    marker=dict(
        size=8,
        color=centroids_df['cluster_id'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Cluster ID"),
        line=dict(width=1, color='white')
    ),
    text=[f"Cluster {cid}<br>Lat: {lat:.2f}<br>Lon: {lon:.2f}"
          for cid, lat, lon in zip(centroids_df['cluster_id'],
                                    centroids_df['latitude'],
                                    centroids_df['longitude'])],
    hovertemplate='%{text}<extra></extra>',
    name='KMeans Centers'
))

# Configurar el layout
fig.update_layout(
    title='Centros de KMeans Geográficos (64 Zonas)',
    geo=dict(
        projection_type='natural earth',
        showland=False,
        showocean=True,
        oceancolor='lightblue'
    ),
    height=700,
    width=1400,
    hovermode='closest'
)

# Guardar
output_path = 'plots/kmeans_centers_map.html'
fig.write_html(output_path)
print(f"\n[INFO] Mapa guardado en: {output_path}")
print(f"[INFO] Total de centros visualizados: {len(centroids_df)}")

