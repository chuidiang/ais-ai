"""
Script para analizar cómo se usan grid_x y grid_y en el modelo Isolation Forest
"""
import joblib
from sklearn.tree import _tree
import numpy as np
import pandas as pd

# Cargar el modelo y datos
model = joblib.load('models/isolation_forest_model.joblib')

# Cargar el scaler para ver normalización
scaler = joblib.load('models/scaler.joblib')

# Información sobre grid_x y grid_y
print("[INFO] === INFORMACIÓN SOBRE GRID_X Y GRID_Y ===\n")

from load_ais_data import preprocess
df_prep = preprocess(geo_zone_mode="grid")

print(f"grid_x - Rango: [{df_prep['grid_x'].min()}, {df_prep['grid_x'].max()}]")
print(f"grid_x - Media: {df_prep['grid_x'].mean():.2f}, Std: {df_prep['grid_x'].std():.2f}")
print(f"grid_x - Valores únicos: {df_prep['grid_x'].nunique()}\n")

print(f"grid_y - Rango: [{df_prep['grid_y'].min()}, {df_prep['grid_y'].max()}]")
print(f"grid_y - Media: {df_prep['grid_y'].mean():.2f}, Std: {df_prep['grid_y'].std():.2f}")
print(f"grid_y - Valores únicos: {df_prep['grid_y'].nunique()}\n")

# Información del scaler
print(f"[INFO] === NORMALIZACIÓN (StandardScaler) ===\n")
feature_names = ['grid_x', 'grid_y', 'hour_sin', 'hour_cos', 'day_of_week', 'month',
                 'sog', 'cog', 'heading', 'status', 'vessel_type', 'length',
                 'width', 'draft', 'cargo']

for i, name in enumerate(feature_names[:2]):
    print(f"{name}: mean={scaler.mean_[i]:.4f}, scale={scaler.scale_[i]:.4f}")

# Analizar splits en todos los árboles
print(f"\n[INFO] === ANÁLISIS DE SPLITS EN TODOS LOS ÁRBOLES ===\n")

grid_x_splits = []
grid_y_splits = []

for tree_idx, estimator in enumerate(model.estimators_):
    tree = estimator.tree_
    feature = tree.feature
    threshold = tree.threshold

    for node_id in range(tree.node_count):
        if feature[node_id] != _tree.TREE_UNDEFINED:
            feat_idx = feature[node_id]
            feat_name = feature_names[feat_idx]
            thresh = threshold[node_id]

            if feat_name == 'grid_x':
                grid_x_splits.append((tree_idx, thresh))
            elif feat_name == 'grid_y':
                grid_y_splits.append((tree_idx, thresh))

print(f"Total splits con grid_x: {len(grid_x_splits)}")
print(f"Total splits con grid_y: {len(grid_y_splits)}")
print(f"Árboles que usan grid_x: {len(set(t[0] for t in grid_x_splits))}")
print(f"Árboles que usan grid_y: {len(set(t[0] for t in grid_y_splits))}\n")

print("[INFO] Primeros 8 splits con grid_x:")
for i, (tree_idx, thresh) in enumerate(grid_x_splits[:8], 1):
    print(f"  {i}. Árbol {tree_idx}: grid_x <= {thresh:.4f}")

print("\n[INFO] Primeros 8 splits con grid_y:")
for i, (tree_idx, thresh) in enumerate(grid_y_splits[:8], 1):
    print(f"  {i}. Árbol {tree_idx}: grid_y <= {thresh:.4f}")

# Análisis más profundo: ejemplo de cómo funciona el grid
print(f"\n[INFO] === EJEMPLO PRÁCTICO DEL GRID ===\n")
print("Función add_grid() calcula:")
print("  grid_x = floor((longitude - LON_MIN) / cell_deg)")
print("  grid_y = floor((latitude - LAT_MIN) / cell_deg)")
print("\nDonde:")
print("  LON_MIN = -180.0, LON_MAX = 180.0")
print("  LAT_MIN = -90.0,  LAT_MAX = 90.0")
print("  cell_deg = 0.5 (tamaño de celda en grados)\n")

# Ejemplos de puntos
print("Ejemplos:")
sample_lon = [-180, -45, 0, 45, 180]
sample_lat = [-90, -45, 0, 45, 90]
for lon in sample_lon:
    for lat in sample_lat:
        grid_x = int(np.floor((lon - (-180.0)) / 0.5))
        grid_y = int(np.floor((lat - (-90.0)) / 0.5))
        print(f"  (lon={lon:4}, lat={lat:3}) → grid_x={grid_x:3}, grid_y={grid_y:3}")
        break

print(f"\n[INFO] === CONCLUSIÓN ===\n")
print("✓ grid_x y grid_y se usan INDEPENDIENTEMENTE en los splits del árbol")
print("✓ Cada uno es una FEATURE SEPARADA en el modelo")
print("✓ Los splits pueden ser solo en grid_x, solo en grid_y, o en ambos")
print(f"✓ En este modelo: {len(grid_x_splits)} splits de grid_x, {len(grid_y_splits)} splits de grid_y")

