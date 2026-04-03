"""
run_comparison.py
-----------------
Ejecuta entrenamiento + grafico para los tres modos de segmentacion geografica
y guarda los HTMLs con nombres claros para comparar:

  plots/anomalies_scatter_comparacion_geo_kmeans.html
  plots/anomalies_scatter_comparacion_geo_both.html
  plots/anomalies_scatter_comparacion_geo_grid.html
"""

import importlib
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

MODES = ["kmeans", "both", "grid"]


def run_training(mode: str) -> None:
    """Re-importa train_anomaly con el modo indicado y ejecuta el pipeline."""
    import train_anomaly as ta

    # Forzar el modo y reconstruir FEATURE_COLS antes de ejecutar
    ta.GEO_ZONE_MODE = mode
    ta.FEATURE_COLS  = ta._build_feature_cols(mode)

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO  modo={mode}")
    print(f"  features: {ta.FEATURE_COLS}")
    print(f"{'='*60}\n")

    ta.main()


def run_plot(mode: str, suffix: str) -> None:
    """Re-importa plot_anomalies y genera el grafico con el sufijo indicado."""
    import plot_anomalies as pa
    import load_ais_data as la
    importlib.reload(la)   # asegurar estado limpio
    importlib.reload(pa)

    print(f"\n{'='*60}")
    print(f"  GRAFICO  modo={mode}  suffix={suffix}")
    print(f"{'='*60}\n")

    pa.main(csv_path=None, name_suffix=suffix)


for mode in MODES:
    suffix = f"comparacion_geo_{mode}"
    print(f"\n{'#'*60}")
    print(f"#  CICLO COMPLETO: {mode.upper()}")
    print(f"{'#'*60}\n")

    run_training(mode)
    run_plot(mode, suffix)

    print(f"\n[OK] Ciclo '{mode}' finalizado -> plots/anomalies_scatter_{suffix}.html\n")

print("\n" + "="*60)
print("  COMPARACION COMPLETADA")
for mode in MODES:
    suffix = f"comparacion_geo_{mode}"
    print(f"  -> plots/anomalies_scatter_{suffix}.html")
print("="*60 + "\n")

