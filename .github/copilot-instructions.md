# Instrucciones de Copilot para ais-ai

Este archivo define reglas persistentes para cualquier asistencia automatizada en este repositorio.

## Contexto del proyecto
- Proyecto Python para deteccion de anomalias AIS.
- Scripts principales: `train_anomaly.py`, `predict_realtime.py`, `plot_anomalies.py`, `load_ais_data.py`.
- Directorios clave:
  - `data/`: entradas y resumenes
  - `models/`: artefactos de entrenamiento
  - `plots/`: salidas HTML/imagenes
  - `shp/`: capas geograficas para mapa base

## Reglas de trabajo (siempre)
1. Mantener compatibilidad con el flujo actual de entrenamiento, prediccion y graficado.
2. Hacer cambios pequenos y enfocados; evitar refactors amplios no solicitados.
3. No cambiar nombres de archivos de salida ni rutas por defecto sin peticion explicita.
4. Conservar interfaces CLI existentes (`argparse`) salvo pedido del usuario.
5. No borrar ni sobrescribir artefactos en `models/` o `plots/` sin confirmacion.
6. Preferir soluciones reproducibles y deterministas cuando aplique (semillas, orden estable).
7. Antes de proponer optimizaciones, priorizar exactitud de deteccion y trazabilidad de resultados.
8. Si una decision puede romper compatibilidad, pedir confirmacion primero.
9. Para entrenamiento de modelos usar el fichero `data/ais-data.csv` y no el sample, a menos que se indique lo contrario.
10. Para pruebas de inferencia y generación de gráficos, usar `data/ais-data-sample.csv` para evitar tiempos largos de procesamiento.

## Estilo tecnico esperado
- Python claro y mantenible.
- Validaciones defensivas para datos faltantes o columnas ausentes.
- Mensajes de log informativos en español.
- Mantener dependencias en `requirements.txt` solo cuando sea necesario.

## Prioridad al responder
1. Seguridad de datos y no regresiones funcionales.
2. Conservacion del comportamiento actual del pipeline AIS.
3. Claridad de cambios y pasos para ejecutar/validar.

