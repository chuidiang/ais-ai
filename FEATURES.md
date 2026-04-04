# Características del Modelo AIS

## Features Utilizadas

El modelo de detección de anomalías utiliza las siguientes 9 features:

### 1. **Posición Geográfica** (2 features)
- `latitude`: Latitud en grados (-90 a 90)
- `longitude`: Longitud en grados (-180 a 180)
- Escaladas con StandardScaler

### 2. **Tipo de Barco** (1 feature)
- `vessel_type_mapped`: Tipo de embarcación categorizado (0, 10, 20, 60, 70, 80, 90)
  - 0 = Desconocido/nulo
  - 10 = Barcos cargo
  - 20 = Barcos tanque
  - 60 = Barcos pasajeros
  - 70 = Barcos pesca
  - 80 = Barcos servicios
  - 90 = Embarcaciones de otro tipo
- Escalada con StandardScaler

### 3. **Rumbo** (2 features circulares)
- `cog_sin`: Seno del Course Over Ground (rumbo)
- `cog_cos`: Coseno del Course Over Ground
- **Por qué sin/cos**: COG es una variable circular (0°-360°), por lo que usar sin/cos mantiene la estructura circular
- Escaladas con StandardScaler

### 4. **Proa** (2 features circulares)
- `heading_sin`: Seno del Heading (proa)
- `heading_cos`: Coseno del Heading
- **Por qué sin/cos**: Al igual que COG, es una variable circular
- Escaladas con StandardScaler

### 5. **Velocidad** (1 feature)
- `sog`: Speed Over Ground (velocidad en nudos)
- Variable continua
- Escalada con StandardScaler

### 6. **Estado** (1 feature)
- `status`: Estado operativo del barco
  - 0 = Under way using engine (en tránsito con motor)
  - 1 = At anchor (fondeado)
  - 2 = Not under command (sin control)
  - 3 = Restricted maneuverability (maniobra restringida)
  - 4 = Constrained by draft (restringido por calado)
  - 5 = Moored (amarrado)
  - 8 = Power driven vessel towing astern (remolcando)
  - 15 = Default/Unknown (desconocido/nulo)
- Escalada con StandardScaler
- **Imputación de nulos**: Por defecto, los valores nulos se convierten a 15 (unknown)

## Opciones de Preprocesamiento

### Opción 1: Imputación de Status (por defecto)
```bash
python train_anomaly.py
```
- Los registros con `status` nulo se convierten a 15 (unknown)
- Se entrenan con **~7.3M registros**
- ✅ Más datos para el entrenamiento
- ⚠️ Registros con datos incompletos se marcan ligeramente como anómalos

### Opción 2: Descartar Status Nulos
```bash
python train_anomaly.py --discard-missing-status
```
- Los registros con `status` nulo se **descartan** completamente
- Se entrenan con **~4.7M registros** (65% de los datos)
- ✅ Modelo más limpio sin datos incompletos
- ⚠️ Menos datos para entrenar, pero potencialmente mejor calidad

## Transformaciones Circulares (sin/cos)

Las variables `cog` (Course Over Ground) y `heading` (Proa) son ángulos que varían de 0° a 360°.

**Problema sin transformación:**
- Si usamos el ángulo directamente, 359° y 1° parecen estar muy lejos (358° de diferencia)
- Pero en realidad están a solo 2° de distancia

**Solución con sin/cos:**
- `cog_sin = sin(cog_radianes)`
- `cog_cos = cos(cog_radianes)`
- Esto convierte el ángulo en un punto en el círculo unitario
- Las distancias se preservan correctamente: 359° y 1° estarán cercanos en el espacio

## Modelo: Isolation Forest

- **n_estimators**: 100 árboles de decisión
- **max_samples**: 1024 muestras por árbol
- **contamination**: 1% (espera 1% de anomalías)
- **random_state**: 42 (reproducibilidad)

## Uso en Predicción

### En `predict_realtime.py`
Cuando predices una fila, el sistema:
1. Mapea `vessel_type` según las reglas
2. Imputa `status=15` si falta
3. Llena con 0 si faltan `sog`, `cog`, `heading`
4. Calcula sin/cos automáticamente
5. Escala todas las features
6. Realiza predicción

### En `plot_anomalies.py`
- El hover muestra las nuevas features: `status` y `sog`
- La razón de anomalía (SHAP) identifica qué feature contribuyó más

## Resultados Típicos

Con el dataset completo (7.3M registros, imputación por defecto):
- **Anomalías detectadas**: ~1% (73,367 registros)
- **Falsos positivos comunes**: 
  - Barcos con velocidad muy alta (>30 nudos)
  - Barcos con status=15 (datos incompletos)
  - Posiciones geográficas inusuales (muy norte/sur)

Con dataset filtrado (4.7M registros, descartando nulos):
- **Anomalías detectadas**: ~1% (47,564 registros)
- **Mejor calidad**: Menos falsos positivos por datos incompletos

