# BovWeight CR — Microservicio de Estimación de Peso Bovino

Microservicio en Python (Flask + YOLOv8) para estimar el peso de ganado bovino a partir de fotografías.

## Requisitos previos

- Python 3.10 o superior
- pip (viene con Python)
- Git

## Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd bovweight-ml
```

### 2. Crear entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> Deben ver `(venv)` al inicio de la línea de comandos.

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Esto tarda varios minutos porque descarga PyTorch y YOLOv8.

### 4. Descargar el modelo YOLOv8

**Windows:**
```powershell
mkdir models
python -c "from ultralytics import YOLO; YOLO('yolov8m-seg.pt'); print('Modelo descargado')"
move yolov8m-seg.pt models\
```

**Mac / Linux:**
```bash
mkdir -p models
python -c "from ultralytics import YOLO; YOLO('yolov8m-seg.pt'); print('Modelo descargado')"
mv yolov8m-seg.pt models/
```

### 5. Ejecutar el microservicio

```bash
python app.py
```

Deben ver:
```
INFO:services.detector:Modelo cargado: models/yolov8m-seg.pt
 * Running on http://0.0.0.0:5000
```

### 6. Verificar que funciona

Abrir en el navegador:
```
http://127.0.0.1:5000/api/health
```

Debe responder:
```json
{"model_loaded": true, "status": "ok"}
```

## Conexión con Laravel

En el archivo `.env` del proyecto Laravel agregar:

```
ML_SERVICE_URL=http://127.0.0.1:5000
```

Para verificar la conexión desde Laravel:
```
http://127.0.0.1:8000/api/estimacion/health
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/health` | Verificar que el servicio está corriendo |
| POST | `/api/estimate` | Estimar peso de una imagen |
| POST | `/api/estimate/batch` | Estimar con múltiples fotos del mismo animal |
| POST | `/api/debug` | Debug: devuelve imagen con detecciones dibujadas |

### Ejemplo de uso

**Con curl (Windows PowerShell):**
```powershell
curl.exe -X POST http://127.0.0.1:8000/api/estimacion/estimar -F "image=@ruta\a\foto.jpg" -F "ganado_id=1" -F "breed=brahman" -H "Accept: application/json"
```

**Parámetros:**
- `image` (requerido): foto del animal en formato JPG, JPEG o PNG
- `ganado_id` (requerido): ID del animal en la base de datos
- `breed` (opcional): raza del animal — `brahman`, `cebu`, `criollo` o `default`
- `reference_length_cm` (opcional): largo del objeto de referencia en cm (default: 100)

## Protocolo de fotografía

Para obtener la mejor estimación posible:

1. Tomar la foto de **perfil** (vista lateral del animal)
2. El animal debe estar **de pie**, en posición natural
3. Todo el cuerpo debe ser visible (de cabeza a cola)
4. Colocar un **palo o tabla de 1 metro** visible al lado del animal
5. El palo debe estar en el **mismo plano** que el animal (no más adelante ni más atrás)
6. Distancia recomendada: **3-5 metros** del animal
7. Buena iluminación, evitar contraluz

## Estructura del proyecto

```
bovweight-ml/
├── app.py                      # Servidor Flask (punto de entrada)
├── requirements.txt            # Dependencias de Python
├── models/
│   └── yolov8m-seg.pt          # Modelo YOLOv8 con segmentación
└── services/
    ├── __init__.py
    ├── detector.py             # Detección de ganado con YOLO
    ├── measurement.py          # Medición corporal (píxeles → cm)
    └── weight_estimator.py     # Estimación de peso (fórmula barrimétrica)
```

## Cómo funciona

1. **YOLOv8** detecta y segmenta al animal en la foto (genera la silueta exacta)
2. **OpenCV** mide las dimensiones del animal en píxeles (largo, alto, ancho del pecho)
3. Si hay **objeto de referencia**, convierte píxeles a centímetros reales
4. La **fórmula barrimétrica** calcula el peso: `Peso = (PT² × L) / K`
   - PT = perímetro torácico estimado
   - L = largo del cuerpo
   - K = constante calibrada por raza

## Notas importantes

- El microservicio debe estar corriendo **al mismo tiempo** que Laravel
- No requiere internet para funcionar (todo es local)
- El modelo YOLOv8 usa ~2 GB de RAM
- La primera estimación tarda más porque carga el modelo en memoria
- **La estimación es aproximada** y no sustituye una báscula oficial

## Solución de problemas

**"No module named 'flask'"**: No activaron el entorno virtual. Correr `.\venv\Scripts\Activate` (Windows) o `source venv/bin/activate` (Mac/Linux).

**"Error cargando modelo"**: Verificar que el archivo `models/yolov8m-seg.pt` existe. Si no, volver a ejecutar el paso 4.

**"No se pudo conectar con el servicio de estimación" (desde Laravel)**: El microservicio no está corriendo. Abrir otra terminal y ejecutar `python app.py`.

**Puerto 5000 ocupado**: Cambiar el puerto con la variable de entorno `PORT=5001 python app.py`.