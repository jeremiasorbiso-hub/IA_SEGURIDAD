# 🚀 IA SEGURIDAD - Sistema de Clasificación Inteligente con Aprendizaje Continuo

Sistema de Inteligencia Artificial para clasificación automática de incidentes de seguridad utilizando **BERT en español** con capacidad de **aprendizaje continuo** a partir de correcciones de usuarios + **Automatización Completa con Bot de Navegación**.

---

## 🎯 ¿Qué Hace Este Sistema?

Transforma tu base histórica de incidentes (Excel con 230k registros) en un **cerebro digital** que:
- ✅ Clasifica automáticamente nuevos incidentes en 10 categorías
- ✅ Aprende de las correcciones de los usuarios
- ✅ Se actualiza automáticamente cada semana sin intervención manual
- ✅ Nunca olvida el conocimiento anterior (evita "olvido catastrófico")
- 🤖 **NUEVO:** Bot de navegación automatizada (Login → Consulta IA → Llenado de formulario)

---

## 📋 Estructura del Proyecto

```
IA_SEGURIDAD/
├── data/
│   ├── raw/                    # Datos originales
│   │   └── base_original.xlsx  # Tu Excel histórico (230k registros)
│   └── processed/              # Datos procesados
│       ├── dataset_v1.jsonl    # Dataset en formato de entrenamiento
│       ├── dataset_full.jsonl  # Dataset fusionado (histórico + feedback)
│       ├── label_maps.json     # Mapeo de categorías (texto -> ID)
│       └── feedback_buffer.csv # Correcciones de la semana
│
├── model/checkpoints/          # Modelos entrenados
│   ├── model_best.pt           # Mejor modelo (multi-head BERT)
│   └── model_config.json       # Configuración del modelo
│
├── models/                     # Backups automáticos
│   └── backups/                
│
├── training/                   # Motor de entrenamiento
│   └── train_model.py          # Script de entrenamiento multi-head
│
├── scripts/
│   ├── prepare_dataset.py      # FASE 1: Preparación inicial del Excel
│   ├── data_merger.py          # FASE 3: Fusión de datos (histórico + feedback)
│   ├── train_model.py          # Script alternativo de entrenamiento
│   └── weekly_retrain.py       # Orquestador de reentrenamiento semanal
│
├── api/
│   └── predict_api.py          # FASE 2: API REST para predicciones y feedback
│
├── frontend/                   # 🤖 Bot de automatización web
│   ├── SD911_AutoBot_Full.user.js  # Tampermonkey userscript completo
│   ├── INSTALACION_BOT.md      # Guía de instalación del bot
│   └── INTEGRACION_COMPLETA.md # Guía de integración end-to-end
│
├── Dockerfile                  # Imagen Docker con PyTorch + CUDA
├── docker-compose.yml          # Orquestación de contenedores
├── .dockerignore              # Optimización de builds
├── logs/                       # Logs de todas las operaciones
├── requirements.txt            # Dependencias de Python
└── README.md                   # Este archivo
```

---

## 🚀 Instalación Rápida

### Opción A: Docker (Recomendado ⭐)

```powershell
# 1. Construir imagen
docker-compose build

# 2. Preparar dataset (coloca tu Excel en data/raw/base_original.xlsx)
docker-compose run --rm ia_seguridad python scripts/prepare_dataset.py

# 3. Entrenar modelo (2-4 horas con GPU)
docker-compose run --rm ia_seguridad python training/train_model.py

# 4. Levantar API
docker-compose up -d
```

**✅ La API estará en:** `http://localhost:8000`  
**📖 Guía completa:** [DOCKER_GUIDE.md](DOCKER_GUIDE.md)

---

### Opción B: Instalación Local

### Requisitos Previos
- Python 3.8 o superior
- (Opcional) GPU NVIDIA con CUDA para entrenamiento acelerado

### 1. Instalar Dependencias

```powershell
# Crear entorno virtual (recomendado)
python -m venv venv
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

**Para GPU (opcional pero muy recomendado):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Preparar Datos

Coloca tu archivo Excel en:
```
data/raw/base_original.xlsx
```

El Excel debe contener:
- Una columna con texto del incidente (ej: `Texto`, `Descripcion`, `Incidente`)
- Columnas de categorías (ej: `Delito`, `Lugar`, `Hora`, etc.)

---

## 📖 Guía de Uso - Las 3 Fases

### 🎬 FASE 1: El "Big Bang" (Día 0)
**Objetivo:** Transformar tu Excel histórico en el primer cerebro digital

#### Paso 1: Preparar el Dataset
```powershell
python scripts/prepare_dataset.py
```

**¿Qué hace?**
- ✅ Lee `base_original.xlsx`
- ✅ Limpia datos (elimina vacíos, duplicados)
- ✅ Detecta automáticamente columnas de texto y categorías
- ✅ Crea mapeos de etiquetas (texto → números)
- ✅ Exporta a `dataset_v1.jsonl`

**Salidas:**
- `data/processed/dataset_v1.jsonl` (formato de entrenamiento)
- `data/processed/label_maps.json` (diccionario de categorías)
- `logs/prepare_dataset.log`

#### Paso 2: Entrenar el Modelo

**Opción A: Motor de Entrenamiento Multi-Head (Recomendado)**
```powershell
python training/train_model.py
```

**Opción B: Script Alternativo**
```powershell
python scripts/train_model.py
```

**¿Qué hace?**
- ✅ Carga BERT en español (`dccuchile/bert-base-spanish-wwm-cased`)
- ✅ Arquitectura multi-head: Un clasificador por cada categoría
- ✅ Entrena en tus 230k registros históricos
- ✅ Guarda el mejor modelo en `model/checkpoints/model_best.pt`

**Tiempo estimado:**
- Con GPU: 2-4 horas
- Sin GPU: 12-24 horas

**Salidas:**
- `model/checkpoints/model_best.pt` (cerebro entrenado)
- `model/checkpoints/model_config.json` (configuración)
- `model/checkpoints/training_history.json` (historial de entrenamiento)
- `logs/training.log`

**¿Qué hace?**
- ✅ Carga BERT en español (`dccuchile/bert-base-spanish-wwm-uncased`)
- ✅ Entrena en tus 230k registros históricos
- ✅ Aprende a predecir las 10 categorías simultáneamente
- ✅ Guarda el mejor modelo en `models/model_best.pt`

**Tiempo estimado:**
- Con GPU: 2-4 horas
- Sin GPU: 12-24 horas

**Salidas:**
- `models/model_best.pt` (cerebro entrenado)
- `logs/training.log`

---

### 💼 FASE 2: Operación Diaria (Lunes a Viernes)
**Objetivo:** Usar la IA en producción y capturar correcciones

#### Paso 1: Iniciar la API
```powershell
python api/predict_api.py
```

La API estará disponible en: `http://localhost:8000`

#### Paso 2: Hacer Predicciones

**Documentación interactiva:** `http://localhost:8000/docs`

**Ejemplo de predicción:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Robo con violencia en tienda comercial. Sustrajeron mercancía por $5000",
    "incidente_id": "INC-2026-001234"
  }'
```

**Respuesta:**
```json
{
  "incidente_id": "INC-2026-001234",
  "predictions": {
    "Delito": {"Robo": 0.85, "Hurto": 0.10, "Asalto": 0.05},
    "Lugar": {"Comercio": 0.92, "Via_Publica": 0.08}
  },
  "best_predictions": {
    "Delito": "Robo",
    "Lugar": "Comercio"
  },
  "timestamp": "2026-01-21 10:30:45"
}
```

#### Paso 3: Enviar Feedback (Correcciones)

**Cuando el usuario corrige:**
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "incidente_id": "INC-2026-001234",
    "text": "Robo con violencia...",
    "predicted_categories": {"Delito": "Hurto"},
    "corrected_categories": {"Delito": "Robo"}
  }'
```

**Respuesta:**
```json
{
  "status": "success",
  "message": "Feedback guardado correctamente",
  "corrections_count": 1
}
```

**¿Qué pasa con las correcciones?**
- Se guardan en `data/processed/feedback_buffer.csv`
- **NO se modifica** el Excel original (seguridad)
- Se acumulan durante la semana para el reentrenamiento dominical

---

### 🔄 FASE 3: Evolución (Domingo por la noche)
**Objetivo:** Reentrenar con conocimiento histórico + feedback de la semana

#### Reentrenamiento Automático
```powershell
python scripts/weekly_retrain.py
```

**¿Qué hace automáticamente?**

1. **Backup** del modelo actual
2. **Fusión** de datos:
   - `base_original.xlsx` (230k registros históricos)
   - `feedback_buffer.csv` (200 correcciones de la semana)
   - Resultado: `dataset_full.jsonl` (230,200 registros)
3. **Reentrenamiento** con datos fusionados
4. **Validación** del nuevo modelo
5. **Archivado** del feedback procesado
6. **Reporte** de la operación

**Tiempo estimado:** 2-4 horas con GPU

**Salidas:**
- `models/model_best.pt` (modelo actualizado)
- `models/backups/backup_YYYYMMDD_HHMMSS/` (respaldo automático)
- `data/processed/feedback_archive/` (feedback procesado)
- `logs/retrain_reports/report_YYYYMMDD_HHMMSS.txt`

---

## 🤖 Automatización con Programador de Tareas

### Windows (Task Scheduler)

Crear una tarea programada para ejecutar el reentrenamiento cada domingo a las 2 AM:

```powershell
# Crear tarea
schtasks /create /tn "IA_Seguridad_Retrain" /tr "C:\ruta\a\venv\Scripts\python.exe C:\ruta\a\scripts\weekly_retrain.py" /sc weekly /d SUN /st 02:00
```

### Linux/Mac (Cron)

```bash
# Editar crontab
crontab -e

# Agregar línea (cada domingo a las 2 AM)
0 2 * * 0 /ruta/a/venv/bin/python /ruta/a/scripts/weekly_retrain.py
```

---

## 📊 Endpoints de la API

### GET `/`
Información general del sistema

### GET `/health`
Estado del sistema (modelo cargado, feedback disponible)

### POST `/predict`
Predice categorías para un incidente
- **Input:** `{text, incidente_id?}`
- **Output:** Predicciones con probabilidades

### POST `/feedback`
Guarda corrección del usuario
- **Input:** `{text, predicted_categories, corrected_categories}`
- **Output:** Confirmación

### GET `/stats`
Estadísticas del feedback acumulado

---

## 🔧 Configuración Avanzada

### Ajustar Hiperparámetros de Entrenamiento

Editar [scripts/train_model.py](scripts/train_model.py):

```python
config = {
    'bert_model': 'dccuchile/bert-base-spanish-wwm-uncased',
    'max_length': 512,        # Longitud máxima del texto
    'batch_size': 8,          # Aumentar si tienes más RAM/GPU
    'learning_rate': 2e-5,    # Tasa de aprendizaje
    'epochs': 3,              # Número de épocas (aumentar para mejor precisión)
}
```

### Cambiar Modelo BERT

Alternativas de modelos en español:
```python
# Opción 1 (actual)
'bert_model': 'dccuchile/bert-base-spanish-wwm-uncased'

# Opción 2: RoBERTa español
'bert_model': 'PlanTL-GOB-ES/roberta-base-bne'

# Opción 3: Bertin
'bert_model': 'bertin-project/bertin-roberta-base-spanish'
```

---

## 🛡️ Seguridad y Respaldos

### Protección del Excel Original
- ✅ El Excel **NUNCA** se modifica automáticamente
- ✅ Correcciones se guardan en CSV separado
- ✅ Fusión ocurre solo en memoria RAM durante reentrenamiento

### Respaldos Automáticos
- ✅ Cada reentrenamiento crea backup del modelo anterior
- ✅ Se mantienen los últimos 5 backups
- ✅ Restauración automática si falla el reentrenamiento

### Logs Completos
Todas las operaciones se registran en `logs/`:
- `prepare_dataset.log`
- `training.log`
- `api.log`
- `weekly_retrain.log`
- `data_merger.log`

---

## 📈 Monitoreo y Métricas

### Ver Feedback Acumulado
```python
import pandas as pd
df = pd.read_csv('data/processed/feedback_buffer.csv')
print(f"Correcciones esta semana: {len(df)}")
print(df['corrections'].value_counts())
```

### Verificar Estado del Modelo
```python
import torch
checkpoint = torch.load('models/model_best.pt', map_location='cpu')
print(f"Categorías: {checkpoint['category_names']}")
print(f"Clases por categoría: {checkpoint['num_labels_per_category']}")
```

---

## ❓ Preguntas Frecuentes

### ¿Cuánto espacio en disco necesito?
- Excel original: ~50 MB
- Dataset JSONL: ~100 MB
- Modelo entrenado: ~500 MB
- **Total recomendado:** 2-3 GB libres

### ¿Puedo usar CPU sin GPU?
Sí, pero el entrenamiento será mucho más lento (12-24 horas vs 2-4 horas).

### ¿Qué pasa si el reentrenamiento falla?
El sistema restaura automáticamente el modelo anterior desde el backup.

### ¿Puedo cambiar las categorías?
Sí, pero requiere reentrenar desde cero. El sistema detecta automáticamente las columnas del Excel.

### ¿Cómo agrego más datos históricos?
Actualiza `base_original.xlsx` y ejecuta `weekly_retrain.py` manualmente.

---

## 🐛 Solución de Problemas

### Error: "No module named 'transformers'"
```powershell
pip install -r requirements.txt
```

### Error: "No se encontró el archivo Excel"
Verifica que `data/raw/base_original.xlsx` existe

### Error: "CUDA out of memory"
Reduce `batch_size` en [scripts/train_model.py](scripts/train_model.py):
```python
'batch_size': 4,  # O 2 si persiste el error
```

### La API no responde
Verifica que el modelo está entrenado:
```powershell
dir models\model_best.pt
```

---

## 📞 Soporte y Contribuciones

### Estructura de Logs
Si necesitas ayuda, comparte los logs relevantes de `logs/`

### Mejoras Futuras
- [ ] Dashboard web para visualizar estadísticas
- [ ] Exportar reportes automáticos en PDF
- [ ] Integración con bases de datos SQL
- [ ] Multi-idioma (inglés, portugués)
- [ ] Validación cruzada para métricas de precisión

---

## 📄 Licencia

Este proyecto es de uso interno. Consulta con el equipo legal antes de distribuir.

---

## 🎓 Referencias Técnicas

- **BERT:** [Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- **Transformers:** [Hugging Face](https://huggingface.co/docs/transformers)
- **FastAPI:** [Documentación oficial](https://fastapi.tiangolo.com/)
- **PyTorch:** [pytorch.org](https://pytorch.org/)
- **Tampermonkey:** [tampermonkey.net](https://www.tampermonkey.net/)

---

## 🚀 Roadmap del Proyecto

- [x] FASE 1: Preparación y entrenamiento inicial
- [x] FASE 2: API de predicción y captura de feedback
- [x] FASE 3: Reentrenamiento automático semanal
- [x] **FASE 4: Bot de automatización web (End-to-End)**
- [ ] FASE 5: Dashboard de métricas y monitoreo
- [ ] FASE 6: Integración con sistema de gestión de incidentes

---

## 🤖 Automatización Completa (End-to-End)

El proyecto incluye un **bot de navegación Tampermonkey** que automatiza completamente el flujo de trabajo:

**Login → Menú → Formulario → Consulta IA → Llenado Automático → (Opcional) Guardar**

### Características del Bot:
- ✅ **Auto-Login:** Credenciales configurables
- ✅ **Navegación automática:** De login a menú a formulario sin intervención
- ✅ **Consulta IA en tiempo real:** Lee el relato y llama a la API local
- ✅ **Llenado inteligente:** Mapea predicciones a los campos del formulario
- ✅ **Modo semi-automático:** Revisar antes de guardar (recomendado)
- ✅ **Modo automático:** Guardado completamente automático
- ✅ **Feedback integrado:** Captura correcciones para reentrenamiento
- ✅ **UI visual:** Barra de estado y panel de control

### Instalación Rápida:
```bash
# 1. Instalar Tampermonkey en tu navegador
# Chrome: https://chrome.google.com/webstore/detail/tampermonkey/
# Firefox: https://addons.mozilla.org/firefox/addon/tampermonkey/

# 2. Copiar el script
# Abrir Tampermonkey → Dashboard → Crear nuevo script
# Copiar contenido de: frontend/SD911_AutoBot_Full.user.js

# 3. Configurar credenciales en el script
USERNAME: "tu_usuario",
PASSWORD: "tu_contraseña"

# 4. Navegar a la URL del sistema
# http://10.100.32.84/SD911/login
```

📖 **Guía completa:** Ver [frontend/INSTALACION_BOT.md](frontend/INSTALACION_BOT.md) y [frontend/INTEGRACION_COMPLETA.md](frontend/INTEGRACION_COMPLETA.md)
```
- [ ] FASE 6: Alertas automáticas para patrones anómalos

---

**¡Sistema listo para transformar tus 230k incidentes históricos en inteligencia accionable! 🎉**
