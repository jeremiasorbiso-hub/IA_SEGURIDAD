# 🔑 CÓDIGOS FUNDAMENTALES DEL PROYECTO

## 📌 RESUMEN DE COMPONENTES

El proyecto tiene **3 componentes clave** que trabajan juntos:

```
1. run_api.ps1 (PowerShell)
   ↓ Inicia el servidor
2. predict_api.py (Python/FastAPI)
   ↓ Recibe predicciones
3. SD911_AutoBot_Full.user.js (JavaScript/Tampermonkey)
   ↓ Ingresa a la página y llena campos
```

---

## 🚀 1. SCRIPT QUE INICIA TODO: `run_api.ps1`

**Archivo:** `run_api.ps1`  
**Lenguaje:** PowerShell  
**Propósito:** Lanzar el API REST y abrir el navegador

### Flujo de ejecución:

```powershell
# 1. Configura rutas y entorno
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root "venv\Scripts\python.exe"

# 2. Inicia el servidor FastAPI en segundo plano
$proc = Start-Process -FilePath $Python `
        -ArgumentList "-m uvicorn api.predict_api:app --host 0.0.0.0 --port 8000" `
        -WindowStyle Hidden -PassThru

# 3. Espera a que el API responda (verifica /health)
$healthUrl = "http://127.0.0.1:8000/health"
Invoke-WebRequest -Uri $healthUrl  # ← Espera hasta que sea 200 OK

# 4. Abre el navegador en la página del 911
Start-Process "http://10.100.32.84/SD911/"
```

**Resultado:**
- ✅ API corriendo en `http://127.0.0.1:8000`
- ✅ Navegador abre automáticamente `http://10.100.32.84/SD911/`
- ✅ Bot Tampermonkey se activa

---

## 🧠 2. API QUE HACE LAS PREDICCIONES: `api/predict_api.py`

**Archivo:** `api/predict_api.py`  
**Lenguaje:** Python 3.10  
**Framework:** FastAPI + Uvicorn  
**Propósito:** Recibir texto y devolver predicciones de IA

### Endpoint principal: `/predict`

```python
@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Recibe un texto de incidente y devuelve clasificaciones de IA
    """
    text = request.text
    
    # 1. Tokeniza el texto con BERT
    inputs = tokenizer(
        text, 
        max_length=512, 
        truncation=True, 
        return_tensors='pt'
    )
    
    # 2. Envía al modelo BERT
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], 
                       attention_mask=inputs['attention_mask'])
    
    # 3. Obtiene predicciones de las 10 cabezas
    predictions = {}
    for i, field in enumerate(OUTPUT_FIELDS):
        logits = heads[i](outputs.last_hidden_state[:, 0, :])
        probs = torch.softmax(logits, dim=1)
        
        # Mapea números a texto
        predictions[field] = map_predictions(probs, field)
    
    return {
        "incidente_id": request.incidente_id,
        "predictions": predictions,
        "best_predictions": {k: v[0] for k, v in predictions.items()},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
```

### Ejemplo de request/response:

**REQUEST:**
```json
{
    "text": "Robo con violencia en establecimiento comercial. Sustrajeron mercancía valorada en 5000 pesos.",
    "incidente_id": "TEST-001"
}
```

**RESPONSE:**
```json
{
    "incidente_id": "TEST-001",
    "predictions": {
        "cObjetivo": {"Comercio": 0.9484, "Kiosko": 0.0062, ...},
        "cMedioempleado": {"No Registra": 0.526, "Con Arma": 0.136, ...},
        "cModusoperandi": {"Asaltante": 0.4875, "No Registra": 0.217, ...},
        ...
    },
    "best_predictions": {
        "cObjetivo": "Comercio",
        "cMedioempleado": "No Registra",
        "cModusoperandi": "Asaltante",
        ...
    },
    "timestamp": "2026-01-22 12:40:23"
}
```

---

## 🤖 3. BOT QUE INGRESA A LA PÁGINA: `SD911_AutoBot_Full.user.js`

**Archivo:** `frontend/SD911_AutoBot_Full.user.js`  
**Lenguaje:** JavaScript (Tampermonkey UserScript)  
**Propósito:** Automatizar login y llenar formulario con IA

### ¿CÓMO INGRESA A LA PÁGINA?

El bot ejecuta **3 fases automáticamente**:

#### **FASE 1: LOGIN (Detecta y rellena credenciales)**

```javascript
function handleLogin() {
    // 1. Busca los campos de login
    const userField = document.querySelector("#usuario");
    const passField = document.querySelector("#password");
    const loginBtn = document.querySelector("button[type='submit']");

    // 2. Rellena con las credenciales configuradas
    userField.value = "45657263";           // Usuario
    passField.value = "911rosario";         // Contraseña

    // 3. Simula que el usuario escribió (eventos)
    userField.dispatchEvent(new Event('input', { bubbles: true }));
    passField.dispatchEvent(new Event('input', { bubbles: true }));

    // 4. Clickea el botón de entrada
    loginBtn.click();
}
```

**Resultado:** Usuario autenticado en SD911

---

#### **FASE 2: NAVEGACIÓN (Va al formulario de desagregación)**

```javascript
function handleMenu() {
    // Busca el botón que lleva al formulario
    const btnCarga = document.querySelector("button[onclick*='form911auto']");
    
    // Clickea el botón
    btnCarga.click();
    
    // Si el click no redirige, fuerza la redirección
    setTimeout(() => {
        if (!window.location.href.includes("form911auto")) {
            window.location.href = 'form911auto';
        }
    }, 1000);
}
```

**Resultado:** Página redirigida a `/form911auto` (el formulario)

---

#### **FASE 3: DESAGREGACIÓN (Llena campos con predicciones de IA)**

```javascript
function handleFormulario() {
    // 1. Busca el campo "relato" (texto del incidente)
    const relato = document.querySelector("textarea[name='relato']");
    
    // 2. Cuando el relato está listo (>10 caracteres)
    if (relato && relato.value.length > 10) {
        
        // 3. Envía el relato al API de IA
        enviarAIA(relato.value);
    }
}

function enviarAIA(texto) {
    // Hace una petición POST al API
    GM_xmlhttpRequest({
        method: "POST",
        url: "http://127.0.0.1:8000/predict",
        headers: { "Content-Type": "application/json" },
        data: JSON.stringify({ text: texto }),
        
        onload: function(response) {
            // 4. Recibe las predicciones del API
            const data = JSON.parse(response.responseText);
            
            // 5. Llena automáticamente los selects con las predicciones
            llenarSelects(data.best_predictions);
        }
    });
}

function llenarSelects(predicciones) {
    // Mapeo de campos IA → HTML
    const campos = {
        "cObjetivo": "objetivo",
        "cMedioempleado": "medio",
        "cModusoperandi": "modus",
        "cMedios_fuga": "fuga",
        "cElementos_sustraidos": "sustraido",
        "cLocalizacion": "localiz",
        "cGenero_Sexo": "sexo_genero",
        "cEdad": "edadvictima",
        "cRectificacion_Tipo": "rectificacion_tipo",
        "cRectificacion_Subtipo": "rectificacion_subtipo"
    };

    // Para cada campo IA
    for (const [keyIA, nameHTML] of Object.entries(campos)) {
        // Encuentra el <select> en la página
        const select = document.querySelector(`select[name='${nameHTML}']`);
        
        if (select) {
            const valorIA = predicciones[keyIA];
            
            // Busca la opción que coincide con la predicción
            for (let i = 0; i < select.options.length; i++) {
                if (select.options[i].text.toLowerCase().includes(valorIA.toLowerCase())) {
                    // La selecciona
                    select.selectedIndex = i;
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    break;
                }
            }
        }
    }
}
```

**Resultado:** Todos los campos del formulario llenos automáticamente ✅

---

## 🔀 FLUJO COMPLETO DE EJECUCIÓN

```
┌─────────────────────────────────────────────────────────────────────┐
│                    USUARIO EJECUTA EN TERMINAL                       │
│                                                                       │
│     PS> cd C:\Users\jorbiso\Desktop\PruebaIA\IA_SEGURIDAD           │
│     PS> .\run_api.ps1                                               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
        ┌───────────────────────────────────────────────────┐
        │      run_api.ps1 (PowerShell)                    │
        │                                                   │
        │  1. Configura Python venv                       │
        │  2. Inicia FastAPI en puerto 8000               │
        │  3. Espera /health = 200                        │
        │  4. Abre navegador → http://10.100.32.84/SD911/ │
        └───────────────────────────────────────────────────┘
                                  ↓
        ┌───────────────────────────────────────────────────┐
        │    Tampermonkey detecta la página SD911           │
        │    SD911_AutoBot_Full.user.js se ACTIVA          │
        │                                                   │
        │  FASE 1: LOGIN                                  │
        │  ├─ Busca #usuario → inserta "45657263"        │
        │  ├─ Busca #password → inserta "911rosario"     │
        │  ├─ Clickea button[type='submit']              │
        │  └─ Espera redirección                         │
        │                                                   │
        │  FASE 2: NAVEGACIÓN                            │
        │  ├─ Busca button[onclick*='form911auto']       │
        │  ├─ Clickea botón                              │
        │  └─ Espera redirección a /form911auto          │
        │                                                   │
        │  FASE 3: DESAGREGACIÓN                         │
        │  ├─ Detecta textarea[name='relato'] con texto  │
        │  ├─ POST → http://127.0.0.1:8000/predict      │
        │  └─ Llena selects con predicciones             │
        └───────────────────────────────────────────────────┘
                                  ↓
        ┌───────────────────────────────────────────────────┐
        │    api/predict_api.py (FastAPI)                 │
        │                                                   │
        │  1. Recibe JSON con texto                       │
        │  2. Tokeniza con BERT                           │
        │  3. Procesa 12 capas de transformers            │
        │  4. Pasa por 10 cabezas multi-head             │
        │  5. Devuelve predicciones + probabilidades     │
        └───────────────────────────────────────────────────┘
                                  ↓
        ┌───────────────────────────────────────────────────┐
        │    Bot recibe predicciones                       │
        │                                                   │
        │  1. Mapea: cObjetivo → "Comercio" 0.94         │
        │  2. Mapea: cMedioempleado → "No Registra"     │
        │  3. Mapea: cModusoperandi → "Asaltante"       │
        │  4. ... (8 campos más)                         │
        │  5. Selecciona opción en cada <select>        │
        │  6. Dispara eventos 'change'                  │
        └───────────────────────────────────────────────────┘
                                  ↓
        ┌───────────────────────────────────────────────────┐
        │    RESULTADO FINAL                               │
        │                                                   │
        │  ✅ Formulario completamente lleno             │
        │  ✅ 10 categorías identificadas automáticamente │
        │  ✅ Listo para guardar/enviar                  │
        └───────────────────────────────────────────────────┘
```

---

## 🔧 CONFIGURACIÓN CRÍTICA

Hay **3 puntos clave** que debes verificar:

### 1. **URL de la página (en `run_api.ps1`)**
```powershell
$SD911_URL = "http://10.100.32.84/SD911/"  # <--- CAMBIA SI ES DIFERENTE
```

### 2. **Credenciales (en `SD911_AutoBot_Full.user.js`)**
```javascript
const CONFIG = {
    USERNAME: "45657263",        # <--- TU USUARIO
    PASSWORD: "911rosario",      # <--- TU CONTRASEÑA
    API_URL: "http://127.0.0.1:8000",  # Puerto del API
};
```

### 3. **Selectors HTML (en `SD911_AutoBot_Full.user.js`)**
```javascript
const SELECTORS = {
    login_user: "#usuario",           // Campo usuario (puede variar)
    login_pass: "#password",          // Campo contraseña
    login_btn: "button[type='submit']",  // Botón de login
    menu_carga_btn: "button[onclick*='form911auto']",  // Botón al formulario
};
```

---

## ❓ PREGUNTAS FRECUENTES

**P: ¿Qué pasa si el bot no ingresa?**  
R: Revisa que:
1. Las credenciales sean correctas
2. Los selectors HTML coincidan (usa DevTools F12)
3. El API esté corriendo (`Ctrl+F12` → Network)

**P: ¿Cómo sé que el API está funcionando?**  
R: Abre en navegador: `http://127.0.0.1:8000/health`  
Debe devolver: `{"status": "healthy", "model_loaded": true}`

**P: ¿Dónde están los logs?**  
R: En la carpeta `logs/api.log` y en consola Tampermonkey (click derecho → Tampermonkey → Logs)

---

## 📊 COMPONENTES RESUMIDOS

| Componente | Archivo | Función | Lenguaje |
|-----------|---------|---------|----------|
| **Lanzador** | `run_api.ps1` | Inicia API + abre navegador | PowerShell |
| **API REST** | `api/predict_api.py` | Realiza predicciones de IA | Python |
| **Modelo IA** | `model/checkpoints/model_best.pt` | BERT multi-head entrenado | PyTorch |
| **Bot Automatización** | `frontend/SD911_AutoBot_Full.user.js` | Auto-login + llenar formulario | JavaScript |
| **Configuración** | `config.py` | Parámetros del sistema | Python |

---

**Generado:** 22 de Enero de 2026  
**Versión:** 1.0
