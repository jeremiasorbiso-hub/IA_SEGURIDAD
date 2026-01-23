# 📚 GUÍA RÁPIDA - CÓDIGOS FUNDAMENTALES DEL PROYECTO

## 🎯 VERSIÓN SUPER SIMPLIFICADA

### Los 3 archivos que hacen que TODO funcione:

---

## 1️⃣ `run_api.ps1` - INICIA TODO

**¿Qué hace?** Lanza el API y abre el navegador

```powershell
# Ejecuta esto en PowerShell:
cd C:\Users\jorbiso\Desktop\PruebaIA\IA_SEGURIDAD
.\run_api.ps1

# Resultado:
# ✅ API inicia en puerto 8000
# ✅ Navegador abre http://10.100.32.84/SD911/
# ✅ Bot Tampermonkey se activa automáticamente
```

**Código clave:**
```powershell
# Inicia FastAPI
$proc = Start-Process -FilePath $Python `
        -ArgumentList "-m uvicorn api.predict_api:app --port 8000"

# Espera que el API esté listo
Invoke-WebRequest -Uri "http://127.0.0.1:8000/health"

# Abre el navegador
Start-Process "http://10.100.32.84/SD911/"
```

---

## 2️⃣ `api/predict_api.py` - HACE LAS PREDICCIONES

**¿Qué hace?** Recibe texto y devuelve predicciones de IA

```python
# Recibe esto:
{
    "text": "Robo en comercio con arma",
    "incidente_id": "INC-001"
}

# Devuelve esto:
{
    "best_predictions": {
        "cObjetivo": "Comercio",           ✅
        "cMedioempleado": "No Registra",   ✅
        "cModusoperandi": "Asaltante",     ✅
        ...
    }
}
```

**Código clave:**
```python
@app.post("/predict")
async def predict(request: PredictionRequest):
    # 1. Tokeniza el texto
    inputs = tokenizer(request.text, max_length=512, truncation=True, 
                      return_tensors='pt')
    
    # 2. Envía al modelo BERT
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], 
                       attention_mask=inputs['attention_mask'])
    
    # 3. Obtiene predicciones de 10 cabezas
    predictions = {}
    for i, field in enumerate(OUTPUT_FIELDS):
        logits = heads[i](outputs.last_hidden_state[:, 0, :])
        predictions[field] = get_predictions(logits, field)
    
    return {"best_predictions": {k: v[0] for k, v in predictions.items()}}
```

---

## 3️⃣ `SD911_AutoBot_Full.user.js` - INGRESA A LA PÁGINA Y LLENA CAMPOS

**¿Qué hace?** Bot que auto-ingresa y llena el formulario

### FASE 1: LOGIN (Ingresa a la página)
```javascript
function handleLogin() {
    // 1. Busca los campos
    const userField = document.querySelector("#usuario");
    const passField = document.querySelector("#password");
    const loginBtn = document.querySelector("button[type='submit']");

    // 2. Rellena credenciales
    userField.value = "45657263";
    passField.value = "911rosario";

    // 3. Simula eventos (para que la página detecte)
    userField.dispatchEvent(new Event('input', { bubbles: true }));
    passField.dispatchEvent(new Event('input', { bubbles: true }));

    // 4. Clickea para entrar
    setTimeout(() => loginBtn.click(), 1000);
}
```

### FASE 2: NAVEGACIÓN (Va al formulario)
```javascript
function handleMenu() {
    const btnCarga = document.querySelector("button[onclick*='form911auto']");
    btnCarga.click();
}
```

### FASE 3: DESAGREGACIÓN (Llena campos con IA)
```javascript
function handleFormulario() {
    const relato = document.querySelector("textarea[name='relato']");
    
    if (relato.value.length > 10) {
        // Envía al API
        fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: JSON.stringify({ text: relato.value })
        })
        .then(r => r.json())
        .then(data => llenarSelects(data.best_predictions));
    }
}

function llenarSelects(predicciones) {
    // Mapeo de campos
    const campos = {
        "cObjetivo": "objetivo",
        "cMedioempleado": "medio",
        // ... 8 campos más
    };

    // Para cada campo
    for (const [keyIA, nameHTML] of Object.entries(campos)) {
        const select = document.querySelector(`select[name='${nameHTML}']`);
        const valor = predicciones[keyIA];
        
        // Busca la opción que coincide
        for (let i = 0; i < select.options.length; i++) {
            if (select.options[i].text.toLowerCase().includes(valor.toLowerCase())) {
                // La selecciona
                select.selectedIndex = i;
                select.dispatchEvent(new Event('change', { bubbles: true }));
                break;
            }
        }
    }
}
```

---

## 🔄 EL FLUJO COMPLETO EN 10 SEGUNDOS

```
Ejecutas:  .\run_api.ps1
    ↓
    T+0s  : API inicia (http://127.0.0.1:8000) 
    ↓
    T+2s  : Navegador abre (http://10.100.32.84/SD911/)
    ↓
    T+3s  : Bot detecta la página, busca campos de login
    ↓
    T+4s  : Bot rellena usuario="45657263" y password="911rosario"
    ↓
    T+5s  : Bot clickea botón → ✅ INGRESA A LA PÁGINA
    ↓
    T+6s  : Bot busca botón de carga del formulario
    ↓
    T+7s  : Bot clickea botón → ✅ VA AL FORMULARIO
    ↓
    T+8s  : Bot detecta texto en campo "relato"
    ↓
    T+9s  : Bot envía texto al API (http://127.0.0.1:8000/predict)
    ↓
    T+10s : API responde con 10 predicciones
    ↓
    T+11s : Bot llena 10 campos select con predicciones
    ↓
    T+12s : ✅ FORMULARIO COMPLETAMENTE LLENO
```

---

## 🔑 CONFIGURACIÓN IMPORTANTE

### Cambia estos datos si es necesario:

**En `run_api.ps1`:**
```powershell
$SD911_URL = "http://10.100.32.84/SD911/"  # Tu URL
```

**En `SD911_AutoBot_Full.user.js`:**
```javascript
const CONFIG = {
    USERNAME: "45657263",              # Tu usuario
    PASSWORD: "911rosario",            # Tu contraseña
    API_URL: "http://127.0.0.1:8000"  # Puerto del API
};
```

---

## ✅ COMPONENTES CRÍTICOS (NO ELIMINAR)

```
IA_SEGURIDAD/
├── run_api.ps1                    ← LANZADOR
├── config.py                      ← CONFIGURACIÓN
├── requirements.txt               ← DEPENDENCIAS
├── api/
│   └── predict_api.py            ← API REST (PREDICCIONES)
├── model/checkpoints/
│   └── model_best.pt             ← MODELO BERT (420 MB)
├── data/processed/
│   ├── label_maps.json           ← MAPEO DE CATEGORÍAS
│   └── dataset_full.jsonl        ← DATOS DE ENTRENAMIENTO
└── frontend/
    └── SD911_AutoBot_Full.user.js ← BOT TAMPERMONKEY
```

---

## 🚀 CÓMO EJECUTAR

### Opción 1: Automático (RECOMENDADO)
```powershell
cd C:\Users\jorbiso\Desktop\PruebaIA\IA_SEGURIDAD
.\run_api.ps1
```

### Opción 2: Manual paso a paso
```powershell
# 1. Activar venv
.\venv\Scripts\Activate.ps1

# 2. Iniciar API
python -m uvicorn api.predict_api:app --port 8000

# 3. En otra terminal, abrir navegador
start "http://10.100.32.84/SD911/"
```

---

## 🧪 VERIFICAR QUE TODO FUNCIONE

```powershell
# ¿Está el API corriendo?
Invoke-WebRequest http://127.0.0.1:8000/health

# Resultado esperado:
# {"status": "healthy", "model_loaded": true}
```

```python
# ¿Hace predicciones?
import requests
requests.post("http://127.0.0.1:8000/predict", json={
    "text": "Robo en comercio"
}).json()

# Resultado esperado:
# {"best_predictions": {"cObjetivo": "Comercio", ...}}
```

---

## 🐛 TROUBLESHOOTING RÁPIDO

| Problema | Solución |
|----------|----------|
| "Venv no encontrado" | `py -3.10 -m venv venv` |
| "Puerto 8000 en uso" | `netstat -ano \| findstr :8000` → `taskkill /PID xxxx` |
| "Bot no entra a página" | Verifica credenciales y selectors (F12 → Inspector) |
| "API error 500" | Revisa `logs/api.log` |
| "Bot no llena campos" | Verifica que el relato tenga >10 caracteres |

---

## 📊 ESTADÍSTICAS DEL PROYECTO

| Componente | Tamaño | Función |
|-----------|--------|---------|
| Modelo BERT | 420 MB | Clasificación de texto |
| Dataset | 32 MB | Histórico de incidentes |
| Código Python | 18 KB | API + configuración |
| Código JavaScript | 18 KB | Bot Tampermonkey |
| Scripts PowerShell | 2 KB | Lanzador |

**Total:** ~470 MB (90% es el modelo)

---

## 🎓 LEARN BY DOING

### Experimento 1: Hacer predicción manual
```bash
# Abre PowerShell
$url = "http://127.0.0.1:8000/predict"
$payload = @{ text = "Robo a mano armada" } | ConvertTo-Json
Invoke-WebRequest -Uri $url -Method Post -Body $payload -ContentType "application/json"
```

### Experimento 2: Activar logs del bot
```javascript
// En Tampermonkey → Dashboard → Logs
// Verás mensajes como:
// [SD911 BOT] ✍️ Rellenando credenciales...
// [SD911 BOT] 🚀 Clickeando botón de entrada...
```

### Experimento 3: Modificar selectors
```javascript
// Si el login no funciona, abre DevTools (F12) y inspecciona:
// - Usuario: Click derecho → Inspeccionar → copia el selector
// - Ejemplo: si es <input name="login"> → usa "input[name='login']"
```

---

## 🎯 CHECKLIST FINAL

- [x] API running en http://127.0.0.1:8000
- [x] Modelo BERT cargado (0.41 GB)
- [x] Bot Tampermonkey instalado
- [x] Credenciales correctas (45657263 / 911rosario)
- [x] Estructura limpia (eliminados 14 archivos innecesarios)
- [x] Documentación completa (este documento)

---

**¡LISTO PARA USAR! 🚀**

Solo ejecuta: `.\run_api.ps1`

Todo se hace automáticamente.
