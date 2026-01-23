# 🎯 CÓMO INGRESA EL BOT A LA PÁGINA - PASO A PASO

## 🚀 LO QUE OCURRE CUANDO EJECUTAS: `.\run_api.ps1`

### PASO 1: Se inicia el API
```powershell
.\run_api.ps1
```

↓ Internamente:
```
1. Ejecuta: python -m uvicorn api.predict_api:app --port 8000
2. El API carga el modelo BERT (0.41 GB)
3. Espera en http://127.0.0.1:8000 listo para recibir predicciones
4. Verifica /health y obtiene: {"status": "healthy", "model_loaded": true}
```

---

### PASO 2: El navegador abre la página del 911
```powershell
Start-Process "http://10.100.32.84/SD911/"
```

↓ Resultado:
```
URL: http://10.100.32.84/SD911/
Estado: PÁGINA DE LOGIN (usuario + contraseña)
```

---

### PASO 3: Tampermonkey detecta la página y el bot se ACTIVA

**Condición:** El script tiene: `@match http://10.100.32.84/SD911/*`

```javascript
if (window.location.href.includes("http://10.100.32.84/SD911/")) {
    console.log("BOT ACTIVADO: Estoy en la página del 911");
    // El script está LISTO para actuar
}
```

---

## 🔐 FASE 1: AUTO-LOGIN (Ingreso a la página)

```javascript
// ===== CÓDIGO QUE INGRESA A LA PÁGINA =====

function handleLogin() {
    // 1. BUSCA LOS CAMPOS DE LOGIN
    const userField = document.querySelector("#usuario");
    const passField = document.querySelector("#password");
    const loginBtn = document.querySelector("button[type='submit']");

    // 2. VERIFICA QUE LOS CAMPOS EXISTAN
    if (userField && passField && loginBtn) {
        
        // 3. INSERTA EL USUARIO
        userField.value = "45657263";
        console.log("✅ Usuario insertado: 45657263");
        
        // 4. INSERTA LA CONTRASEÑA
        passField.value = "911rosario";
        console.log("✅ Contraseña insertada: 911rosario");

        // 5. SIMULA QUE EL USUARIO ESCRIBIÓ (EVENTOS)
        userField.dispatchEvent(new Event('input', { bubbles: true }));
        userField.dispatchEvent(new Event('change', { bubbles: true }));
        passField.dispatchEvent(new Event('input', { bubbles: true }));
        passField.dispatchEvent(new Event('change', { bubbles: true }));

        // 6. ESPERA 1 SEGUNDO
        setTimeout(() => {
            
            // 7. CLICKEA EL BOTÓN DE ENTRADA
            console.log("🚀 Clickeando botón de login...");
            loginBtn.click();
            
            // 8. SI NO FUNCIONA, INTENTA DE NUEVO
            setTimeout(() => {
                if (document.querySelector("#usuario")) {
                    console.log("⚠️ Primer click no funcionó, intentando de nuevo...");
                    loginBtn.click();
                }
            }, 1500);
            
        }, 1000);
    }
}
```

**¿Qué hace línea por línea?**

| Línea | Acción | Resultado |
|-------|--------|-----------|
| 1 | Busca `<input id="usuario">` | Encuentra campo de usuario |
| 2 | Busca `<input id="password">` | Encuentra campo de contraseña |
| 3 | Busca `<button type="submit">` | Encuentra botón de login |
| 5 | `userField.value = "45657263"` | Rellena usuario |
| 8 | `passField.value = "911rosario"` | Rellena contraseña |
| 11 | `dispatchEvent('input')` | Simula que escribiste (para validación) |
| 16 | `loginBtn.click()` | **CLICKEA - INICIA SESIÓN** |

**Resultado:** ✅ **INGRESASTE A LA PÁGINA DEL 911**

---

## 🗺️ FASE 2: NAVEGACIÓN (Va al formulario de desagregación)

Después del login, el bot busca automáticamente dónde llenar incidentes:

```javascript
function handleMenu() {
    // 1. BUSCA EL BOTÓN QUE DICE "CARGAR"
    const btnCarga = document.querySelector("button[onclick*='form911auto']");
    
    if (btnCarga) {
        console.log("✅ Botón de carga encontrado");
        
        // 2. CLICKEA EL BOTÓN
        btnCarga.click();
        console.log("🔄 Navegando al formulario...");
        
        // 3. SI NO REDIRIGE, FUERZA LA REDIRECCIÓN
        setTimeout(() => {
            if (!window.location.href.includes("form911auto")) {
                window.location.href = "form911auto";
                console.log("🔗 Redirección forzada a: form911auto");
            }
        }, 1500);
    }
}
```

**Resultado:** ✅ **ESTÁS EN LA PÁGINA DEL FORMULARIO DE DESAGREGACIÓN**

---

## 🧠 FASE 3: DESAGREGACIÓN (Llena campos automáticamente)

Cuando llega al formulario, el bot detecta el relato y envía a la IA:

```javascript
function handleFormulario() {
    // 1. BUSCA EL CAMPO "RELATO"
    const relato = document.querySelector("textarea[name='relato']");
    
    if (relato && relato.value.length > 10) {
        console.log("📝 Relato encontrado, enviando a IA...");
        console.log("Texto:", relato.value);
        
        // 2. ENVÍA AL API DE IA
        enviarAIA(relato.value);
    }
}

function enviarAIA(texto) {
    // 3. HACER PETICIÓN POST AL API EN LOCALHOST:8000
    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
            text: texto,
            incidente_id: "AUTO-" + Date.now()
        })
    })
    .then(resp => resp.json())
    .then(data => {
        console.log("🧠 IA respondió con predicciones");
        console.log(data.best_predictions);
        
        // 4. LLENA LOS CAMPOS CON LAS PREDICCIONES
        llenarSelects(data.best_predictions);
    })
    .catch(err => console.error("❌ Error:", err));
}

function llenarSelects(predicciones) {
    // Mapeo: Nombre IA → Nombre HTML
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

    let contador = 0;
    
    // Para cada predicción de IA
    for (const [keyIA, nameHTML] of Object.entries(campos)) {
        const prediccion = predicciones[keyIA];
        const select = document.querySelector(`select[name='${nameHTML}']`);
        
        if (select && prediccion) {
            // BUSCA EN EL <SELECT> LA OPCIÓN QUE COINCIDE
            for (let i = 0; i < select.options.length; i++) {
                const textoOpcion = select.options[i].text.toLowerCase();
                const valorPrediccion = String(prediccion).toLowerCase();
                
                // SI COINCIDE (ej: "Comercio" == "Comercio")
                if (textoOpcion.includes(valorPrediccion)) {
                    // SELECCIONA LA OPCIÓN
                    select.selectedIndex = i;
                    select.value = select.options[i].value;
                    
                    // DISPARA EVENTO PARA QUE LA PÁGINA LO DETECTE
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    
                    // CAMBIA BORDE A VERDE (INDICA QUE SE LLENÓ)
                    select.style.border = "2px solid #27ae60";
                    
                    contador++;
                    console.log(`✅ ${keyIA} → ${prediccion}`);
                    break;
                }
            }
        }
    }
    
    console.log(`🎉 ${contador} campos llenados automáticamente`);
}
```

**Resultado:** ✅ **TODOS LOS CAMPOS LLENOS AUTOMÁTICAMENTE**

---

## 📊 EJEMPLO REAL DE PREDICCIONES

**Input (Relato):**
```
"Robo con violencia en establecimiento comercial. Sustrajeron mercancía 
valorada en 5000 pesos. El delincuente ingresó por la puerta principal 
a las 23:45 horas armado con un arma blanca. Escapó en motocicleta."
```

**IA Predice:**
```javascript
{
    "cObjetivo": "Comercio",              // Lugar del hecho
    "cMedioempleado": "Con Arma Blanca",  // Arma utilizada
    "cModusoperandi": "Asaltante",        // Forma de actuar
    "cMedios_fuga": "Motocicleta",        // Cómo escapó
    "cElementos_sustraidos": "Mercaderia", // Qué robó
    "cLocalizacion": "Interior De Un Inmueble", // Dónde pasó
    "cGenero_Sexo": "No Registra",        // Del perpetrador
    "cEdad": "No Registra",               // Del perpetrador
    "cRectificacion_Tipo": "ROBO",        // Tipo de delito
    "cRectificacion_Subtipo": "CONSUMADO" // Subtipo (ejecutado)
}
```

**Bot selecciona en el formulario:**
```
<select name="objetivo">
    <option value="1">Comercio</option>           ← SELECCIONA
    <option value="2">Kiosko</option>
</select>

<select name="medio">
    <option value="0">No Registra</option>
    <option value="3">Con Arma Blanca</option>    ← SELECCIONA
</select>

... (8 selects más)
```

---

## 🔄 FLUJO TEMPORAL COMPLETO

```
T+0s   : Ejecutas .\run_api.ps1
         └─ API inicia en puerto 8000

T+3s   : Navegador abre http://10.100.32.84/SD911/
         └─ Bot Tampermonkey detecta la página

T+5s   : handleLogin() ejecuta
         ├─ Busca campos #usuario, #password
         ├─ Rellena: usuario="45657263", password="911rosario"
         ├─ Dispara eventos 'input' y 'change'
         └─ Clickea button[type='submit']

T+6s   : Se verifica login nuevamente
         └─ Si aún ves el login, reintenta el click

T+7s   : ✅ LOGIN EXITOSO
         └─ handleMenu() busca botón de carga

T+8s   : Bot clickea button[onclick*='form911auto']
         └─ Página redirige a /form911auto

T+9s   : ✅ ESTÁS EN EL FORMULARIO
         └─ handleFormulario() busca textarea[name='relato']

T+10s  : Bot detecta texto en relato
         └─ Envía POST a http://127.0.0.1:8000/predict

T+11s  : API procesa:
         ├─ Tokeniza texto con BERT
         ├─ Procesa 12 capas de transformers
         ├─ 10 cabezas clasifican
         └─ Devuelve predicciones

T+12s  : Bot recibe predicciones
         ├─ Mapea cada predicción
         ├─ Busca opción en cada <select>
         ├─ Selecciona opción
         ├─ Dispara evento 'change'
         └─ Cambia borde a verde

T+13s  : ✅ FORMULARIO COMPLETAMENTE LLENO
         └─ Listo para guardar/enviar
```

---

## ⚠️ SI NO FUNCIONA

### Problema 1: "Bot no entra a la página"
**Verificar:**
1. Las credenciales sean correctas en el script
2. Selectors HTML coincidan (usa DevTools F12)
3. URL sea la correcta (revisa `run_api.ps1`)

### Problema 2: "Bot entra pero no llena campos"
**Verificar:**
1. API esté corriendo: `http://127.0.0.1:8000/health`
2. Consola Tampermonkey muestre mensajes (Tampermonkey → Dashboard → Logs)
3. Network en DevTools muestre POST a `/predict` con 200 OK

### Problema 3: "Bot no encuentra los campos"
**Solución:**
1. Abre DevTools (F12) en la página del 911
2. Copia los selectores reales:
   - Usuario: Inspecciona el input → copia el id/name/selector
   - Contraseña: Inspecciona el input → copia el id/name/selector
   - Botón: Inspecciona el botón → copia el selector
3. Actualiza los selectors en el script

---

## 🎯 RESUMEN FINAL

```
run_api.ps1 (Inicia API)
    ↓
Navegador abre http://10.100.32.84/SD911/
    ↓
Tampermonkey + SD911_AutoBot_Full.user.js se ACTIVA
    ↓
FASE 1: handleLogin()
    ├─ Busca #usuario, #password, button[type='submit']
    ├─ Rellena: usuario="45657263", password="911rosario"
    └─ CLICKEA → Ingresa a la página ✅
    ↓
FASE 2: handleMenu()
    ├─ Busca button[onclick*='form911auto']
    └─ CLICKEA → Va al formulario ✅
    ↓
FASE 3: handleFormulario()
    ├─ Detecta textarea[name='relato']
    ├─ POST a http://127.0.0.1:8000/predict
    ├─ Recibe predicciones de IA
    └─ Llena 10 selects automáticamente ✅
```

**¡Así ingresa el bot a la página y hace todo automáticamente!**
