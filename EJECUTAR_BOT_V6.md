# 🚀 INSTRUCCIONES PARA EJECUTAR BOT V6.0 ULTRA

## ⚠️ ANTES DE EJECUTAR - CHECKLIST:

### 1. ✅ Instala el script en Tampermonkey
```
1. Abre: http://10.100.32.84/SD911/
2. Click derecho → Tampermonkey → Crear nuevo script
3. Borra todo y copia el código de: frontend/SD911_AutoBot_Full.user.js
4. Guarda (Ctrl+S)
5. Cierra el navegador
```

### 2. ✅ Verifica el API está corriendo
```powershell
$resp = Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
$resp.Content
# Debe mostrar: {"status": "healthy", "model_loaded": true}
```

### 3. ✅ Limpia procesos viejos
```powershell
taskkill /f /im python.exe 2>$null
taskkill /f /im chrome.exe 2>$null
```

---

## 🎯 EJECUCIÓN CORRECTA:

### PASO 1: Abre PowerShell
```powershell
cd C:\Users\jorbiso\Desktop\PruebaIA\IA_SEGURIDAD
```

### PASO 2: Ejecuta el lanzador
```powershell
.\run_api.ps1
```

**Deberías ver:**
```
Servidor iniciando en http://127.0.0.1:8000 (PID xxx)
API lista (health 200)
Se abrio el navegador en: http://10.100.32.84/SD911/
```

---

## 📺 OBSERVA LA PÁGINA - DEBERÍA VER UNA BARRA ARRIBA:

### ✅ FLUJO CORRECTO:

```
┌─────────────────────────────────────────────────────┐
│ [BOT V6.0 ULTRA]                                    │
│ 1. INICIANDO LOGIN...                               │
└─────────────────────────────────────────────────────┘
        ↓ (1-2 segundos)
┌─────────────────────────────────────────────────────┐
│ [BOT V6.0 ULTRA]                                    │
│ Buscando campos... Usuario:SI Pass:SI Btn:SI        │
│ 2. CAMPOS ENCONTRADOS! Inyectando credenciales...   │
└─────────────────────────────────────────────────────┘
        ↓ (1 segundo)
┌─────────────────────────────────────────────────────┐
│ [BOT V6.0 ULTRA]                                    │
│ 3. CLICKEANDO BOTON DE INGRESO...                   │
└─────────────────────────────────────────────────────┘
        ↓ (SE DEBE REDIRIGIR)
┌─────────────────────────────────────────────────────┐
│ [BOT V6.0 ULTRA]                                    │
│ 4. BUSCANDO BOTON CARGAR...                         │
│ 5. BOTON ENCONTRADO! CLICKEANDO...                  │
└─────────────────────────────────────────────────────┘
        ↓ (SE VA AL FORMULARIO)
┌─────────────────────────────────────────────────────┐
│ [BOT V6.0 ULTRA]                                    │
│ 6. BUSCANDO RELATO EN FORMULARIO...                 │
│ Relato encontrado (XXX caracteres)                  │
│ 7. TEXTO LISTO! ENVIANDO A LA IA...                 │
└─────────────────────────────────────────────────────┘
        ↓ (CONECTA CON API)
┌─────────────────────────────────────────────────────┐
│ [BOT V6.0 ULTRA]                                    │
│ 8. CONECTANDO A IA EN http://127.0.0.1:8000...     │
│ 9. IA RESPONDIO! PROCESANDO...                      │
│ 10. LLENANDO CAMPOS CON PREDICCIONES...             │
│ 11. COMPLETO! 10 campos llenos. GUARDAR AHORA!      │
└─────────────────────────────────────────────────────┘
```

---

## ❌ SI NO FUNCIONA - DEBUGGING:

### Problema 1: "1. INICIANDO LOGIN..." y se queda
**CAUSA:** No encuentra los campos  
**SOLUCIÓN:**
```
1. Abre DevTools: F12
2. Console: document.getElementById("usuario")
   - Si dice "null" → el ID no existe o cambió
   - Si dice "<input id='usuario'...>" → existe
3. Copia el selector correcto y actualiza en el script
```

### Problema 2: "Buscando campos... Usuario:NO"
**CAUSA:** #usuario no existe  
**SOLUCIÓN:**
```
En DevTools:
  - Inspecciona el campo de usuario
  - Copia su: id, name, o atributo que tenga
  - Comunícame exactamente qué ves
```

### Problema 3: "ERROR CONEXION: Ventana negra NO corre"
**CAUSA:** API no está en http://127.0.0.1:8000  
**SOLUCIÓN:**
```powershell
# Verifica que el API esté corriendo
$resp = Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
echo $resp.Content

# Si da error, ejecuta run_api.ps1 en otra terminal
```

### Problema 4: "ERROR: API devolvio 500"
**CAUSA:** Hay un error en el API  
**SOLUCIÓN:**
```
1. Mira la ventana negra (terminal donde corre el API)
2. Busca mensajes de error
3. Comparte conmigo el error
```

---

## 📋 DATOS PARA DEBUGGEAR:

### Si algo falla, copia esto y comparte:

**En PowerShell:**
```powershell
# 1. Verifica API
$resp = Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
Write-Host $resp.Content

# 2. Verifica puerto
netstat -ano | findstr :8000
```

**En DevTools (F12 → Console):**
```javascript
// Copia y pega esto:
console.log("Usuario field:", document.getElementById("usuario"));
console.log("Password field:", document.getElementById("password"));
console.log("Submit button:", document.querySelector("button[type='submit']"));
console.log("Formulario:", document.querySelector("textarea[name='relato']"));
console.log("Boton cargar:", document.querySelector("button[onclick*='form911auto']"));
```

**En Tampermonkey (Logs):**
```
1. Click derecho en página
2. Tampermonkey → Dashboard
3. Haz clic en tu script
4. Pestaña "Logs"
5. Copia todo lo que ves rojo/error
```

---

## ✅ CHECKLIST FINAL:

- [ ] Script V6.0 instalado en Tampermonkey
- [ ] API corriendo en http://127.0.0.1:8000
- [ ] run_api.ps1 ejecutado
- [ ] Barra de color aparece en la página
- [ ] Ves mensajes "1. INICIANDO LOGIN..."
- [ ] Bot ingresa (redirecciona después del login)
- [ ] Bot va al formulario (URL cambia a /form911auto)
- [ ] Bot ve el relato (detecta > 5 caracteres)
- [ ] Bot conecta a IA (muestra "CONECTANDO A IA...")
- [ ] Campos se llenan con borde VERDE
- [ ] Ves "COMPLETO! 10 campos llenos"

---

## 🎯 ÚLTIMA OPCIÓN (Si nada funciona):

1. **Envíame screenshot de:**
   - La barra del bot (qué dice exactamente)
   - DevTools → Console (qué errores ves)
   - La ventana negra (qué logs muestra)

2. **Dime:**
   - ¿Hasta qué paso llega la barra?
   - ¿Se redirige después del login?
   - ¿Qué errores ves en la consola?

**¡Vamos a hacerlo funcionar! 🚀**
