# 🚀 SOLUCIÓN NUCLEAR: SELENIUM BOT

## PASO 1: Instalar dependencias (Solo una vez)

Abre PowerShell en la carpeta del proyecto:

```powershell
cd C:\Users\jorbiso\Desktop\PruebaIA\IA_SEGURIDAD

# Si aún no activaste venv, hazlo:
.\venv\Scripts\Activate.ps1

# Instala Selenium y WebDriver
pip install selenium webdriver-manager
```

**Esperado:**
```
Successfully installed selenium-X.X.X webdriver-manager-X.X.X
```

---

## PASO 2: El script ya está listo

El archivo `bot_completo.py` ya fue creado en tu carpeta.

---

## PASO 3: EJECUTAR (Esto es lo que haces AHORA)

### Opción A: Doble clic directo
1. Ve a: `C:\Users\jorbiso\Desktop\PruebaIA\IA_SEGURIDAD\`
2. Busca: `bot_completo.py`
3. Doble clic

### Opción B: Desde PowerShell
```powershell
cd C:\Users\jorbiso\Desktop\PruebaIA\IA_SEGURIDAD
python bot_completo.py
```

---

## 📺 ¿QUÉ VERÁS?

### Secuencia correcta:

```
Iniciando Motor de IA (Backend)...
Esperando que la IA despierte...
IA LISTA PARA TRABAJAR.
Abriendo Navegador Controlado...
Iniciando sesion...
Login enviado.
MODO AUTO-PILOTO ACTIVADO. Navega al formulario...
```

Luego se abrirá **Chrome con el control de Selenium**.

El bot:
1. ✅ Ingresa automáticamente con usuario 45657263
2. ✅ Navega al formulario
3. ✅ **ESPERA** a que escribas el relato (o lo cargues manualmente)
4. ✅ Cuando detecta texto, lo envía a la IA
5. ✅ La IA devuelve predicciones
6. ✅ Bot rellena automáticamente los selects
7. ✅ Los selects se ponen VERDES (feedback visual)

---

## 🎯 WORKFLOW FINAL

```
Tu navegador se abre → Bot ingresa → Espera formulario → 
Tú escribes/cargas relato → Bot lo detecta → Envía a IA → 
IA responde → Bot llena campos → TODO HECHO
```

---

## ❌ Si algo falla:

### "No se encontró el elemento #usuario"
- Los IDs de la página son diferentes
- Solución: Abre DevTools (F12), inspecciona los campos y dime qué IDs tienen

### "Error de conexión con API"
- Asegúrate que `.\venv\Scripts\Activate.ps1` esté activado
- Verifica que `api.predict_api:app` es el path correcto

### Chrome no abre
- Selenium necesita Chrome instalado
- Descarga Chrome desde: https://www.google.com/chrome/

---

## 💡 Notas útiles

- **Selenium TOMA CONTROL DEL NAVEGADOR** - verás que escribe, clickea, etc. Es normal.
- **Puedes pausar** - Presiona `Ctrl+C` en la terminal para detener.
- **Sin límite de tiempo** - El bot espera indefinidamente a que escribas el relato.
- **Puedes testear manualmente** - Escribe un relato en el formulario y el bot lo detectará automáticamente.

---

**¡ES HORA DE EJECUTAR! 🚀**

```powershell
python bot_completo.py
```
