import time
import requests
import subprocess
import sys
import os
import socket
import unicodedata
import difflib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- CAMBIAR AL DIRECTORIO DEL SCRIPT ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# --- CONFIGURACIÓN ---
URL_LOGIN = "http://10.100.32.84/SD911/login"
USER = "45657263"
PASS = "911rosario"
API_URL = "http://127.0.0.1:8000"
API_PORT = 8000

def puerto_disponible(puerto):
    """Verifica si un puerto está disponible"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', puerto))
    sock.close()
    return result != 0  # Retorna True si el puerto está LIBRE

def iniciar_api():
    """Arranca la API de IA en segundo plano"""
    # Verificar si el puerto ya está en uso
    if not puerto_disponible(API_PORT):
        print(f"⚠ Advertencia: Puerto {API_PORT} ya está en uso. Intentando limpiar...")
        if sys.platform == "win32":
            os.system(f"taskkill /F /IM python.exe /FI \"WINDOWTITLE eq*uvicorn*\" 2>nul")
        time.sleep(2)
    
    print("Iniciando Motor de IA (Backend)...")
    if sys.platform == "win32":
        # En Windows usamos CREATE_NEW_PROCESS_GROUP para mejor manejo de señales
        cmd = f"{sys.executable} -m uvicorn api.predict_api:app --host 127.0.0.1 --port 8000 --log-level warning"
        proceso = subprocess.Popen(
            cmd, 
            shell=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
    else:
        proceso = subprocess.Popen(
            ["python3", "-m", "uvicorn", "api.predict_api:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"]
        )
    
    return proceso

def esperar_api():
    """Espera a que la API responda"""
    print("Esperando que la IA despierte...")
    
    # Esperar inicial con manejo de interrupción
    for _ in range(3):
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠ Interrupción durante espera inicial. Continuando...")
    
    max_intentos = 30
    for intento in range(max_intentos):
        try:
            r = requests.get(f"{API_URL}/health", timeout=1)
            if r.status_code == 200:
                print("✓ IA LISTA PARA TRABAJAR.")
                time.sleep(1)  # Pequeña pausa extra para estabilizar
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if intento % 10 == 0 and intento > 0:
                print(f"  Reintentando... ({intento}/{max_intentos})")
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print("\n⚠ Interrupción durante espera de API. Continuando...")
        except KeyboardInterrupt:
            raise  # Re-lanzar para salir correctamente
        except Exception as e:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print("\n⚠ Interrupción. Continuando...")
    
    print("⚠ La API no responde, pero intentaré continuar...")
    return True

def bot_navegador():
    # Configurar Chrome
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # options.add_argument("--headless") # Descomentar si no quieres ver el navegador
    
    print("Abriendo Navegador Controlado...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(60)  # Timeout para cargas de página
    
    try:
        # 1. LOGIN
        print("Iniciando sesion...")
        driver.get(URL_LOGIN)
        time.sleep(2)
        
        try:
            driver.find_element(By.ID, "usuario").send_keys(USER)
            driver.find_element(By.ID, "password").send_keys(PASS)
            
            # Buscar boton de varias formas
            try:
                driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
            except:
                driver.find_element(By.CLASS_NAME, "btn-info").click()
                
            print("Login enviado.")
        except Exception as e:
            print(f"Nota: Quizas ya estabas logueado o cambio el login. {e}")

        # 1.5 TOCAR BOTON CARGAR
        print("Buscando boton Cargar...")
        time.sleep(2)
        try:
            # Buscar el boton "Cargar" por varias formas
            try:
                btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Cargar')]")
            except:
                btn = driver.find_element(By.CSS_SELECTOR, "button.btn-info.btn-block")
            
            print("Tocando boton Cargar...")
            btn.click()
            time.sleep(2)
        except Exception as e:
            print(f"Nota: Boton Cargar no encontrado. {e}")

        # 2. BUCLE DE TRABAJO (AUTO-PILOTO)
        print("MODO AUTO-PILOTO ACTIVADO. Esperando texto en relato...")
        print("Chrome permanecerá abierto. Presiona CTRL+C en la terminal para salir.\n")
        
        relatos_procesados = set()  # Guardar relatos ya procesados
        contador_ciclos = 0
        
        while True:
            try:
                time.sleep(0.5)  # Revisar cada 500ms para detectar rápido
                contador_ciclos += 1
                
                # Verificar si estamos en el formulario
                if "form911auto" in driver.current_url:
                    try:
                        relato_box = driver.find_element(By.NAME, "relato")
                        texto_actual = relato_box.get_attribute("value").strip()
                        
                        # SI HAY TEXTO Y NO LO HEMOS PROCESADO AÚN
                        if len(texto_actual) > 10 and texto_actual not in relatos_procesados:
                            print(f"\n[CICLO {contador_ciclos}] Nuevo relato detectado: {texto_actual[:30]}...")
                            
                            # LLAMAR A LA IA
                            try:
                                resp = requests.post(f"{API_URL}/predict", json={"text": texto_actual}, timeout=5)
                                if resp.status_code == 200:
                                    preds = resp.json().get("best_predictions", {})
                                    
                                    # LLENAR CAMPOS
                                    mapeo = {
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
                                    }
                                    
                                    count = 0
                                    for k_ia, k_html in mapeo.items():
                                        if k_ia in preds:
                                            val_ia = str(preds[k_ia]).strip().lower()  # Valor que predijo la IA
                                            
                                            # REINTENTOS ESPECIALES PARA SUBTIPO
                                            reintentos = 3 if k_html == "rectificacion_subtipo" else 1

                                            # Normalización para comparación (quita acentos)
                                            def _norm(s):
                                                if not s:
                                                    return ""
                                                return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode().strip().lower()

                                            # ===== LÓGICA ESPECIAL PARA rectificacion_subtipo =====
                                            if k_html == "rectificacion_subtipo":
                                                texto_norm = _norm(texto_actual)
                                                
                                                # PASO 1: ANÁLISIS SEMÁNTICO PRIMERO (ANTES de confiar en la IA)
                                                print(f"  ℹ Analizando contexto del relato...")
                                                
                                                # Definir palabras clave por categoría (EXACTAS Y PRECISAS)
                                                palabras_consumado = [
                                                    'robaron', 'sustrajeron', 'sustraído', 'sustraida',
                                                    'están robando', 'estan robando', 'está robando', 'esta robando',
                                                    'están sustrayendo', 'estan sustrayendo',
                                                    'fue robado', 'fue sustraído', 'le robaron',
                                                    'robo consumado', 'hurto consumado', 'se llevaron',
                                                    'desaparecieron', 'desapareci', 'sustrae', 'roba'
                                                ]
                                                palabras_tentativa = [
                                                    'intenta', 'intentó', 'intento', 'fue frustrado', 'fue interrumpido',
                                                    'no logró', 'no logro', 'no pudo', 'fue detenido', 'fue capturado',
                                                    'en la tentativa', 'tentativa de', 'intentaba', 'trató de',
                                                    'no completó', 'sin consumar'
                                                ]
                                                # CALIFICADO: Robo CON VIOLENCIA o CON ARMA
                                                palabras_calificado = [
                                                    # Con armas
                                                    'arma de fuego', 'arma fuego', 'pistola', 'revolver', 'fusil',
                                                    'escopeta', 'rifle', 'arma blanca', 'cuchillo', 'machete',
                                                    'puñal', 'navaja', 'con arma', 'armado', 'armada',
                                                    'disparó', 'disparo', 'balazo', 'herida de bala',
                                                    # Con violencia
                                                    'con violencia', 'violencia fisica', 'violencia fisica',
                                                    'golpizado', 'golpiza', 'golpearón', 'agredió', 'agredieron',
                                                    'herida', 'herido', 'lesión', 'lesionó',
                                                    'daño grave', 'grave', 'agravio',
                                                    'amenaza', 'amenazó', 'amenazaron',
                                                    'forcejeo', 'resistencia', 'enfrentamiento'
                                                ]
                                                
                                                # Verificar palabras clave en el relato
                                                tiene_consumado = any(palabra in texto_norm for palabra in palabras_consumado)
                                                tiene_tentativa = any(palabra in texto_norm for palabra in palabras_tentativa)
                                                tiene_calificado = any(palabra in texto_norm for palabra in palabras_calificado)
                                                
                                                categoria_detectada = None
                                                motivo = None
                                                
                                                # PRIORIDAD: Tentativa (más específica)
                                                if tiene_tentativa:
                                                    categoria_detectada = 'TENTATIVA'
                                                    motivo = "describe un intento frustrado"
                                                    print(f"  ✓ DETECTADO POR PALABRAS CLAVE: TENTATIVA ({motivo})")
                                                # Calificado: Robo CONSUMADO + VIOLENCIA/ARMA
                                                elif tiene_calificado and tiene_consumado:
                                                    categoria_detectada = 'CALIFICADO'
                                                    # Determinar si fue por arma o violencia
                                                    tiene_arma = any(p in texto_norm for p in ['arma', 'pistola', 'revolver', 'fusil', 'escopeta', 'rifle', 'arma blanca', 'cuchillo', 'machete', 'puñal', 'navaja', 'disparó', 'disparo', 'balazo'])
                                                    motivo = "tiene arma" if tiene_arma else "tiene violencia"
                                                    print(f"  ✓ DETECTADO POR PALABRAS CLAVE: CALIFICADO (robo consumado con {motivo})")
                                                # Simple: Solo robo consumado sin violencia
                                                elif tiene_consumado:
                                                    categoria_detectada = 'SIMPLE'
                                                    motivo = "robo consumado sin violencia"
                                                    print(f"  ✓ DETECTADO POR PALABRAS CLAVE: SIMPLE ({motivo})")
                                                # Si solo hay violencia sin robo claro, es calificado
                                                elif tiene_calificado:
                                                    categoria_detectada = 'CALIFICADO'
                                                    motivo = "describe violencia/arma (posible robo calificado)"
                                                    print(f"  ✓ DETECTADO POR PALABRAS CLAVE: CALIFICADO ({motivo})")
                                                
                                                # PASO 2: Consultar la IA solo si no se detectó por palabras clave
                                                try:
                                                    print(f"  ℹ Consultando IA para validar...")
                                                    refined_resp = requests.post(
                                                        f"{API_URL}/predict-refined",
                                                        json={
                                                            "text": texto_actual,
                                                            "category": "cRectificacion_Subtipo"
                                                        },
                                                        timeout=5
                                                    )
                                                    if refined_resp.status_code == 200:
                                                        refined_data = refined_resp.json()
                                                        ia_pred = refined_data.get("best_prediction", val_ia).strip().lower()
                                                        top_options = refined_data.get("predictions", {})

                                                        if top_options:
                                                            print(f"  ✓ IA sugiere (en orden de confianza):")
                                                            for i, (opcion, conf) in enumerate(list(top_options.items())[:3], 1):
                                                                print(f"    {i}. {opcion} ({conf*100:.1f}%)")

                                                        # DECISIÓN FINAL: Usar palabras clave si las detectamos, sino usar IA
                                                        if categoria_detectada:
                                                            val_ia = categoria_detectada
                                                            ia_pred_norm = _norm(ia_pred)
                                                            if ia_pred_norm != _norm(categoria_detectada):
                                                                print(f"  🔄 PRIORIDAD A CONTEXTO: Detectamos '{categoria_detectada}' pero IA predijo '{ia_pred}'")
                                                        else:
                                                            val_ia = ia_pred
                                                            print(f"  ℹ Sin palabras clave claras, usando predicción IA: {ia_pred}")

                                                        # MAPEO FINAL: Convertir predicción de IA a opciones válidas del formulario
                                                        mapeo_final = {
                                                            'consumado': 'SIMPLE',
                                                            'simple': 'SIMPLE',
                                                            'calificado': 'CALIFICADO',
                                                            'calificado consumado': 'CALIFICADO',
                                                            'calificado en proceso': 'CALIFICADO',
                                                            'tentativa': 'TENTATIVA',
                                                            'en proceso': 'SIMPLE',  # La IA a veces predice esto, pero no existe
                                                            'en proceso de consumacion': 'SIMPLE',
                                                            'otra': 'OTRA CLASIFICACION',
                                                            'otra clasificacion': 'OTRA CLASIFICACION'
                                                        }

                                                        val_ia_norm = _norm(val_ia)
                                                        if val_ia_norm in mapeo_final:
                                                            val_ia = mapeo_final[val_ia_norm]
                                                            print(f"  ✓ Convertido a opción válida: {val_ia}")
                                                        elif val_ia not in ['CALIFICADO', 'SIMPLE', 'TENTATIVA', 'OTRA CLASIFICACION']:
                                                            # Si no está en el mapeo y no es una opción válida, asumir SIMPLE por defecto
                                                            print(f"  ⚠ Valor '{val_ia}' no reconocido, usando SIMPLE por defecto")
                                                            val_ia = 'SIMPLE'
                                                except Exception as e:
                                                    print(f"  ⚠ Error consultando API refinada: {type(e).__name__}")
                                                
                                            for intento_reintento in range(reintentos):
                                                try:
                                                    # ESPERAR A QUE EL ELEMENTO ESTÉ DISPONIBLE (más tiempo para subtipo)
                                                    wait_time = 5 if k_html == "rectificacion_subtipo" else 2
                                                    wait = WebDriverWait(driver, wait_time)
                                                    select_element = wait.until(
                                                        EC.presence_of_element_located((By.NAME, k_html))
                                                    )

                                                    # Pequeño delay extra antes de interactuar
                                                    time.sleep(0.2)

                                                    select_obj = Select(select_element)

                                                    # Esperar hasta que las opciones estén cargadas (útil si se cargan por JS)
                                                    start_t = time.time()
                                                    while time.time() - start_t < 3:
                                                        if len(select_obj.options) > 0:
                                                            break
                                                        time.sleep(0.15)

                                                    found_index = -1

                                                    # ===== LÓGICA ESPECIAL DE MATCHING PARA rectificacion_subtipo =====
                                                    if k_html == "rectificacion_subtipo":
                                                        # BÚSQUEDA DIRECTA: Buscar la opción EXACTA en el formulario
                                                        opciones_textos = [opt.text for opt in select_obj.options]
                                                        
                                                        # Búsqueda 1: Búsqueda exacta sin normalizar
                                                        if val_ia in opciones_textos:
                                                            found_index = opciones_textos.index(val_ia)
                                                            print(f"  ✓ Encontrada opción exacta: {val_ia}")
                                                        else:
                                                            # Búsqueda 2: Búsqueda en mayúsculas
                                                            val_upper = val_ia.upper()
                                                            if val_upper in opciones_textos:
                                                                found_index = opciones_textos.index(val_upper)
                                                                print(f"  ✓ Encontrada opción en mayúsculas: {val_upper}")
                                                            else:
                                                                # Búsqueda 3: Búsqueda parcial (contiene la palabra)
                                                                for i, opt_text in enumerate(opciones_textos):
                                                                    if val_ia.upper() in opt_text.upper():
                                                                        found_index = i
                                                                        print(f"  ✓ Encontrada opción parcial: {opt_text}")
                                                                        break
                                                        
                                                        if found_index == -1:
                                                            opciones = [o.text for o in select_obj.options if o.text.strip()]
                                                            print(f"  ⚠ No se encontró '{val_ia}' en {k_html}. Opciones disponibles: {opciones}")
                                                    else:
                                                        # PARA OTROS CAMPOS: Usar lógica anterior
                                                        val_norm = _norm(val_ia)

                                                        # Diccionario de sinónimos/mapeos para rectificacion_subtipo (VALORES DEL FORMULARIO)
                                                        # Mapear desde predicciones de la IA a opciones del formulario
                                                        synonym_map = {
                                                            'consumado': 'SIMPLE',
                                                            'simple': 'SIMPLE',
                                                            'calificado': 'CALIFICADO',
                                                            'calificado consumado': 'CALIFICADO',
                                                            'tentativa': 'TENTATIVA',
                                                            'en proceso': 'TENTATIVA',
                                                            'otra': 'OTRA CLASIFICACION',
                                                            'otra clasificacion': 'OTRA CLASIFICACION'
                                                        }

                                                        # Heurísticas adicionales para cubrir frases comunes (VALORES DEL FORMULARIO)
                                                        if 'consum' in val_norm or 'simple' in val_norm:
                                                            val_norm = 'SIMPLE'
                                                        elif 'calific' in val_norm:
                                                            val_norm = 'CALIFICADO'
                                                        elif 'tentat' in val_norm or 'intento' in val_norm or 'en proceso' in val_norm:
                                                            val_norm = 'TENTATIVA'
                                                        elif 'otra' in val_norm:
                                                            val_norm = 'OTRA CLASIFICACION'
                                                        # aplicar mapping directo si existe
                                                        elif val_norm in synonym_map:
                                                            val_norm = synonym_map[val_norm]

                                                        # --- PASO 1: BÚSQUEDA EXACTA NORMALIZADA ---
                                                        for i, option in enumerate(select_obj.options):
                                                            txt_opcion = option.text
                                                            if _norm(txt_opcion) == val_norm:
                                                                found_index = i
                                                                break

                                                        # --- PASO 2: BÚSQUEDA PARCIAL NORMALIZADA ---
                                                        if found_index == -1:
                                                            for i, option in enumerate(select_obj.options):
                                                                txt_opcion = option.text

                                                                # PARCHE ANTI-ANIMALES: Si buscamos "otros" y la opción es "animales...", ignorarla
                                                                if val_norm == "otros" and "animales" in _norm(txt_opcion):
                                                                    continue

                                                                if val_norm in _norm(txt_opcion):
                                                                    found_index = i
                                                                    break

                                                        # --- PASO 3: FUZZY MATCH / MAPEOS ESPECIALES ---
                                                        if found_index == -1:
                                                            try:
                                                                opts_norm = [_norm(o.text) for o in select_obj.options]
                                                                # Intentar fuzzy match (umbral moderado)
                                                                match = difflib.get_close_matches(val_norm, opts_norm, n=1, cutoff=0.6)
                                                                if match:
                                                                    matched_norm = match[0]
                                                                    for i, option in enumerate(select_obj.options):
                                                                        if _norm(option.text) == matched_norm:
                                                                            found_index = i
                                                                            break
                                                            except Exception:
                                                                pass

                                                    # --- EJECUTAR SELECCIÓN ---
                                                    if found_index != -1:
                                                        try:
                                                            # Intentar selección por índice
                                                            select_obj.select_by_index(found_index)
                                                        except Exception:
                                                            # Fallback por valor o por JS si index falla
                                                            opt = select_obj.options[found_index]
                                                            val_attr = opt.get_attribute('value') or opt.text
                                                            try:
                                                                select_obj.select_by_value(val_attr)
                                                            except Exception:
                                                                driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('change'))", select_element, val_attr)

                                                        time.sleep(0.12)
                                                        driver.execute_script("arguments[0].style.border='3px solid #2ecc71'", select_element)
                                                        count += 1
                                                        break  # éxito: salir de reintentos
                                                    else:
                                                        # No encontró la opción: si es el último reintento, mostrar debug para subtipo
                                                        if intento_reintento < reintentos - 1:
                                                            time.sleep(0.35)
                                                            continue
                                                        else:
                                                            if k_html == "rectificacion_subtipo":
                                                                print(f"  ⚠ Error: No se pudo seleccionar {k_html} después de {reintentos} intentos")

                                                except Exception as e:
                                                    if intento_reintento < reintentos - 1:
                                                        time.sleep(0.35)
                                                        continue
                                                    else:
                                                        if k_html == "rectificacion_subtipo":
                                                            print(f"  ⚠ Error al procesar {k_html}: {type(e).__name__}")
                                    
                                    print(f"✓ IA completo {count} campos. Esperando que presiones ENVIAR...")
                                    relatos_procesados.add(texto_actual)  # Marcar como procesado
                                else:
                                    print(f"✗ Respuesta inesperada de la API: {resp.status_code}")
                                    
                            except requests.exceptions.Timeout:
                                print(f"✗ Timeout consultando API (más de 5s)")
                            except requests.exceptions.RequestException as e:
                                print(f"✗ Error de conexión con API: {type(e).__name__}")
                            except Exception as e:
                                print(f"✗ Error consultando API: {e}")
                                
                    except Exception as e:
                        # Error al buscar relato_box, simplemente continuar
                        pass
                else:
                    # NO ESTAMOS EN EL FORMULARIO (probablemente tras presionar Enviar)
                    # Simplemente esperar sin hacer nada
                    if contador_ciclos % 20 == 0:  # Mostrar cada 10 segundos aprox
                        print(f"[ESPERANDO] URL actual: {driver.current_url}")
                        
            except Exception as e:
                # Capturar cualquier excepción y continuar sin cerrar
                if "invalid session id" not in str(e):  # No mostrar este error común
                    print(f"Excepción en bucle (ignorada): {type(e).__name__}")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nDETENIDO POR USUARIO. Chrome sigue abierto para que lo cierres manualmente.")
        print("Presiona Alt+F4 o cierra la ventana de Chrome cuando termines.")
        # Mantener Chrome abierto indefinidamente
        try:
            while True:
                time.sleep(1)
        except:
            pass
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    # 1. Arrancar API
    proceso_api = None
    try:
        proceso_api = iniciar_api()
        time.sleep(0.5)  # Dar al menos un tick al proceso
        
        # 2. Esperar que arranque
        try:
            esperar_api()
        except KeyboardInterrupt:
            print("\n⚠ Interrupción durante espera de API. Deteniendo...")
            if proceso_api:
                proceso_api.terminate()
            sys.exit(1)
        
        # 3. Arrancar Navegador
        try:
            bot_navegador()
        except KeyboardInterrupt:
            print("\n\nDETENIDO POR USUARIO.")
            pass
            
    except Exception as e:
        print(f"✗ Error en el flujo principal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nLimpiando recursos...")
        # Matar API al salir
        if proceso_api:
            try:
                proceso_api.terminate()
                proceso_api.wait(timeout=2)
            except:
                try:
                    proceso_api.kill()
                except:
                    pass
        print("✓ Bot detenido correctamente.")
