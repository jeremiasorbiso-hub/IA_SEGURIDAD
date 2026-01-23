import time
import requests
import subprocess
import sys
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURACIÓN ---
URL_LOGIN = "http://10.100.32.84/SD911/login"
USER = "45657263"
PASS = "911rosario"
API_URL = "http://127.0.0.1:8000"

def iniciar_api():
    """Arranca la API de IA en segundo plano"""
    print("Iniciando Motor de IA (Backend)...")
    if sys.platform == "win32":
        cmd = f"{sys.executable} -m uvicorn api.predict_api:app --host 127.0.0.1 --port 8000"
        return subprocess.Popen(cmd, shell=False)
    else:
        return subprocess.Popen(["python3", "-m", "uvicorn", "api.predict_api:app", "--host", "127.0.0.1", "--port", "8000"])

def esperar_api():
    """Espera a que la API responda"""
    print("Esperando que la IA despierte...")
    for _ in range(20):
        try:
            r = requests.get(f"{API_URL}/health", timeout=1)
            if r.status_code == 200:
                print("IA LISTA PARA TRABAJAR.")
                return True
        except:
            time.sleep(1)
    return False

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
                                resp = requests.post(f"{API_URL}/predict", json={"text": texto_actual})
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
                                            try:
                                                select_element = driver.find_element(By.NAME, k_html)
                                                select_obj = Select(select_element)
                                                
                                                found_index = -1
                                                
                                                # --- PASO 1: BÚSQUEDA EXACTA (Prioridad total) ---
                                                for i, option in enumerate(select_obj.options):
                                                    txt_opcion = option.text.strip().lower()
                                                    if txt_opcion == val_ia:
                                                        found_index = i
                                                        break  # ¡Encontrado exacto!
                                                
                                                # --- PASO 2: BÚSQUEDA PARCIAL (Solo si falló el exacto) ---
                                                if found_index == -1:
                                                    for i, option in enumerate(select_obj.options):
                                                        txt_opcion = option.text.strip().lower()
                                                        
                                                        # 🔥 PARCHE ANTI-ANIMALES: Si buscamos "otros" pero la opción es "animales...", SALTARLA
                                                        if val_ia == "otros" and "animales" in txt_opcion:
                                                            continue
                                                        
                                                        if val_ia in txt_opcion:
                                                            found_index = i
                                                            break
                                                
                                                # --- EJECUTAR SELECCIÓN ---
                                                if found_index != -1:
                                                    select_obj.select_by_index(found_index)
                                                    # Pintar borde verde (feedback visual)
                                                    driver.execute_script("arguments[0].style.border='3px solid #2ecc71'", select_element)
                                                    count += 1
                                            except:
                                                pass
                                    
                                    print(f"✓ IA completo {count} campos. Esperando que presiones ENVIAR...")
                                    relatos_procesados.add(texto_actual)  # Marcar como procesado
                                    
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
                print(f"Excepción en bucle (ignorada): {e}")
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
    proceso_api = iniciar_api()
    
    # 2. Esperar que arranque
    if esperar_api():
        # 3. Arrancar Navegador
        bot_navegador()
    else:
        print("La API no arranco. Revisa errores.")
    
    # Matar API al salir
    proceso_api.terminate()
