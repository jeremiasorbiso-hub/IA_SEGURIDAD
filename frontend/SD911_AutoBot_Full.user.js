// ==UserScript==
// @name         SD911 - BOT ULTRA AGRESIVO V6.0
// @namespace    http://10.100.32.84/
// @version      6.0
// @description  Login 100% agresivo con DEBUG en pantalla
// @author       IA Seguridad Team
// @match        http://10.100.32.84/SD911/*
// @connect      127.0.0.1
// @connect      localhost
// @connect      10.100.32.84
// @grant        GM_xmlhttpRequest
// @grant        GM_log
// ==/UserScript==

(function() {
    'use strict';

    // --- CONFIGURACIÓN ---
    const CONFIG = {
        USERNAME: "45657263",
        PASSWORD: "911rosario",
        API_URL: "http://127.0.0.1:8000"
    };

    let logs = [];

    // --- BARRA DE ESTADO VISUAL CON LOGS ---
    function showStatus(msg, color) {
        logs.push("[" + new Date().toLocaleTimeString() + "] " + msg);
        
        let div = document.getElementById("bot-status");
        if (!div) {
            div = document.createElement("div");
            div.id = "bot-status";
            div.style = "position:fixed; top:0; left:0; width:100%; max-height:30vh; padding:10px; z-index:99999; font-weight:bold; color:white; font-family:monospace; font-size:11px; overflow-y:auto; border-bottom:3px solid black;";
            document.body.appendChild(div);
        }
        div.style.background = color;
        div.innerHTML = "<strong>[BOT V6.0 ULTRA]</strong><br>" + logs.slice(-8).join("<br>");
        console.log("[BOT V6.0]", msg);
    }


    // --- FASE 1: LOGIN (ULTRA AGRESIVA) ---
    function forceLogin() {
        showStatus("1. INICIANDO LOGIN...", "#d32f2f");
        
        const loginAttempt = setInterval(() => {
            // Buscar campos por TODOS los métodos posibles
            let user = document.getElementById("usuario") || 
                      document.querySelector("input[id='usuario']") ||
                      document.querySelector("input[name='usuario']") ||
                      document.querySelector("input[placeholder*='usuario']") ||
                      document.querySelector("input[placeholder*='Usuario']");
                      
            let pass = document.getElementById("password") || 
                      document.querySelector("input[id='password']") ||
                      document.querySelector("input[name='password']") ||
                      document.querySelector("input[type='password']") ||
                      document.querySelector("input[placeholder*='password']");
            
            let btn = document.querySelector("button[type='submit']") || 
                      document.querySelector(".btn-primary") || 
                      document.querySelector(".btn-info") ||
                      document.querySelector("button:contains('Ingresar')") ||
                      document.querySelector("button");

            showStatus("Buscando campos... Usuario:" + (user ? "SI" : "NO") + " Pass:" + (pass ? "SI" : "NO") + " Btn:" + (btn ? "SI" : "NO"), "#ff6f00");

            if (user && pass && btn) {
                showStatus("2. CAMPOS ENCONTRADOS! Inyectando credenciales...", "#1976d2");
                
                // METODO 1: Directa
                user.value = CONFIG.USERNAME;
                pass.value = CONFIG.PASSWORD;
                
                // METODO 2: Eventos
                [user, pass].forEach(field => {
                    field.dispatchEvent(new Event('input', { bubbles: true }));
                    field.dispatchEvent(new Event('change', { bubbles: true }));
                    field.dispatchEvent(new Event('blur', { bubbles: true }));
                    field.dispatchEvent(new Event('focus', { bubbles: true }));
                    
                    // Atributo (por si valida así)
                    field.setAttribute('value', field.value);
                });

                // METODO 3: Triggear validación del formulario
                const form = document.querySelector("form");
                if (form) {
                    form.dispatchEvent(new Event('change', { bubbles: true }));
                }

                setTimeout(() => {
                    showStatus("3. CLICKEANDO BOTON DE INGRESO...", "#388e3c");
                    btn.click();
                    
                    // SEGUNDO INTENTO si falla
                    setTimeout(() => {
                        if (document.getElementById("usuario")) {
                            showStatus("3b. REINTENTANDO CLICK...", "#ff9800");
                            btn.click();
                            
                            // TERCER INTENTO: Submit directo
                            setTimeout(() => {
                                if (document.getElementById("usuario") && form) {
                                    showStatus("3c. ENVIANDO FORM DIRECTAMENTE...", "#f57c00");
                                    form.submit();
                                }
                            }, 1000);
                        }
                    }, 1500);
                }, 500);

                clearInterval(loginAttempt);
            }
        }, 800);
    }


    // --- FASE 2: NAVEGACIÓN AGRESIVA ---
    function forceMenu() {
        showStatus("4. BUSCANDO BOTON CARGAR...", "#7b1fa2");
        
        const menuLoop = setInterval(() => {
            let btn = document.querySelector("button[onclick*='form911auto']") ||
                     document.querySelector("button[onclick*='911auto']") ||
                     document.querySelector("a[href*='form911auto']") ||
                     document.querySelector("[onclick*='form911auto']");
            
            if (btn) {
                clearInterval(menuLoop);
                showStatus("5. BOTON ENCONTRADO! CLICKEANDO...", "#388e3c");
                btn.click();
                
                setTimeout(() => {
                    if (!window.location.href.includes("form911auto")) {
                        showStatus("5b. FORZANDO REDIRECCION...", "#d32f2f");
                        window.location.href = window.location.origin + '/SD911/form911auto';
                    }
                }, 1500);
            } else {
                showStatus("Buscando boton... (Si ves esto mucho, revisamos HTML)", "#ff6f00");
            }
        }, 1000);
    }

    // --- FASE 3: LLENAR FORMULARIO ---
    function forceForm() {
        showStatus("6. BUSCANDO RELATO EN FORMULARIO...", "#0277bd");
        
        const formLoop = setInterval(() => {
            const relato = document.querySelector("textarea[name='relato']") ||
                          document.querySelector("textarea");
            
            if (relato) {
                const textoLength = relato.value ? relato.value.length : 0;
                showStatus("Relato encontrado (" + textoLength + " caracteres)", "#1565c0");
                
                if (textoLength > 5) {
                    clearInterval(formLoop);
                    showStatus("7. TEXTO LISTO! ENVIANDO A LA IA...", "#00838f");
                    llamarIA(relato.value);
                }
            }
        }, 1500);
    }


    // --- CONEXIÓN A IA ---
    function llamarIA(texto) {
        showStatus("8. CONECTANDO A IA EN " + CONFIG.API_URL + "...", "#1976d2");
        
        GM_xmlhttpRequest({
            method: "POST",
            url: CONFIG.API_URL + "/predict",
            headers: { "Content-Type": "application/json" },
            data: JSON.stringify({ text: texto, incidente_id: "AUTO-" + Date.now() }),
            timeout: 30000,
            
            onload: function(response) {
                console.log("Response status:", response.status);
                console.log("Response text:", response.responseText);
                
                if (response.status == 200) {
                    showStatus("9. IA RESPONDIO! PROCESANDO...", "#00897b");
                    try {
                        const data = JSON.parse(response.responseText);
                        const preds = data.best_predictions || data.predictions;
                        showStatus("10. LLENANDO CAMPOS CON PREDICCIONES...", "#2e7d32");
                        llenarSelects(preds);
                    } catch (e) {
                        showStatus("ERROR parseando JSON: " + e.message, "#d32f2f");
                    }
                } else {
                    showStatus("ERROR: API devolvio " + response.status, "#d32f2f");
                    console.log("Full response:", response.responseText);
                }
            },
            
            onerror: function(error) {
                showStatus("ERROR CONEXION: Ventana negra NO corre", "#d32f2f");
                console.log("XHR Error:", error);
            }
        });
    }

    function llenarSelects(preds) {
        if (!preds) {
            showStatus("ERROR: No hay predicciones!", "#d32f2f");
            return;
        }

        let count = 0;
        const mapa = {
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

        for (const [kIA, nameHTML] of Object.entries(mapa)) {
            const select = document.querySelector(`select[name='${nameHTML}']`);
            
            if (select && preds[kIA]) {
                const valIA = String(preds[kIA]).toLowerCase().trim();
                let found = false;
                
                for (let i = 0; i < select.options.length; i++) {
                    const opcionText = select.options[i].text.toLowerCase().trim();
                    
                    // Busca coincidencia exacta o parcial
                    if (opcionText === valIA || opcionText.includes(valIA) || valIA.includes(opcionText.substring(0, 5))) {
                        select.selectedIndex = i;
                        select.value = select.options[i].value;
                        
                        // Disparar eventos
                        select.dispatchEvent(new Event('change', { bubbles: true }));
                        select.dispatchEvent(new Event('click', { bubbles: true }));
                        select.dispatchEvent(new Event('blur', { bubbles: true }));
                        
                        // Indicador visual
                        select.style.border = "3px solid #4caf50";
                        select.style.backgroundColor = "#e8f5e9";
                        
                        count++;
                        found = true;
                        console.log("LLENO: " + kIA + " = " + preds[kIA]);
                        break;
                    }
                }
                
                if (!found) {
                    console.warn("NO ENCONTRE OPCION PARA: " + kIA + " = " + preds[kIA]);
                }
            }
        }
        
        showStatus("11. COMPLETO! " + count + " campos llenos. GUARDAR AHORA!", "#1b5e20");
    }

    // --- DETECCIÓN Y ACTIVACIÓN ---
    showStatus("BOT ACTIVO EN: " + window.location.href, "#e91e63");
    
    const url = window.location.href;
    console.log("URL actual:", url);
    console.log("Usuario encontrado:", !!document.getElementById("usuario"));
    console.log("Formulario encontrado:", !!document.querySelector("textarea[name='relato']"));
    
    // Esperar a que el DOM esté listo
    setTimeout(() => {
        if (url.includes("login") || document.getElementById("usuario")) {
            console.log(">>> ENTRANDO EN MODO LOGIN");
            forceLogin();
        } else if (url.includes("form911auto") || document.querySelector("textarea[name='relato']")) {
            console.log(">>> ENTRANDO EN MODO FORMULARIO");
            forceForm();
        } else {
            console.log(">>> ENTRANDO EN MODO MENU");
            forceMenu();
        }
    }, 1000);

})();