// ==UserScript==
// @name         SD911 - BOT MAESTRO V6.1 (FINAL)
// @namespace    http://10.100.32.84/
// @version      6.1
// @description  Login Agresivo + Fix Animales + Fix Tiempo Subtipo
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

    // --- 1. CONFIGURACIÓN ---
    const CONFIG = {
        USERNAME: "35584499",
        PASSWORD: "cargador",
        API_URL: "http://127.0.0.1:8000"
    };

    let logs = [];

    // --- 2. BARRA DE ESTADO VISUAL ---
    function showStatus(msg, color) {
        logs.push("[" + new Date().toLocaleTimeString() + "] " + msg);
        let div = document.getElementById("bot-status");
        if (!div) {
            div = document.createElement("div");
            div.id = "bot-status";
            div.style = "position:fixed; top:0; left:0; width:100%; max-height:25vh; padding:10px; z-index:99999; font-weight:bold; color:white; font-family:monospace; font-size:11px; overflow-y:auto; border-bottom:3px solid black;";
            document.body.appendChild(div);
        }
        div.style.background = color;
        div.innerHTML = "<strong>[BOT V6.1 FINAL]</strong><br>" + logs.slice(-6).join("<br>");
        console.log("[BOT]", msg);
    }

    // --- 3. LOGICA DE LOGIN (FUERZA BRUTA) ---
    function forceLogin() {
        showStatus("🔐 INICIANDO LOGIN...", "#d32f2f");
        const loginAttempt = setInterval(() => {
            let user = document.getElementById("usuario") || document.querySelector("input[name='nombre']");
            let pass = document.getElementById("password") || document.querySelector("input[name='password']");
            let btn = document.querySelector("button[type='submit']") || document.querySelector(".btn-info");

            if (user && pass && btn) {
                showStatus("✅ CAMPOS ENCONTRADOS. INYECTANDO...", "#1976d2");
                
                // Inyectar valores
                user.value = CONFIG.USERNAME;
                pass.value = CONFIG.PASSWORD;
                
                // Disparar eventos para engañar al navegador
                [user, pass].forEach(f => {
                    f.dispatchEvent(new Event('input', { bubbles: true }));
                    f.dispatchEvent(new Event('change', { bubbles: true }));
                    f.dispatchEvent(new Event('blur', { bubbles: true }));
                });

                setTimeout(() => {
                    showStatus("🚀 CLICKEANDO INGRESAR...", "#388e3c");
                    btn.click();
                    
                    // Reintento de seguridad
                    setTimeout(() => {
                        if(document.getElementById("usuario")) {
                            const form = document.querySelector("form");
                            if(form) form.submit(); // Submit forzado si el click falla
                        }
                    }, 1000);
                }, 500);
                clearInterval(loginAttempt);
            }
        }, 1000);
    }

    // --- 4. LOGICA DE NAVEGACION ---
    function forceMenu() {
        showStatus("🗺️ BUSCANDO BOTON CARGAR...", "#7b1fa2");
        const menuLoop = setInterval(() => {
            const btn = document.querySelector("button[onclick*='form911auto']");
            if (btn) {
                clearInterval(menuLoop);
                showStatus("➡️ YENDO AL FORMULARIO...", "#388e3c");
                btn.click();
                
                // Redirección forzada de respaldo
                setTimeout(() => {
                    if (!window.location.href.includes("form911auto")) {
                        window.location.href = window.location.origin + '/SD911/form911auto';
                    }
                }, 1500);
            }
        }, 1000);
    }

    // --- 5. LOGICA DE FORMULARIO ---
    function forceForm() {
        showStatus("📝 ESPERANDO RELATO...", "#0277bd");
        const formLoop = setInterval(() => {
            const relato = document.querySelector("textarea[name='relato']");
            if (relato && relato.value.trim().length > 5) {
                clearInterval(formLoop);
                showStatus("🧠 TEXTO DETECTADO. CONSULTANDO IA...", "#00838f");
                llamarIA(relato.value);
            }
        }, 1500);
    }

    // --- 6. CONEXIÓN API ---
    function llamarIA(texto) {
        GM_xmlhttpRequest({
            method: "POST",
            url: CONFIG.API_URL + "/predict",
            headers: { "Content-Type": "application/json" },
            data: JSON.stringify({ text: texto, incidente_id: "WEB-" + Date.now() }),
            onload: function(response) {
                if (response.status == 200) {
                    const data = JSON.parse(response.responseText);
                    const preds = data.best_predictions || data.predictions;
                    showStatus("⚡ IA RESPONDIÓ. LLENANDO CAMPOS...", "#2e7d32");
                    llenarSelects(preds);
                } else {
                    showStatus("❌ ERROR API: " + response.status, "#d32f2f");
                }
            },
            onerror: function() { showStatus("❌ ERROR DE CONEXIÓN (Check Run_API)", "#d32f2f"); }
        });
    }

    // --- 7. INTELIGENCIA DE LLENADO (CRÍTICO) ---
    function llenarSelects(preds) {
        if (!preds) return;

        let count = 0;
        const mapa = {
            "cObjetivo": "objetivo", "cMedioempleado": "medio", "cModusoperandi": "modus",
            "cMedios_fuga": "fuga", "cElementos_sustraidos": "sustraido", "cLocalizacion": "localiz",
            "cGenero_Sexo": "sexo_genero", "cEdad": "edadvictima",
            "cRectificacion_Tipo": "rectificacion_tipo"
            // SUBTIPO SE PROCESA APARTE
        };

        // A) Llenar campos inmediatos
        for (const [kIA, nameHTML] of Object.entries(mapa)) {
            if (aplicarSeleccion(kIA, nameHTML, preds[kIA])) count++;
        }

        // B) Llenar Subtipo con RETRASO (Fix Timing)
        if (preds["cRectificacion_Subtipo"]) {
            showStatus("⏳ ESPERANDO CARGA DE SUBTIPO...", "#ff8f00");
            setTimeout(() => {
                if (aplicarSeleccion("cRectificacion_Subtipo", "rectificacion_subtipo", preds["cRectificacion_Subtipo"])) {
                    showStatus("✅ SUBTIPO APLICADO.", "#2e7d32");
                }
            }, 1000); // 1 segundo de espera
        }
        
        showStatus("🏁 PROCESO FINALIZADO (" + count + " campos).", "#1b5e20");
    }

    function aplicarSeleccion(keyIA, nameHTML, valorIA) {
        const select = document.querySelector(`select[name='${nameHTML}']`);
        if (!select || !valorIA) return false;

        const valNorm = String(valorIA).toUpperCase().trim();
        let found = false;

        for (let i = 0; i < select.options.length; i++) {
            const optText = String(select.options[i].text).toUpperCase().trim();
            const optVal = String(select.options[i].value).toUpperCase().trim();

            // --- FIX: PROBLEMA ANIMALES (OTROS) ---
            // Si la IA dice "OTROS", ignoramos cualquier opción que diga "ANIMALES"
            if (valNorm === "OTROS" && optText.includes("ANIMALES")) {
                continue; 
            }

            // --- LOGICA DE COINCIDENCIA ---
            // 1. Coincidencia exacta de VALUE (Ej: TENTATIVA)
            // 2. Coincidencia exacta de TEXTO
            // 3. Coincidencia parcial de TEXTO
            if (optVal === valNorm || optText === valNorm || optText.includes(valNorm)) {
                select.selectedIndex = i;
                select.value = select.options[i].value;
                
                // Disparar eventos
                select.dispatchEvent(new Event('change', { bubbles: true }));
                select.dispatchEvent(new Event('click', { bubbles: true }));
                
                // Estilo visual
                select.style.border = "3px solid #2ecc71";
                select.style.backgroundColor = "#e8f5e9";
                found = true;
                break;
            }
        }
        return found;
    }

    // --- 8. ARRANCAR ---
    const url = window.location.href;
    setTimeout(() => {
        if (url.includes("login") || document.getElementById("usuario")) {
            forceLogin();
        } else if (url.includes("form911auto") || document.querySelector("textarea[name='relato']")) {
            forceForm();
        } else {
            forceMenu();
        }
    }, 800);

})();