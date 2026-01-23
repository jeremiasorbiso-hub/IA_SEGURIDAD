// ============================================
// SCRIPT DE DEBUG - PEGA EN CONSOLA (F12)
// ============================================

console.log("=" .repeat(60));
console.log("DIAGNOSTICO DEL FORMULARIO SD911");
console.log("=" .repeat(60));

// 1. Verificar URL
console.log("\n1. URL ACTUAL:");
console.log("   " + window.location.href);

// 2. Buscar campos de login
console.log("\n2. CAMPOS DE LOGIN:");

const campos_busqueda = [
    { nombre: "#usuario (ID)", selector: "#usuario" },
    { nombre: "input[name='usuario']", selector: "input[name='usuario']" },
    { nombre: "input[name='nombre']", selector: "input[name='nombre']" },
    { nombre: "input[type='text'] (primero)", selector: "input[type='text']" },
    { nombre: "#password (ID)", selector: "#password" },
    { nombre: "input[name='password']", selector: "input[name='password']" },
    { nombre: "input[type='password']", selector: "input[type='password']" }
];

campos_busqueda.forEach(({ nombre, selector }) => {
    const elem = document.querySelector(selector);
    if (elem) {
        console.log("   ENCONTRADO: " + nombre);
        console.log("      -> ID: " + (elem.id || "sin ID"));
        console.log("      -> name: " + (elem.name || "sin name"));
        console.log("      -> type: " + (elem.type || "sin type"));
        console.log("      -> placeholder: " + (elem.placeholder || "sin placeholder"));
    }
});

// 3. Buscar botón de submit
console.log("\n3. BOTONES:");
const botones = document.querySelectorAll("button");
console.log("   Total botones encontrados: " + botones.length);
botones.forEach((btn, i) => {
    console.log("   Boton " + i + ":");
    console.log("      -> Tipo: " + btn.type);
    console.log("      -> Texto: " + btn.innerText.substring(0, 50));
    console.log("      -> onclick: " + (btn.onclick ? "SI" : "NO"));
    console.log("      -> class: " + btn.className);
});

// 4. Verificar si Tampermonkey está activo
console.log("\n4. TAMPERMONKEY:");
if (typeof GM_xmlhttpRequest !== 'undefined') {
    console.log("   ✓ Tampermonkey ACTIVO");
    console.log("   GM_xmlhttpRequest disponible");
} else {
    console.log("   ✗ Tampermonkey NO ACTIVO");
    console.log("   GM_xmlhttpRequest NO disponible");
}

// 5. Ver si nuestro bot está en la página
console.log("\n5. BOT V6.0:");
if (document.getElementById("bot-status")) {
    console.log("   ✓ Barra del bot PRESENTE");
    console.log("   Contenido: " + document.getElementById("bot-status").innerText.substring(0, 100));
} else {
    console.log("   ✗ Barra del bot NO PRESENTE");
    console.log("   El script de Tampermonkey NO se ejecutó");
}

// 6. Formulario
console.log("\n6. FORMULARIO:");
const formulario = document.querySelector("form");
if (formulario) {
    console.log("   ✓ Formulario encontrado");
    console.log("   -> action: " + formulario.action);
    console.log("   -> method: " + formulario.method);
} else {
    console.log("   ✗ Formulario NO encontrado");
}

// 7. Todos los inputs
console.log("\n7. TODOS LOS INPUTS:");
const inputs = document.querySelectorAll("input");
console.log("   Total: " + inputs.length);
inputs.forEach((inp, i) => {
    console.log("   Input " + i + ": type=" + inp.type + " name=" + inp.name + " id=" + inp.id);
});

console.log("\n" + "=".repeat(60));
console.log("FIN DEL DIAGNOSTICO");
console.log("=".repeat(60));
