param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# Ubica la carpeta del proyecto
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Rutas y entorno
$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "No se encontro venv. Crea el entorno con: py -3.10 -m venv venv"
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = $Root

$Args = "-m uvicorn api.predict_api:app --host 0.0.0.0 --port $Port"

# Lanza el servidor en segundo plano
$proc = Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $Root -WindowStyle Hidden -PassThru
Write-Host "Servidor iniciando en http://127.0.0.1:$Port (PID $($proc.Id))"

# Espera a que el health responda
$healthUrl = "http://127.0.0.1:$Port/health"
$maxAttempts = 30
for ($i = 1; $i -le $maxAttempts; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 3
        if ($resp.StatusCode -eq 200) {
            Write-Host "API lista (health 200)"
            break
        }
    } catch {
        if ($i -eq $maxAttempts) {
            Write-Warning "No se pudo verificar /health tras $maxAttempts intentos. Revisa el log de Uvicorn."
        }
    }
}

if (-not $NoBrowser) {
    # Abre la página del 911 (EDITA ESTA URL SI ES DISTINTA)
    $SD911_URL = "http://10.100.32.84/SD911/"  # <--- CAMBIA AQUÍ SI TU URL ES DIFERENTE
    Start-Process $SD911_URL
    Write-Host "Se abrio el navegador en: $SD911_URL"
    Write-Host "El bot debería loguearse y empezar a desagregar automáticamente."
}

Write-Host "Para detenerlo: Stop-Process -Id $($proc.Id)"
