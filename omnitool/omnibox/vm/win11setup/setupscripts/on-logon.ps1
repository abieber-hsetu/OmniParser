# --- KONFIGURATION ---
$driveZ = "Z:"
$shareZ = "\\172.18.0.1\Data"

$user = "docker"
$pass = "docker"
$scriptFile = "Z:\server\main.py"
$port = 5050

Write-Host "Setze DNS-Server auf 8.8.8.8..." -ForegroundColor Yellow
Get-NetAdapter | Where-Object Status -eq 'Up' | Set-DnsClientServerAddress -ServerAddresses '8.8.8.8', '1.1.1.1'

Write-Host "--- OmniBox Agent: Network Drive Mode ---" -ForegroundColor Cyan

# Funktion für Netzwerk-Mount
function Connect-NetworkDrive {
    param([string]$letter, [string]$path)
    
    Write-Host "Bereinige alte unsichtbare Verbindungen..." -ForegroundColor DarkGray
    net use * /delete /y 2>$null
    # Gezielt die versteckte Systemverbindung zur IP killen:
    net use \\172.18.0.1\IPC$ /delete /y 2>$null 
    
    Start-Sleep -Seconds 1
    
    Write-Host "Verbinde Netzwerk-Laufwerk $letter mit $path..." -ForegroundColor Yellow
    net use $letter $path /user:$user $pass /persistent:no
}

# --- HAUPTSCHLEIFE ---
$maxAttempts = 15
$attempt = 1
$connected = $false

while (-not $connected -and $attempt -le $maxAttempts) {
    Write-Host "Verbindungsversuch $attempt von $maxAttempts..."
    
    # Z: über das Netzwerk verbinden
    Connect-NetworkDrive -letter $driveZ -path $shareZ

    # Prüfen, ob Z: da ist (wir testen, ob das Python-Skript erreichbar ist)
    if (Test-Path $scriptFile) {
        $connected = $true
        Write-Host "Erfolg! Laufwerk Z: ist bereit." -ForegroundColor Green
    } else {
        Write-Host "Warte auf Netzwerk/Datei: $scriptFile..." -ForegroundColor Magenta
        Start-Sleep -Seconds 5
        $attempt++
    }
}

# --- EXECUTION ---
if ($connected) {
    Write-Host "Starte Python App auf Port $port..." -ForegroundColor Cyan
    Set-Location -Path "Z:\"
    try {
        python $scriptFile --port $port
    } catch {
        Write-Host "Kritischer Fehler beim Ausführen von Python!" -ForegroundColor Red
        $_.Exception.Message
    }
} else {
    Write-Host "!!! ABBRUCH: Z: Laufwerk oder Skript wurde nicht gefunden !!!" -ForegroundColor Red
}