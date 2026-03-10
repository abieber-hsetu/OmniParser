# --- KONFIGURATION ---
$driveLetter = "Z:"
# WICHTIG: Nutze die Gateway-IP 10.0.2.2 und den Samba-Freigabenamen aus deiner Docker-Config (meist 'data')
$sharePath = "\\10.0.2.2\data" 
$user = "docker"
$pass = "docker"
$scriptFile = "Z:\server\main.py"
$port = 5050

Write-Host "--- OmniBox Agent: Persistent Boot Mode ---" -ForegroundColor Cyan

# Funktion zum sauberen Trennen und Neu-Verbinden
function Connect-Drive {
    Write-Host "Versuche Laufwerk $driveLetter mit $sharePath zu verbinden..." -ForegroundColor Yellow
    
    # 1. Altes Laufwerk hart entfernen (falls vorhanden)
    net use $driveLetter /delete /y 2>$null
    Start-Sleep -Seconds 2

    # 2. Neu verbinden mit Anmeldedaten
    # /persistent:no ist wichtig, damit wir beim nächsten Skriptstart keine Konflikte haben
    net use $driveLetter $sharePath /user:$user $pass /persistent:no
}

# --- HAUPTSCHLEIFE ---
$maxAttempts = 15
$attempt = 1
$connected = $false

while (-not $connected -and $attempt -le $maxAttempts) {
    Write-Host "Samba-Verbindungsversuch $attempt von $maxAttempts..."
    
    Connect-Drive

    # Prüfen ob die Datei nun wirklich da ist
    if (Test-Path $scriptFile) {
        $connected = $true
        Write-Host "Verbindung hergestellt! Datei gefunden." -ForegroundColor Green
    } else {
        Write-Host "Pfad noch nicht erreichbar. Warte 5 Sekunden..." -ForegroundColor Magenta
        Start-Sleep -Seconds 5
        $attempt++
    }
}

# --- EXECUTION ---
if ($connected) {
    Write-Host "Starte Python App auf Port $port..." -ForegroundColor Cyan
    # Wechsel ins Verzeichnis, damit Python relative Pfade im Skript findet
    Set-Location -Path "Z:\"
    
    # Python-Start mit Fehlerprüfung
    try {
        python $scriptFile --port $port
    } catch {
        Write-Host "Kritischer Fehler beim Ausführen von Python!" -ForegroundColor Red
        $_.Exception.Message
    }
} else {
    Write-Host "!!! ABBRUCH: Z: konnte nach $maxAttempts Versuchen nicht gemountet werden !!!" -ForegroundColor Red
    Write-Host "Prüfe: Läuft der Samba-Container? Ist 10.0.2.2 anpingbar?" -ForegroundColor Gray
}