# --- KONFIGURATION ---
$driveZ = "Z:"
$shareZ = "\\10.0.2.2\Data" 

$driveT = "T:"
$testDataDir = "Z:\testmuster" # Der Pfad auf dem bereits gemounteten Z:

$user = "docker"
$pass = "docker"
$scriptFile = "Z:\server\main.py"
$port = 5050

Write-Host "--- OmniBox Agent: Persistent Multi-Drive Mode (SUBST) ---" -ForegroundColor Cyan

# Funktion für Netzwerk-Mount (Nur für Z:)
function Connect-NetworkDrive {
    param([string]$letter, [string]$path)
    Write-Host "Verbinde Netzwerk-Laufwerk $letter mit $path..." -ForegroundColor Yellow
    net use $letter /delete /y 2>$null
    Start-Sleep -Seconds 1
    net use $letter $path /user:$user $pass /persistent:no
}

# --- HAUPTSCHLEIFE ---
$maxAttempts = 15
$attempt = 1
$connected = $false

while (-not $connected -and $attempt -le $maxAttempts) {
    Write-Host "Verbindungsversuch $attempt von $maxAttempts..."
    
    # 1. Z: über das Netzwerk verbinden
    Connect-NetworkDrive -letter $driveZ -path $shareZ

    # 2. Prüfen, ob Z: da ist UND der Unterordner existiert
    if (Test-Path $testDataDir) {
        Write-Host "Basis-Pfad $testDataDir gefunden. Erstelle virtuelles Laufwerk $driveT..." -ForegroundColor Yellow
        
        # T: über SUBST lokal von Z: abspalten
        subst $driveT /d 2>$null # Alten Subst löschen
        subst $driveT $testDataDir
        
        if (Test-Path "$driveT\") {
            $connected = $true
            Write-Host "Erfolg! Z: (Netz) und T: (Subst) sind bereit." -ForegroundColor Green
        }
    } else {
        Write-Host "Warte auf Netzwerk/Ordner: $testDataDir..." -ForegroundColor Magenta
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
    Write-Host "!!! ABBRUCH: Testdaten-Pfad wurde nicht gefunden !!!" -ForegroundColor Red
}