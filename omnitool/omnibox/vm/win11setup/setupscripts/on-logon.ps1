# --- KONFIGURATION ---
$scriptDir = "C:\setup" 
$scriptFile = "C:\setup\server\main.py"
$port = 5050
$localInstaller = "C:\OmniParser\python-3.10.11-amd64.exe"

Write-Host "Setze DNS-Server auf 8.8.8.8..." -ForegroundColor Yellow
Get-NetAdapter | Where-Object Status -eq 'Up' | Set-DnsClientServerAddress -ServerAddresses '8.8.8.8', '1.1.1.1'
ipconfig /flushdns | Out-Null

Write-Host "--- OmniBox Agent: Standalone Mode ---" -ForegroundColor Cyan

# --- PYTHON PFADE ERMITTELN ---
$pythonExe = "C:\Program Files\Python310\python.exe"
$pythonUserExe = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"

# --- PYTHON CHECK & LOKALE INSTALLATION ---
if (-not (Test-Path $pythonExe) -and -not (Test-Path $pythonUserExe)) {
    Write-Host "Echtes Python nicht gefunden!" -ForegroundColor Yellow
    
    if (Test-Path $localInstaller) {
        Write-Host "Lokaler Installer gefunden. Installiere Python (silent)... Bitte warten." -ForegroundColor Yellow
        Start-Process -FilePath $localInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait -NoNewWindow
        
        Start-Sleep -Seconds 5
        Write-Host "Python Installation abgeschlossen!" -ForegroundColor Green
    } else {
        Write-Host "!!! FEHLER: Python fehlt und die Datei $localInstaller wurde nicht gefunden !!!" -ForegroundColor Red
        Start-Sleep -Seconds 15
        exit
    }
} else {
    Write-Host "Python ist bereits installiert." -ForegroundColor Green
}

# Welcher Pfad ist jetzt aktiv?
if (Test-Path $pythonExe) { $activePython = $pythonExe }
elseif (Test-Path $pythonUserExe) { $activePython = $pythonUserExe }
else { $activePython = "python" }

# --- DEPENDENCIES INSTALLIEREN (NEU) ---
Write-Host "Installiere OmniParser Abhängigkeiten..." -ForegroundColor Yellow
Set-Location -Path $scriptDir
& $activePython -m pip install -r "$scriptDir/server/requirements.txt"

# --- EXECUTION ---
if (Test-Path $scriptFile) {
    Write-Host "Starte Python App auf Port $port..." -ForegroundColor Green
    
    # Python direkt ausführen (ohne Try/Catch, damit Fehler sichtbar ins Terminal laufen)
    & $activePython $scriptFile --port $port
    
    Write-Host " "
    Write-Host "Der OmniParser wurde beendet oder ist abgestürzt." -ForegroundColor Yellow
    Read-Host "Drücke Enter, um das Fenster zu schließen..."
} else {
    Write-Host "!!! ABBRUCH: Datei $scriptFile wurde nicht gefunden !!!" -ForegroundColor Red
    Read-Host "Drücke Enter, um das Fenster zu schließen..."
}