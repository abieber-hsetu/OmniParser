# --- Hilfsfunktion: Repariert Zeilenenden für Linux-Kompatibilität ---
function Fix-LineEndings {
    Write-Host "`n[1/3] Ueberpruefe Skript-Formate (LF Fix)..." -ForegroundColor Cyan
    # Wir suchen im gesamten omnibox-Verzeichnis nach .sh Dateien
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
    $baseDir = Resolve-Path "$scriptPath\.."
    
    $shFiles = Get-ChildItem -Path $baseDir -Filter *.sh -Recurse
    foreach ($file in $shFiles) {
        $content = Get-Content -Raw $file.FullName
        if ($content -match "`r`n") {
            Write-Host "  -> Repariere: $($file.Name)" -ForegroundColor Yellow
            $content -replace "`r`n", "`n" | Set-Content -NoNewline $file.FullName
        }
    }
    Write-Host "  -> Alle Linux-Skripte sind nun im LF-Format." -ForegroundColor Green
}

# --- Funktion: Erstellt die VM und den Server ---
function Create-VM {
    # 1. Wir suchen den absoluten Pfad zum omnibox-Ordner (ein Ordner über scripts)
    $scriptDir = $PSScriptRoot
    if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition }
    $baseDir = (Get-Item "$scriptDir\..").FullName
    
    Write-Host "Arbeitsverzeichnis: $baseDir" -ForegroundColor Gray

    # 2. Fix-LineEndings aufrufen
    Fix-LineEndings

    # 3. Build-Prozess im richtigen Verzeichnis starten
    Set-Location $baseDir
    Write-Host "Starte Build..." -ForegroundColor Cyan
    docker build -t windows-local .

    # 4. Compose im richtigen Verzeichnis starten
    Write-Host "Starte Compose..." -ForegroundColor Cyan
    docker compose up -d

    Wait-ForServer
}

# --- Funktion: Wartet auf den Health-Check des Servers ---
function Wait-ForServer {
    Write-Host "`n--- Warte auf den Computer Control Server ---" -ForegroundColor White
    Write-Host "HINWEIS: Laut OmniParser kann das erste Setup bis zu 90 Minuten dauern." -ForegroundColor Cyan
    
    $url = "http://localhost:5000/probe"
    $startTime = Get-Date

    while ($true) {
        try {
            # Versuche den Server zu erreichen
            $response = Invoke-WebRequest -Uri $url -Method GET -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $duration = New-TimeSpan -Start $startTime -End (Get-Date)
                Write-Host "`n[!] Server ist nach $($duration.Minutes) Minuten bereit!" -ForegroundColor Green
                return
            }
        } catch {
            # Ausgabe alle 5 Minuten (300 Sekunden)
            $elapsed = New-TimeSpan -Start $startTime -End (Get-Date)
            Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] Seit $($elapsed.Minutes) Min. aktiv: Windows wird im Container noch initialisiert..." -ForegroundColor Gray
            
            # 5 Minuten warten
            Start-Sleep -Seconds 300 
        }
    }
}

# --- Funktion: Startet bestehende VM ---
function Start-LocalVM {
    Write-Host "Starte VM..." -ForegroundColor Cyan
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
    docker compose -f "$scriptPath\..\compose.yml" start
    Wait-ForServer
    Write-Host "VM gestartet." -ForegroundColor Green
}

# --- Funktion: Stoppt die VM ---
function Stop-LocalVM {
    Write-Host "Stoppe VM..." -ForegroundColor Yellow
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
    docker compose -f "$scriptPath\..\compose.yml" stop
    Write-Host "VM gestoppt." -ForegroundColor Gray
}

# --- Funktion: Loescht alles ---
function Remove-VM {
    Write-Host "Entferne VM und Netzwerk..." -ForegroundColor Red
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
    docker compose -f "$scriptPath\..\compose.yml" down
    Write-Host "Alles entfernt." -ForegroundColor Gray
}

# --- Haupt-Logik (Argument-Verarbeitung) ---
if (-not $args[0]) {
    Write-Host "`nNutzung: powershell -File manage_vm.ps1 [create|start|stop|delete]" -ForegroundColor Magenta
    exit 1
}

switch ($args[0]) {
    "create" { Create-VM }
    "start"  { Start-LocalVM }
    "stop"   { Stop-LocalVM }
    "delete" { Remove-VM }
    default {
        Write-Host "Ungueltige Option: $($args[0])" -ForegroundColor Red
        exit 1
    }
}