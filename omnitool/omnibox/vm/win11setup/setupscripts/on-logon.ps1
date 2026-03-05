$scriptFile = "Z:\server\main.py"
$port = 5050

cd "Z:\"
Write-Host "Starte OmniBox Agent auf Port $port..."
python $scriptFile --port $port