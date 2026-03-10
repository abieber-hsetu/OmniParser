import os
from dotenv import load_dotenv
from rag_manager import HsetuRagManager

# 1. Lade die Keys aus der .env
load_dotenv() 

# 2. Prüfe zur Sicherheit, ob der Key geladen wurde (nur zum Debuggen)
key = os.getenv("OPENAI_API_KEY")
if not key or "dein-key" in key:
    print("FEHLER: Der API-Key wurde nicht korrekt aus der .env geladen!")
else:
    # 3. Wenn alles okay ist, starte den Manager
    rag = HsetuRagManager()
    rag.update_database(docs_folder="./docs")