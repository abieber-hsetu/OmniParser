import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import fitz
import re
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

class HsetuRagManager:
    def __init__(self, db_path="./hsetu_knowledge"):
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        if os.path.exists(self.db_path):
            self.vector_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            print(f"✅ Bestehende RAG-Datenbank geladen aus {self.db_path}")
        else:
            self.vector_db = None
            print(f"⚠️ Keine Datenbank unter {self.db_path} gefunden. Bitte update_docs.py ausführen.")
    
    def parse_instruction_pdf(self, pdf_path):
        if not pdf_path or not os.path.exists(pdf_path):
            return {"program_name": None, "steps": []}
        
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            lines = md_text.split("\n")
            
            # --- 1. Programmnamen extrahieren ---
            program_name = None
            for line in lines:
                clean = line.strip().replace("#", "").replace("*", "")
                if clean: # Die erste nicht-leere Zeile ist meist der Titel
                    program_name = clean
                    break
            
            all_steps = []
            idx_tp, idx_soll = -1, -1
            
            # --- 2A. Versuch: Markdown-Tabelle parsen (Altes Format) ---
            for line in lines:
                if "|" in line:
                    cells = [c.strip() for c in line.split("|") if c.strip() != ""]
                    l_cells = [c.lower() for c in cells]

                    if any("testpunkt" in c for c in l_cells) and any("soll" in c for c in l_cells):
                        for i, c in enumerate(l_cells):
                            if "testpunkt" in c: idx_tp = i
                            if "soll" in c: idx_soll = i
                        continue

                    if idx_tp != -1 and idx_soll != -1 and len(cells) > max(idx_tp, idx_soll):
                        if all(re.match(r'^[ :\-_]*$', c) for c in cells):
                            continue
                        
                        tp_text = cells[idx_tp].replace("**", "").replace("__", "").strip()
                        soll_text = cells[idx_soll].replace("**", "").replace("__", "").strip()
                        
                        if len(tp_text) > 3 and "testpunkt" not in tp_text.lower():
                            all_steps.append(f" {tp_text} |  {soll_text}")

            # --- 2B. NEU: Fallback für nummerierte Listen (Neues Format) ---
            # Wenn die Tabelle oben nichts gefunden hat, suchen wir nach Listen!
            if not all_steps:
                print("ℹ️ Keine Tabelle gefunden. Lade nummerierte Liste (mit Zeilenumbruch-Erkennung)...")
                
                current_step_text = ""
                
                for line in lines:
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    
                    # Nur für die Prüfung putzen wir störende Zeichen weg
                    check_text = clean_line.replace("**", "").replace("__", "").replace("\\", "")
                    
                    # 1. Ist es ein GANZ NEUER Schritt? (Muss zwingend mit einer Zahl starten, z.B. "1. " oder "49) ")
                    is_new_step = re.match(r'^(\d+[\.\)])\s+', check_text)
                    
                    if is_new_step:
                        # Den vorherigen, fertigen Schritt in die Liste packen (falls vorhanden)
                        if current_step_text and len(current_step_text) > 5:
                            all_steps.append(current_step_text.strip())
                            
                        # Den neuen Schritt starten
                        current_step_text = clean_line.replace("**", "").replace("__", "")
                        # Eventuell escapete Punkte (wie 49\.) reparieren
                        current_step_text = re.sub(r'^(\d+)\\\.', r'\1.', current_step_text)
                        
                    elif current_step_text:
                        # 2. Es ist eine FORTSETZUNGSZEILE! Wir hängen sie an den aktuellen Schritt an.
                        add_text = clean_line.replace("**", "").replace("__", "")
                        
                        # Falls (wie in deinem Screenshot) ein Bindestrich durch den Umbruch an den Zeilenanfang gerutscht ist, löschen wir ihn weg
                        add_text = re.sub(r'^-?\s*', '', add_text) 
                        
                        # Mit einem Leerzeichen an den vorherigen Satzteil kleben
                        current_step_text += " " + add_text
                        
                # 3. Den allerletzten Schritt am Ende der Schleife nicht vergessen!
                if current_step_text and len(current_step_text) > 5:
                    all_steps.append(current_step_text.strip())

            print(f"✅ PyMuPDF4LLM hat {len(all_steps)} Schritte gefunden.")
            return {"program_name": program_name, "steps": all_steps}

        except Exception as e:
            print(f"❌ Fehler bei PyMuPDF4LLM: {str(e)}")
            return {"program_name": None, "steps": []}

    def update_database(self, docs_folder="./docs"):
        """Taggt Dokumente basierend auf dem Dateinamen und nutzt Markdown-Chunking."""
        print(f"🚀 Starte Dokumenten-Update aus {docs_folder}...")
        
        if not os.path.exists(docs_folder):
            print(f"⚠️ Ordner {docs_folder} existiert nicht!")
            return

        all_chunks = []
        
        # 1. Wir definieren, an welchen Markdown-Überschriften getrennt werden soll
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # Fallback-Splitter, falls ein Kapitel doch mal riesig ist (z.B. > 1500 Zeichen)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

        # 2. Dateien manuell durchgehen und mit pymupdf4llm verarbeiten
        for filename in os.listdir(docs_folder):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(docs_folder, filename)
                program_name = os.path.splitext(filename)[0]
                print(f"📄 Verarbeite {filename} für Programm: {program_name}")
                
                try:
                    # Markdown-Extraktion (erhält Tabellen, Listen, Fettgedrucktes)
                    md_text = pymupdf4llm.to_markdown(file_path)
                    
                    # Semantischer Split anhand der Überschriften
                    md_header_splits = markdown_splitter.split_text(md_text)
                    
                    # Zu große Chunks nochmals sanft teilen
                    splits = text_splitter.split_documents(md_header_splits)
                    
                    # Metadaten injizieren
                    for split in splits:
                        split.metadata["source"] = file_path
                        split.metadata["program"] = program_name
                        
                    all_chunks.extend(splits)
                except Exception as e:
                    print(f"❌ Fehler beim Verarbeiten von {filename}: {e}")

        if not all_chunks:
            print("⚠️ Keine neuen Inhalte zum Indizieren gefunden.")
            return

        # 3. Datenbank sicher updaten
        if self.vector_db:
            # Wenn DB schon existiert, fügen wir die neuen Chunks einfach hinzu
            self.vector_db.add_documents(all_chunks)
        else:
            # Falls sie komplett neu ist
            self.vector_db = Chroma.from_documents(
                documents=all_chunks, 
                embedding=self.embeddings, 
                persist_directory=self.db_path
            )
            
        print(f"✅ Datenbank aktualisiert. {len(all_chunks)} semantische Chunks hinzugefügt.")


    def get_context(self, query, program=None, k=3):
        # 1. Datentyp-Check
        if isinstance(query, list):
            query = " ".join([str(q) for q in query])
        
        if not query or str(query).strip() == "":
            return "Keine Suchanfrage für den Kontext angegeben."

        # 2. Datenbank-Check
        if not self.vector_db:
            print("⚠️ RAG-Manager: Anfrage erhalten, aber vector_db ist None.")
            return "Hinweis: Die Wissensdatenbank ist aktuell nicht geladen."

        # 3. Filter-Logik
        search_filter = None
        if program and program != "Unknown":
            search_filter = {"program": program}
            print(f"🔍 RAG-Suche für Programm: {program} | Query: {query}")
        else:
            print(f"🔍 RAG-Suche (global) | Query: {query}")

        try:
            # 4. OPTIMIERUNG: Maximal Marginal Relevance (MMR) statt normaler Suche
            # MMR holt nicht einfach die 3 ähnlichsten Chunks (die oft identisch sind),
            # sondern balanciert Ähnlichkeit mit Diversität. So bekommt der Agent einen
            # viel breiteren Kontext zum Problem geliefert.
            results = self.vector_db.max_marginal_relevance_search(
                query, 
                k=k, 
                fetch_k=k*3, # Holt intern 9 Ergebnisse und wählt die 3 diversesten aus
                filter=search_filter
            )

            if not results:
                return f"Keine spezifischen Informationen in der Datenbank zu '{query}' gefunden."

            # 5. Ergebnisse formatieren (Jetzt mit Header-Kontext!)
            formatted_results = []
            for i, res in enumerate(results):
                source = res.metadata.get('source', 'Unbekannt')
                prog = res.metadata.get('program', 'Allgemein')
                
                # Wenn der Markdown-Splitter eine Überschrift gefunden hat, zeigen wir sie an!
                header_info = ""
                for h_level in ["Header 1", "Header 2", "Header 3"]:
                    if h_level in res.metadata:
                        header_info += f" -> {res.metadata[h_level]}"
                
                header_str = f" (Kapitel:{header_info})" if header_info else ""
                
                content = f"--- Dokument-Ausschnitt {i+1} [Programm: {prog}{header_str}] ---\n{res.page_content}"
                formatted_results.append(content)
            
            return "\n\n".join(formatted_results)

        except Exception as e:
            print(f"❌ Kritischer Fehler bei der RAG-Abfrage: {e}")
            return f"Fehler bei der Wissensabfrage: {str(e)}"