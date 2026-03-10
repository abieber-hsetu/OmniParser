import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
# Laden des env-Files für den API-Key
load_dotenv()

class HsetuRagManager:

    def __init__(self, db_path="./hsetu_knowledge"):
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_db = None

    def update_database(self, docs_folder="./docs"):
        """Liest alle Dateien aus dem Ordner docs/ und aktualisiert die DB."""
        print(f"🚀 Starte Dokumenten-Update aus {docs_folder}...")
        
        # 1. Dokumente laden (PDF, Word, Text)
        loader = DirectoryLoader(docs_folder, glob="**/*.*", 
                                 loader_cls=PyPDFLoader) # Erweitert für Word/Text
        docs = loader.load()
        
        # 2. In mundgerechte Stücke schneiden (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        # 3. Vektordatenbank erstellen oder aktualisieren
        self.vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.db_path
        )
        print(f"✅ Update abgeschlossen. {len(chunks)} Wissens-Chunks gespeichert.")

    def get_context(self, query, k=3):
        """Sucht die passendsten Informationen zu einer Anfrage."""
        if not self.vector_db:
            self.vector_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        
        results = self.vector_db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])