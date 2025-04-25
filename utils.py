import PyPDF2
import chromadb
from typing import List
import os

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_chunks(text: str, chunk_size: int = 1500) -> List[str]:
    """Split text into chunks by sentences, trying to keep sentences together."""
    # Split by periods to maintain sentence structure
    sentences = text.replace('\n', ' ').split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip() + '.'
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += ' ' + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

class SimpleChromaDB:
    def __init__(self):
        # Create a directory for persistent storage
        db_dir = "chroma_db"
        os.makedirs(db_dir, exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=db_dir)
        
        try:
            self.collection = self.client.get_collection(name="pdf_docs")
        except:
            self.collection = self.client.create_collection(name="pdf_docs")
    
    def add_documents(self, chunks: List[str]):
        """Add document chunks to the collection."""
        if not chunks:
            return
            
        # Generate simple IDs
        ids = [str(i) for i in range(len(chunks))]
        # Add documents with minimal metadata
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"source": "pdf"} for _ in chunks]
        )
    
    def search(self, query: str, n_results: int = 2) -> List[str]:
        """Search for relevant chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results["documents"][0] 