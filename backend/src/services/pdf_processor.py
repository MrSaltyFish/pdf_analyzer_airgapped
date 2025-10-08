from pathlib import Path
import fitz
import numpy as np
import PyPDF2
from typing import Dict, List
import hashlib
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.agents.model_agent import ModelAgent
from src.utils.json_object import write_object_as_json
from .vector_store import save_embedded_document_in_faiss

from src.core.logger import get_logger
logger = get_logger(__name__)

def process_and_embed_document(pdf_path: Path):
    logger.debug(f">>>> Processing: {pdf_path.name}")

    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text_recursive(raw_text, metadata={"source": pdf_path.name})
    embedded_chunks = ModelAgent.embed_chunks(chunks)

    output = save_embedded_document_in_json(pdf_path, embedded_chunks)
    
        # --- NEW: push into FAISS ---
    vectors = [chunk["vector"] for chunk in embedded_chunks]
    metadatas = [
        {
            "pdf": pdf_path.name,
            "chunk_id": chunk["metadata"]["chunk_id"],
            "content": chunk["content"],
            "metadata": chunk["metadata"],
        }
        for chunk in embedded_chunks
    ]
    save_embedded_document_in_faiss(vectors, metadatas)

    return {"pdf_path": str(pdf_path), "chunks": embedded_chunks}

def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text_recursive(text: str, metadata: dict) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        # separators=["\n\n", "\n", " ", ""]
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    chunks = splitter.split_text(text)
    
    return [
        {
            "content": chunk,
            "metadata": {
                **metadata,
                "chunk_id": idx
            }
        }
        for idx, chunk in enumerate(chunks)
    ]

def save_embedded_document_in_json(pdf_path: Path, embedded_chunks: list[dict]):
    output = {
        "pdf_path": str(pdf_path),
        "chunks": embedded_chunks
    }

    # Save as JSON to project root / temp
    temp_dir = Path(__file__).resolve().parent.parent / "vectorDB_data/JSON"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"{pdf_path.stem}_embedding.json"

    write_object_as_json(str(output_path), output)
    logger.debug(f">>>> Saved embeddings to: {output_path}")

    return output

class PDFProcessor:
    MAX_CHUNK_SIZE = 500  # Maximum characters per chunk
    MAX_CHUNKS = 500      # Maximum number of chunks to process
    
    def process_pdf(self, file_path: str) -> Dict:
        try:
            chunks = []
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Process pages until we hit max chunks
                for page in reader.pages:
                    if len(chunks) >= self.MAX_CHUNKS:
                        break
                        
                    text = page.extract_text()
                    page_chunks = self._create_chunks(text)
                    chunks.extend(page_chunks)
                    
                    # Check chunk limit
                    if len(chunks) > self.MAX_CHUNKS:
                        chunks = chunks[:self.MAX_CHUNKS]
                        break

            return {
                "chunks": chunks
            }
                
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def _create_chunks(self, text: str) -> List[str]:
        """Create smaller, memory-efficient chunks."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in text.split('.'):
            sentence = sentence.strip() + '.'
            if current_size + len(sentence) > self.MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks