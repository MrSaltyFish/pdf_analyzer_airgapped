from pathlib import Path
import fitz
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter

from .models import ModelAgent
from .utils.json_object import write_object_as_json
from .faiss_vectorDB import save_embedded_document_in_faiss

def process_and_embed_document(pdf_path: Path):
    print(f">>>> Processing: {pdf_path.name}")

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
        chunk_size=1000,
        chunk_overlap=200,
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
    print(f">>>> Saved embeddings to: {output_path}")

    return output