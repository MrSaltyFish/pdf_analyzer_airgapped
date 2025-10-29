import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from src.services.pdf_processor import PDFProcessor
from src.services.vector_store import VectorStore
from src.core.logger import get_logger
from src.core.config import UPLOAD_DIR

logger = get_logger(__name__)
router = APIRouter()
pdf_processor = PDFProcessor()
vector_store = VectorStore()

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

@router.post("/single-file")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, processes it into embeddings, and returns a session doc ID."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    unique_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{unique_id}_{file.filename}"

    try:
        # Stream to disk (efficient for large PDFs)
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF → Extract text
        doc_content = pdf_processor.process_pdf(str(temp_path))

        # Store in vector DB (FAISS, Chroma, etc.)
        doc_id = await vector_store.store_document(doc_content, file.filename)

        # Cleanup local copy after success
        # temp_path.unlink(missing_ok=True)

        # Save status (async-safe recommended in vector_store)
        vector_store.processing_status[doc_id] = "completed"

        return {
            "id": doc_id,
            "name": file.filename,
            "status": "completed"
        }

    except Exception as e:
        logger.exception(f"Error processing {file.filename}: {e}")
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Internal PDF processing error")


@router.get("/status/{doc_id}")
async def get_processing_status(doc_id: str):
    """Check current processing status for a given doc_id."""
    try:
        status = vector_store.processing_status.get(doc_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Document ID {doc_id} not found")

        return {"id": doc_id, "status": status}

    except Exception as e:
        logger.exception(f"Status check failed for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal status retrieval error")
