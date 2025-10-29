import tempfile
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from src.services.pdf_processor import PDFProcessor
from src.services.vector_store import VectorStore
from src.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
pdf_processor = PDFProcessor()
vector_store = VectorStore()

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Save file to temp location
        temp_file = UPLOAD_DIR / f"temp_{file.filename}"
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        doc_content = pdf_processor.process_pdf(str(temp_file))
        doc_id = await vector_store.store_document(doc_content, file.filename)

        # Cleanup
        temp_file.unlink()

        return {
            "id": doc_id,
            "name": file.filename,
            "status": "processing"
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{doc_id}")
async def get_processing_status(doc_id: str):
    """Get the processing status of a document."""
    try:
        if not hasattr(vector_store, 'processing_status'):
            raise HTTPException(status_code=404, detail="Status tracking not initialized")
            
        status = vector_store.processing_status.get(doc_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Document ID {doc_id} not found")
            
        return {"status": status}
    except Exception as e:
        logger.error(f"Status check failed for {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
