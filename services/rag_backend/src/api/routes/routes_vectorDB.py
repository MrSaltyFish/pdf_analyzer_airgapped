from fastapi import APIRouter, HTTPException
import os
import shutil
from src.core.logger import get_logger
from src.core.config import FAISS_DIR, UPLOAD_DIR

router = APIRouter()
logger = get_logger(__name__)

@router.delete("/delete-all")
async def delete_faiss_index():
    """
    Deletes all FAISS vector index files and uploaded PDFs safely.
    - Keeps the directories intact.
    - Recreates them if missing.
    """
    try:
        # 🧩 Clean FAISS directory
        if os.path.exists(FAISS_DIR):
            if os.path.isdir(FAISS_DIR):
                for item in os.listdir(FAISS_DIR):
                    item_path = os.path.join(FAISS_DIR, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                logger.info(f"✅ Cleared FAISS directory: {FAISS_DIR}")
            else:
                os.remove(FAISS_DIR)
                os.makedirs(FAISS_DIR, exist_ok=True)
                logger.info(f"✅ Recreated FAISS directory after deleting file: {FAISS_DIR}")
        else:
            os.makedirs(FAISS_DIR, exist_ok=True)
            logger.info(f"✅ Created FAISS directory: {FAISS_DIR}")

        # 🧩 Clean UPLOAD directory
        if os.path.exists(UPLOAD_DIR):
            if os.path.isdir(UPLOAD_DIR):
                for item in os.listdir(UPLOAD_DIR):
                    item_path = os.path.join(UPLOAD_DIR, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                logger.info(f"✅ Cleared uploads directory: {UPLOAD_DIR}")
            else:
                os.remove(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                logger.info(f"✅ Recreated uploads directory after deleting file: {UPLOAD_DIR}")
        else:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            logger.info(f"✅ Created uploads directory: {UPLOAD_DIR}")

        return {
            "status": "success",
            "message": "FAISS index and uploaded PDF contents cleared successfully.",
            "paths": {
                "faiss": str(FAISS_DIR),
                "uploads": str(UPLOAD_DIR),
            },
        }

    except Exception as e:
        logger.error(f"❌ Error clearing FAISS or uploads directory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
