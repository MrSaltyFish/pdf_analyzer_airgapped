import re
from pathlib import Path
from datetime import datetime

from src.utils.json_object import load_json_as_object, write_object_as_json
from src.services.model_agent import ModelAgent
from src.services.pdf_processor import process_and_embed_document
from src.core.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    TOP_K_RESULTS,
    VERBOSE,
    CLEAR_OLD_FAISS
)
from src.core.logger import get_logger

logger = get_logger(__name__)

def process_collection(path: Path):
    if VERBOSE:
        logger.info(f"Processing collection: {path.name}")

    json_path = path / "challenge1b_input.json"
    input_json = load_json_as_object(json_path)

    pdfs_dir = path / "PDFs"
    persona = input_json.persona.role
    task = input_json.job_to_be_done.task

    ModelAgent.set_user_query(persona, task)

    if VERBOSE:
        logger.info(f"User Persona: {persona}")
        logger.info(f"User Task: {task}")

    for document in input_json.documents:
        filename = document.filename.to_dict()
        pdf_path = pdfs_dir / filename

        if VERBOSE:
            logger.info(f"Found PDF: {pdf_path}")
        process_and_embed_document(pdf_path)


def main():
    if VERBOSE:
        logger.info("Initializing models...")
    ModelAgent.initialize()

    if VERBOSE:
        logger.info("=========== Models Loaded ============")
        logger.info(f"Embedding Model: {ModelAgent._embedding_model}")
        logger.info(f"LLM Model: {ModelAgent._llm_model}")
        logger.info("======================================")

    for folder in sorted(INPUT_DIR.glob("Collection *")):
        if folder.is_dir() and re.match(r"Collection \d+", folder.name):
            if VERBOSE:
                logger.info(f"Processing {folder.name}")

            process_collection(folder)
            results = ModelAgent.query_documents(top_k=TOP_K_RESULTS)

            logger.debug("==== Top Results for Persona & Task ====")
            for r in results:
                logger.debug(f"[{r['rank']}] {r['pdf']} (chunk {r['chunk_id']}) → {r['distance']:.4f}")
                logger.debug(r['content'])

            output_json = {
                "metadata": {
                    "input_documents": [d.filename for d in load_json_as_object(folder / "challenge1b_input.json").documents],
                    "persona": ModelAgent._user_persona,
                    "job_to_be_done": ModelAgent._user_job_to_do,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": results,
                "subsection_analysis": []
            }

            OUTPUT_DIR.mkdir(exist_ok=True)
            output_path = OUTPUT_DIR / f"{folder.name}_results.json"
            write_object_as_json(str(output_path), output_json)


if __name__ == "__main__":
    if CLEAR_OLD_FAISS:
        for path in [FAISS_INDEX_PATH, FAISS_METADATA_PATH]:
            if path.exists():
                path.unlink()
                logger.warning(f"!!! Deleted old {path.name}")

    logger.info("Running main()")
    main()
