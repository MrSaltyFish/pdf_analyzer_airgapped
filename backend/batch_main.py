import re
from pathlib import Path
from datetime import datetime

from src.utils.json_object import load_json_as_object, write_object_as_json
from src.agents.model_agent import ModelAgent
from src.services.pdf_processor import process_and_embed_document
from src.core import config
from src.core.logger import get_logger

logger = get_logger(__name__)

def process_collection(agent: ModelAgent, path: Path):

    logger.info(f"Processing collection: {path.name}")
    json_path = path / "user_input.json"
    input_json = load_json_as_object(json_path)

    pdfs_dir = path / "PDFs"
    persona = input_json.persona.role
    task = input_json.job_to_be_done.task

    agent.set_user_query(persona, task)

    logger.info(f"User Persona: {persona}")
    logger.info(f"User Task: {task}")

    for document in input_json.documents:
        filename = document.filename.to_dict()
        pdf_path = pdfs_dir / filename

        logger.info(f"Found PDF: {pdf_path}")
        process_and_embed_document(pdf_path)


def main():
    
    logger.info("Initializing models...")
    
    agent = ModelAgent.instance()

    for folder in sorted(config.INPUT_DIR.glob("Collection *")):
        if folder.is_dir() and re.match(r"Collection \d+", folder.name):
            
            logger.info(f"Processing {folder.name}")

            process_collection(folder)
            results = agent.query_documents(top_k=config.TOP_K_RESULTS)

            logger.debug("==== Top Results for Persona & Task ====")
            for r in results:
                logger.debug(f"[{r['rank']}] {r['pdf']} (chunk {r['chunk_id']}) → {r['distance']:.4f}")
                logger.debug(r['content'])

            output_json = {
                "metadata": {
                    "input_documents": [d.filename for d in load_json_as_object(folder / "user_input.json").documents],
                    "persona": ModelAgent._user_persona,
                    "job_to_be_done": ModelAgent._user_job_to_do,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": results,
                "subsection_analysis": []
            }

            config.OUTPUT_DIR.mkdir(exist_ok=True)
            output_path = config.OUTPUT_DIR / f"{folder.name}_results.json"
            write_object_as_json(str(output_path), output_json)


if __name__ == "__main__":
    if config.CLEAR_OLD_FAISS:
        for path in [config.FAISS_INDEX_PATH, config.FAISS_METADATA_PATH]:
            if path.exists():
                path.unlink()
                logger.warning(f"!!! Deleted old {path.name}")

    logger.info("Running main()")
    main()
