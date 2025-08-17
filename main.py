import re
from pathlib import Path
from src.utils.json_object import load_json_as_object, write_object_as_json
from src.models import ModelAgent
from src.pdf_utils import process_and_embed_document
from datetime import datetime
from src.faiss_vectorDB import FAISS_INDEX_PATH, FAISS_METADATA_PATH

VERBOSE = True
def process_collection(path: Path):

    if(VERBOSE):
        print(f"=-=-=-=-=- Processing collection: {path.name} -=-=-=-=-=")

    json_path = path / "challenge1b_input.json"
    input_json = load_json_as_object(json_path)

    pdfs_dir = path / "PDFs"
    persona = input_json.persona.role
    task = input_json.job_to_be_done.task

    ModelAgent.set_user_query(persona, task)
    
    if(VERBOSE):
        print(f"User Persona: {persona}")
        print(f"User Task: {task}")

    for document in input_json.documents:
        filename = document.filename.to_dict()
        pdf_path = pdfs_dir / filename

        if(VERBOSE):
            print(f"Found PDF: {pdf_path}")
        result = process_and_embed_document(pdf_path)
    


def main():

    if(VERBOSE):
        print("Initializing models...")
    ModelAgent.initialize()

    if(VERBOSE):
        print("=========== Models Loaded ============")
        print("Embedding Model:", ModelAgent._embedding_model)
        print("LLM Model:", ModelAgent._llm_model)
        print("======================================")

    input_root = Path("./input")
    for folder in sorted(input_root.glob("Collection *")):
        if folder.is_dir() and re.match(r"Collection \d+", folder.name):
            
            
            if(VERBOSE):
                print("Processing ", folder.name)

            # Saves embeddings inside the FAISS DB
            process_collection(folder)

            results = ModelAgent.query_documents(top_k=5)
            print("==== Top Results for Persona & Task ====")
            for r in results:
                print(f"[{r['rank']}] {r['pdf']} (chunk {r['chunk_id']}) → {r['distance']:.4f}")
                print(r['content'])
                print("----")

                output_json = {
                    "metadata": {
                        "input_documents": [d.filename for d in load_json_as_object(folder / "challenge1b_input.json").documents],
                        "persona": ModelAgent._user_persona,
                        "job_to_be_done": ModelAgent._user_job_to_do,
                        "processing_timestamp": datetime.now().isoformat()
                    },
                    "extracted_sections": results,   # or filter down to the sections you want
                    "subsection_analysis": []        # fill in if you refine further
                }

            
            output_dir = Path("./output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{folder.name}_results.json"


            
            write_object_as_json(str(output_path), output_json)





if __name__ == "__main__":
    for path in [FAISS_INDEX_PATH, FAISS_METADATA_PATH]:
        if path.exists():
            path.unlink()
            print(f"🧹 Deleted old {path.name}")
    print("Running main()")
    main()
