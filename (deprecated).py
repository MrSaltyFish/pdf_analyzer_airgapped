import os
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import glob
import re
import multiprocessing
# Core dependencies
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Lightweight semantic model (under 200mb)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# GGUF Model for refining extracted text
from llama_cpp import Llama

@dataclass
class ProcessedSection:
    document: str
    section_title: str
    content: str
    page_number: int
    embedding: np.ndarray
    importance_rank: int = 0

class PDFEncoder:
    def __init__(self, model_path: str = "./models/all-MiniLM-L12-v2"):
        """
        Initialize with a lightweight semantic model (~90MB)
        all-MiniLM-L6-v2 is fast, accurate, and well under 1GB
        """
        threads = multiprocessing.cpu_count() // 2  # or 3/4th if thermals allow

        self.model = SentenceTransformer(model_path)
        self.processed_sections: List[ProcessedSection] = []
        self.llm = Llama(   model_path="models/jina-embeddings-v4-text-retrieval-IQ2_XXS.gguf",  # replace with your GGUF path
                            n_ctx=512,
                            n_threads=threads,  # tune for your CPU - 8
                            n_batch=64,          # tune for speed-vs-memory tradeoff
                            use_mlock=True,      # optional: locks model in RAM for faster response
                            use_mmap=True,       # recommended for CPU
                            n_gpu_layers=0       # ❗ this ensures **CPU-only** mode
                        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF into chunks"""
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=768,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", ""]
        )
        
        return text_splitter.split_documents(docs)
    
    def extract_section_title(self, chunk: Document) -> str:
        """Extract or generate section title from chunk"""
        content = chunk.page_content.strip()
        lines = content.split('\n')
        
        # Look for title-like patterns
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if len(line) > 0 and (
                line.isupper() or 
                line.endswith(':') or 
                (len(line.split()) <= 8 and not line.endswith('.'))
            ):
                return line.rstrip(':')
        
        # Fallback: use first sentence or first 50 chars
        first_sentence = content.split('.')[0]
        if len(first_sentence) <= 50:
            return first_sentence
        return content[:50] + "..."
    
    def extract_section_title_llama(self, chunk: Document) -> str:
        """Use LLaMA to extract or generate a meaningful section title from chunk content"""
        content = chunk.page_content.strip()
        prompt = f"""You're given a section of a document.
    \"\"\"
    {content}
    \"\"\"

    Title:"""
        
        # Call LLaMA model with the prompt
        title = self.query_llama(prompt)

        # Clean and return
        return title.strip().rstrip(':')
    
    def query_llama(self, prompt: str) -> str:
        """Run the prompt on the LLaMA model and return the generated title."""
        output = self.llm(prompt)
        return output["choices"][0]["text"]  # or output["choices"][0]["text"].strip()

    
    def encode_documents(self, document_configs: List[Dict[str, str]], base_dir = './input') -> None:
        """Process and encode all PDF documents"""
        self.processed_sections = []
        
        for doc_config in document_configs:
            filename = doc_config["filename"]
            full_path = os.path.join(base_dir, "PDFs", filename)
            title = doc_config["title"]
            
            if not os.path.exists(full_path):
                print(f"Warning: {full_path} not found, skipping...")
                continue
            
            print(f"================== Processing {full_path}... =================")

            chunks = self.process_pdf(full_path)
            
            for chunk in chunks:
                # if len(chunk.page_content.split()) < 100:
                section_title = self.extract_section_title(chunk)  # fast heuristic
                # else:
                #     section_title = self.extract_section_title_llama(chunk)  # LLaMA-powered

                content = chunk.page_content
                page_number = chunk.metadata.get('page', 1)
                
                # Generate embedding
                embedding = self.model.encode(content)
                
                section = ProcessedSection(
                    document=filename,
                    section_title=section_title,
                    content=content,
                    page_number=page_number + 1,
                    embedding=embedding
                )
                
                self.processed_sections.append(section)
                
    def clean_text(self, text: str) -> str:
        # Remove extra newlines and spaces
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\b(?:Confidential|Draft|Internal Use Only)\b', '', text, flags=re.IGNORECASE)

        # Remove unicode artifacts, bullet chars etc.
        text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219]', '', text)

        # Remove sequences of special characters
        text = re.sub(r'[-=]{3,}', '', text)
        text = re.sub(r'\.{3,}', '.', text)

        # Strip leading/trailing whitespace
        return text.strip()
    
    def refine_with_llm(self, text: str) -> str:
        prompt = f"""You are an expert editor. Clean and refine the following extracted document section to improve clarity and coherence. Remove repetition and irrelevant details, but preserve technical accuracy. Output only the refined section.

        --- BEGIN TEXT ---
        {text}
        --- END TEXT ---
    """
        output = self.llm(prompt, max_tokens=1024, stop=["--- END"])
        return output["choices"][0]["text"].strip()

    
    def retrieve_relevant_sections(self, query_config: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Retrieve most relevant sections based on persona and job to be done"""
        
        # Create query from persona and job
        persona = query_config["persona"]["role"]
        job = query_config["job_to_be_done"]["task"]
        
        # Combine persona and job into search query
        search_query = f"{persona}: {job}"
        
        # Encode the query
        query_embedding = self.model.encode(search_query)
        
        # Calculate similarities
        similarities = []
        for section in self.processed_sections:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                section.embedding.reshape(1, -1)
            )[0][0]
            similarities.append((section, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top sections
        top_sections = similarities[:top_k]
        
        # Build response in required format
        result = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in query_config["documents"]],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
        
        # Add extracted sections with importance ranking
        for i, (section, similarity) in enumerate(top_sections):
            raw = section.content
            # regexed_text = self.clean_text(raw)
            refined = self.refine_with_llm(raw)

            result["extracted_sections"].append({
                "document": section.document,
                "section_title": section.section_title,
                "importance_rank": i + 1,
                "page_number": section.page_number
            })
            
            # Add subsection analysis
            result["subsection_analysis"].append({
                "document": section.document,
                "refined_text": refined,
                "page_number": section.page_number
            })
        
        return result

class TravelPlannerRetrieval:
    """Main class for travel planning document retrieval"""
    
    def __init__(self):
        self.encoder = PDFEncoder()
    
    def process_and_retrieve(self, query_config: Dict[str, Any], base_dir) -> Dict[str, Any]:
        """Main method to process documents and retrieve relevant sections"""
        
        # Encode all documents
        document_configs = query_config["documents"]
        self.encoder.encode_documents(document_configs, base_dir=base_dir)
        
        # Retrieve relevant sections
        results = self.encoder.retrieve_relevant_sections(query_config, top_k=5)
        
        return results

# Example usage

def main():
    base_path = "./input"

    

    # base_path = Path(base_path)
    # all_pdfs = []

    # for collection_dir in base_path.iterdir():
    #     if collection_dir.is_dir():
    #         pdfs = list(collection_dir.glob("*.pdf"))
    #         all_pdfs.extend(pdfs)

    pattern = os.path.join(base_path, "Collection */challenge1b_input.json")
    input_files = glob.glob(pattern)

    for input_path in input_files:
        collection_dir = os.path.dirname(input_path)

        with open(input_path, 'r') as f:
            query_config = json.load(f)

        # Process the input using your pipeline
        retrieval_system = TravelPlannerRetrieval()
        results = retrieval_system.process_and_retrieve(query_config, base_dir=collection_dir)

        # Define output path
        output_path = os.path.join(os.path.dirname(input_path), "challenge1b_output_appp.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[✓] Processed: {input_path} → {output_path}")

if __name__ == "__main__":
    print("Running main")  # or app.run(), cli(), whatever
    main()

# Dependencies to install:
# pip install langchain pymupdf sentence-transformers scikit-learn numpy