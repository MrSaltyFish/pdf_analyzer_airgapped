import time
from typing import Optional, List, Dict
from datetime import datetime
from llama_cpp import Llama
from app.core.config import settings
from app.models.api_models import QueryType

class ModelManager:
    """Manages model lifecycle and inference"""
    
    def __init__(self, config: settings):
        self.config = config
        self.model: Optional[Llama] = None
        self.load_time: Optional[datetime] = None
        self.total_requests = 0
        
    def load_model(self):
        """Load GGUF model with optimized settings"""
        print(f"🔄 Loading model from: {self.config.MODEL_PATH}")
        start = time.time()
        
        try:
            self.model = Llama(
                model_path=self.config.MODEL_PATH,
                n_ctx=self.config.N_CTX,
                n_threads=self.config.N_THREADS,
                n_gpu_layers=self.config.N_GPU_LAYERS,
                verbose=self.config.DEBUG
            )
        except Exception as e:
            print(f"❌ FAILED to load model: {e}")
            print("Please ensure the MODEL_PATH in your .env file is correct and the file exists.")
            return

        self.load_time = datetime.now()
        elapsed = time.time() - start
        print(f"✅ Model loaded successfully in {elapsed:.2f}s")
        print(f"   - Context window: {self.config.N_CTX}")
        print(f"   - GPU layers: {self.config.N_GPU_LAYERS}")
        print(f"   - CPU threads: {self.config.N_THREADS}")
    
    def create_chat_prompt(self, 
                           policy_text: str, 
                           new_query: str, 
                           history: List[Dict[str, str]],
                           query_type: Optional[QueryType] = None) -> str:
        """
        Create optimized prompt including conversation history
        for follow-up questions.
        """
        
        # Query type specific instructions
        instructions = {
            QueryType.DESCRIPTIVE: "Provide a comprehensive summary of the key features and benefits.",
            QueryType.DETAIL: "Extract specific details and structured information clearly.",
            QueryType.COMPARATIVE: "Compare the options and provide a recommendation with reasoning.",
            QueryType.CONCEPT: "Explain the concept in simple terms that anyone can understand.",
            QueryType.COVERAGE: "List the coverage limits, exclusions, and conditions clearly.",
            QueryType.FINANCIAL: "Summarize the financial terms, premiums, and payment conditions."
        }
        
        instruction = instructions.get(query_type, "Provide a clear and accurate answer based *only* on the policy document and conversation history.")
        
        # Build history string
        history_str = ""
        if history:
            history_str = "--- Conversation History ---\n"
            for msg in history:
                role = "User" if msg['role'] == 'user' else 'Answer'
                history_str += f"{role}: {msg['content']}\n"
            history_str += "--- End History ---\n\n"

        # Truncate policy text to save context window space
        # We assume the model can find details in ~8k characters
        truncated_policy = policy_text[:8000]

        prompt = f"""### Insurance Policy Document:
{truncated_policy}

{history_str}### New Question:
{new_query}

### Instructions:
{instruction}

### Answer:"""
        
        return prompt
    
    def generate(self, prompt: str, temperature: float, 
                 max_tokens: int, stream: bool = False):
        """Generate response from model"""
        if not self.model:
            raise RuntimeError("Model is not loaded.")
            
        self.total_requests += 1
        
        return self.model(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.config.TOP_P,
            stream=stream,
            stop=["###", "User:", "Question:"] # Stop tokens
        )

# Single instance for the app
model_manager = ModelManager(config=settings)