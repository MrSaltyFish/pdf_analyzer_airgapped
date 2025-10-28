from llama_cpp import Llama
from src.core.logger import get_logger
from src.core import config

logger = get_logger(__name__)

class LLMAgent:
    """Handles LLM interactions."""

    def __init__(self):
        logger.info(f"|> Loading LLM model from: {config.LLM_MODEL_PATH}")
        self.model = Llama(
            model_path=str(config.LLM_MODEL_PATH),
            n_ctx=config.LLM_CONTEXT,
            n_threads=config.LLM_THREADS,
            n_batch=config.LLM_BATCH,
        )
        logger.info("|> LLM model initialized successfully.")

    def generate(self, prompt: str, max_tokens=256) -> str:
        """Generate or refine text from prompt."""
        result = self.model(prompt, max_tokens=max_tokens)
        content = (
            result["choices"][0].get("text")
            or result["choices"][0].get("content")
            or ""
        ).strip()
        return content
