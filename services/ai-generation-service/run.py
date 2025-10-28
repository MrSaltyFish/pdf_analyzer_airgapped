import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Insurance Policy Summarization API                          ║
    ║  Fine-tuned Mistral 7B GGUF Model (Refactored)               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",  # Points to the app object in app/main.py
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )