from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.agents.model_agent import ModelAgent
from src.api.routes import routes_query, routes_docs

from src.core.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="PDF Analyzer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(routes_query.router, prefix="/query")
app.include_router(routes_docs.router, prefix="/docs")


@app.on_event("startup")
def startup_event():
    ModelAgent.initialize()
    logger.info("|> All models initialized and ready.")