from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from src.agents.model_agent import ModelAgent
from src.api.routes import routes_query, routes_docs, routes_upload, routes_vectorDB
from src.core.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="PDF Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only, restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(routes_query.router, prefix="/query", tags=["Query"])
api_router.include_router(routes_docs.router, prefix="/docs", tags=["Docs"])
api_router.include_router(routes_upload.router, prefix="/upload", tags=["Upload"])
api_router.include_router(routes_vectorDB.router, prefix="/vector-db", tags=["VectorDB"])
app.include_router(api_router)

@app.on_event("startup")
def startup_event():
    ModelAgent.initialize()
    logger.info("|> All models initialized and ready.")
