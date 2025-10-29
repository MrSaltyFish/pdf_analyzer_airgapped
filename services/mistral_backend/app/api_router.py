from fastapi import APIRouter

from app.endpoints import chat, batch, admin

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(chat.router)
api_router.include_router(batch.router)
api_router.include_router(admin.router)