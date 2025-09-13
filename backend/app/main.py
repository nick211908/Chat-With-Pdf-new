from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import upload, chat
from app.core.logger import setup_logging
from app.core.config import settings
import uvicorn
import os

# --- CHANGE ---
# The lifespan manager is removed. Logging is now called directly.
setup_logging()

app = FastAPI(title="Chat with PDF API")

# CORS Middleware Configuration
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(upload.router, prefix="/api/v1", tags=["PDF Management"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])

@app.get("/", tags=["Health Check"])
def read_root():
    """Health check endpoint."""
    return {"status": "ok"}  

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # default 8000 locally
    uvicorn.run("main:app", host="0.0.0.0", port=port)